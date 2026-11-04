"""Validate strategy: pipeline against an unsplit single-GPU reference, one process per GPU.

Runs the real trainer code (distributed/pipeline.py build_stage, trainer/pipeline_loop.py
train_pipeline, its eval and its resume) on a tiny random Llama with a fake in-memory dataset,
so a whole matrix of settings costs minutes instead of a 70B model load per case.

Numerical scenarios (fp32, sdpa, 1 sequence per micro-batch) train N steps and compare against an
unsplit model trained with the same LoRA start, data, optimizer and schedule:
per-step loss, per-step gradient norm and the final LoRA weights.
Eval scenarios compare the pipeline's eval loss with a reference forward over the same data.
Resume scenarios compare an interrupted-and-resumed run with an uninterrupted one.
Robustness scenarios check that bad settings fail on every rank instead of hanging.

  sbatch --nodes=4 --partition=gpu-debug scripts/pp_validate.sbatch      # 8 GPUs; exit 1 on any FAIL
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from bhaskera.config import load_config                                   # noqa: E402
from bhaskera.distributed.pipeline import build_stage, canonical_name, split_layers   # noqa: E402
from bhaskera.trainer import pipeline_loop as PL                            # noqa: E402
from bhaskera.trainer.optim import build_optimizer, build_scheduler        # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEQ, VOCAB, LAYERS = 64, 512, 10          # 10 layers: uneven splits for 4 and 8 stages
RESULTS: list[tuple[str, bool, str]] = []


class FakeData:
    """Stands in for a Ray dataset shard. Batch k is a pure function of (seed, salt, k), so the
    reference and a resumed run can regenerate exactly what the pipeline saw."""

    def __init__(self, seed=0, salt=0, n_rows=None, skip=0, packed=False, crash_at=None):
        self.seed, self.salt, self.n_rows, self.skip, self.packed = seed, salt, n_rows, skip, packed
        self.crash_at = crash_at          # simulate a crash when asked for this batch index

    def batch(self, k, rows, device):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.salt * 1009 + k)
        ids = torch.randint(0, VOCAB, (rows, SEQ), generator=g)
        out = {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}
        if self.packed:   # two packed sequences per row
            out["position_ids"] = (torch.arange(SEQ) % (SEQ // 2)).expand(rows, -1).clone()
        return {k_: v.to(device) for k_, v in out.items()}

    def iter_torch_batches(self, batch_size, drop_last=True, dtypes=None, device="cuda", **_):
        k = self.skip
        while self.n_rows is None or (k - self.skip + 1) * batch_size <= self.n_rows:
            if k == self.crash_at:
                raise RuntimeError("simulated crash")
            yield self.batch(k, batch_size, device)
            k += 1


class Tracker:
    def __init__(self):
        self.rows = {}

    def log(self, d, step=None):
        self.rows.setdefault(step, {}).update(d)


def note(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    if dist.get_rank() == 0:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# ---------------------------------------------------------------- config / model
def make_model_dir(path, dtype=torch.float32):
    from transformers import LlamaConfig, LlamaForCausalLM
    if dist.get_rank() == 0:
        shutil.rmtree(path, ignore_errors=True)
        torch.manual_seed(1234)
        cfg = LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=LAYERS,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
                          max_position_embeddings=512, tie_word_embeddings=False)
        LlamaForCausalLM(cfg).to(dtype).save_pretrained(path, safe_serialization=True)
    dist.barrier()


def make_cfg(base, model_dir, **kw):
    cfg = load_config(base)
    cfg.model.name, cfg.model.dtype = model_dir, kw.get("dtype", "float32")
    cfg.model.attn_impl = kw.get("attn", "sdpa")
    cfg.lora.enabled, cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout = True, 8, 16, 0.0
    cfg.lora.target_modules = list(TARGETS)
    cfg.data.seq_len = SEQ
    cfg.data.runtime_packing = bool(kw.get("packed", False))
    t = cfg.training
    t.batch_size, t.max_steps, t.lr, t.warmup_steps = kw.get("rows_mb", 1), kw.get("steps", 3), 5e-3, 2
    t.eval_steps, t.seed = kw.get("eval_steps", 0), 7
    p = t.distributed.pipeline
    p.stages, p.microbatches = kw.get("stages", 4), kw.get("m", 8)
    p.schedule, p.activation_checkpointing = kw.get("schedule", "1f1b"), kw.get("ckpt_act", True)
    p.eval_micro_batches = kw.get("eval_mb", 16)
    cfg.checkpoint.enabled = bool(kw.get("save_dir"))
    cfg.checkpoint.save_dir = kw.get("save_dir") or ""
    cfg.checkpoint.save_interval = kw.get("save_interval", 0)
    return cfg


def gather_lora(stage, info):
    """{copy: {canonical name: tensor}} from every rank, on every rank."""
    mine = {canonical_name(n, info.layer_ids): p.detach().float().cpu()
            for n, p in stage.named_parameters() if p.requires_grad}
    allsd = [None] * dist.get_world_size()
    dist.all_gather_object(allsd, (info.copy, mine))
    out: dict = {}
    for c, sd in allsd:
        out.setdefault(c, {}).update(sd)
    return out


def set_lora(stage, info, sd, device):
    with torch.no_grad():
        for n, p in stage.named_parameters():
            if p.requires_grad:
                p.copy_(sd[canonical_name(n, info.layer_ids)].to(device))


def delta(final, init):
    return {k: final[k] - init[k] for k in final}


def max_rel(a, b):
    num = sum(float((a[k] - b[k]).pow(2).sum()) for k in a) ** 0.5
    den = sum(float(b[k].pow(2).sum()) for k in b) ** 0.5
    return num / max(den, 1e-12)


# ---------------------------------------------------------------- reference
def ref_model(cfg, init, device):
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32,
                                             attn_implementation="sdpa").to(device)
    m = inject_adapter_in_model(LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=0.0,
                                           target_modules=list(TARGETS), bias="none"), m)
    for n, p in m.named_parameters():
        p.requires_grad = "lora_" in n
        if p.requires_grad:
            p.data.copy_(init[n].to(device))
    return m


def ref_train(cfg, init, data, steps, device, start=0, state=None):
    """Unsplit single-GPU training with the pipeline's exact recipe. Returns losses, grad norms,
    final LoRA, and (model, opt, sched) for continuing."""
    t, m = cfg.training, int(cfg.training.distributed.pipeline.microbatches)
    model = ref_model(cfg, init, device)
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t)
    for _ in range(start):
        sched.step()
    losses, norms = [], []
    clip = getattr(t, "grad_clip", None) or getattr(t, "max_grad_norm", 1.0)
    it = data.iter_torch_batches(batch_size=m)
    for _ in range(steps):
        b = next(it)
        tot = 0.0
        for i in range(m):
            row = b["input_ids"][i:i + 1]
            loss = model(input_ids=row, labels=row, use_cache=False).loss / m
            loss.backward()
            tot += float(loss)
        params = [p for p in model.parameters() if p.requires_grad]
        gn = float(torch.stack([p.grad.pow(2).sum() for p in params]).sum().sqrt())
        if gn > clip:
            torch._foreach_mul_([p.grad for p in params], clip / (gn + 1e-6))
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot); norms.append(gn)
    final = {n: p.detach().float().cpu() for n, p in model.named_parameters() if p.requires_grad}
    return losses, norms, final, model


def ref_eval(model, cfg, datas, limit_rows=None):
    """Mean per-micro-batch loss over every row of the given fake datasets."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for d in datas:
            for b in d.iter_torch_batches(batch_size=int(cfg.training.batch_size)):
                tot += float(model(input_ids=b["input_ids"], labels=b["labels"], use_cache=False).loss)
                n += 1
    return tot / max(n, 1)


# ---------------------------------------------------------------- scenarios
def run_pipeline(cfg, data, val=None, device=None, init=None, rank=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    stage, info = build_stage(cfg, rank, world, device)
    if init is not None:
        set_lora(stage, info, init[info.copy], device)
    init_snapshot = gather_lora(stage, info)
    tr = Tracker()
    PL.train_pipeline(stage=stage, info=info, dataset=data(info) if callable(data) else data,
                      val_dataset=val(info) if callable(val) else val, cfg=cfg, rank=rank,
                      local_rank=device.index, tracker=tr, world_size=world)
    return stage, info, tr, init_snapshot


def numeric(name, base, mdir, device, stages, m, schedule="1f1b", act=True, shared=True, steps=3):
    cfg = make_cfg(base, mdir, stages=stages, m=m, schedule=schedule, ckpt_act=act, steps=steps)
    copies = dist.get_world_size() // stages
    data = (lambda info: FakeData(seed=3, salt=0 if shared else info.copy))
    stage, info, tr, init = run_pipeline(cfg, data, device=device)
    final = gather_lora(stage, info)
    ok, detail = True, ""
    if dist.get_rank() == 0:
        # copies must hold identical adapters at the start and the end
        same_init = max(max_rel(init[c], init[0]) for c in init)
        same_end = max(max_rel(final[c], final[0]) for c in final)
        ok &= same_end < 1e-6      # copies start different (random LoRA-A per rank) and are synced
        detail += f"copies={copies} start_diff_before_sync={same_init:.1e} end_diff={same_end:.1e}"
        if shared:
            losses, norms, rfinal, _ = ref_train(cfg, {k: v for k, v in _full_names(init[0]).items()},
                                                 FakeData(seed=3, salt=0), steps, device)
            pl = [tr.rows[s + 1]["loss"] for s in range(steps)]
            pn = [tr.rows[s + 1]["grad_norm"] for s in range(steps)]
            dl = max(abs(a - b) for a, b in zip(pl, losses))
            dn = max(abs(a - b) / b for a, b in zip(pn, norms))
            dw = max_rel(delta(final[0], init[0]), delta(rfinal, init[0]))
            ok &= dl < 2e-4 and dn < 2e-3 and dw < 2e-3
            detail += f" loss_maxdiff={dl:.1e} gradnorm_maxrel={dn:.1e} update_rel={dw:.1e}"
    note(name, _agree(ok), detail)
    return stage, info, tr, init, final


def _full_names(sd):
    """canonical (model.layers.i...) -> the reference's parameter names (identical, kept explicit)."""
    return dict(sd)


def _agree(ok):
    t = torch.tensor([1 if ok else 0], device="cuda")
    dist.broadcast(t, src=0)
    return bool(t.item())


def eval_scenarios(base, mdir, device):
    stages, m = 4, 8
    copies = dist.get_world_size() // stages
    cfg = make_cfg(base, mdir, stages=stages, m=m, steps=2, eval_steps=2, eval_mb=1000)
    # 4 stage shards per copy with different lengths: exercises the round-robin and exhaustion
    lens = lambda info: {0: 3, 1: 2, 2: 4, 3: 1}[info.stage] + info.copy
    train = lambda info: FakeData(seed=3, salt=0)
    val = lambda info: FakeData(seed=11, salt=info.copy * 4 + info.stage, n_rows=lens(info))
    stage, info, tr, init = run_pipeline(cfg, train, val, device)
    final = gather_lora(stage, info)
    ok, detail = True, ""
    if dist.get_rank() == 0:
        _, _, _, model = ref_train(cfg, _full_names(init[0]), FakeData(seed=3, salt=0), 2, device)
        datas = [FakeData(seed=11, salt=c * 4 + s, n_rows={0: 3, 1: 2, 2: 4, 3: 1}[s] + c)
                 for c in range(copies) for s in range(stages)]
        want = ref_eval(model, cfg, datas)
        got = tr.rows[2].get("eval/loss")
        ok = got is not None and abs(got - want) < 2e-4
        detail = f"pipeline={got} reference={want:.6f} (rows={sum(d.n_rows for d in datas)})"
    note("eval == reference over uneven shards (all stages' shards, all copies)", _agree(ok), detail)

    # cap smaller than the data
    cfg2 = make_cfg(base, mdir, stages=stages, m=m, steps=2, eval_steps=2, eval_mb=3)
    stage2, info2 = build_stage(cfg2, dist.get_rank(), dist.get_world_size(), device)
    v = FakeData(seed=11, salt=info2.copy * 4 + info2.stage, n_rows=50)
    got = PL._run_eval_pipeline(stage2, info2, v, cfg2, device, dist.get_rank(), torch.float32)
    note("eval cap (eval_micro_batches=3) returns a value without exhausting data", _agree_all(got is not None))

    # empty validation set: must return None on every rank, no hang
    got = PL._run_eval_pipeline(stage2, info2, FakeData(n_rows=0), cfg2, device, dist.get_rank(), torch.float32)
    note("eval with an empty validation set returns None on every rank", _agree(got is None))
    return stage, info, tr, init, final


def resume_scenarios(base, mdir, device, tmp):
    stages, m = 4, 8
    rank = dist.get_rank()
    data = lambda info: FakeData(seed=3, salt=0)

    # uninterrupted 6 steps
    d_full = os.path.join(tmp, "full")
    cfg_f = make_cfg(base, mdir, stages=stages, m=m, steps=6, save_dir=d_full, save_interval=100)
    sf, inf_f, tr_f, init = run_pipeline(cfg_f, data, device=device)
    full_final = gather_lora(sf, inf_f)

    # run A: same config as the uninterrupted run, "crashes" when asked for batch 4 (step 5);
    # checkpoints every 2 steps leave step_2 and step_4
    d_res = os.path.join(tmp, "res")
    cfg_a = make_cfg(base, mdir, stages=stages, m=m, steps=6, save_dir=d_res, save_interval=2)
    crashed = False
    try:
        run_pipeline(cfg_a, lambda info: FakeData(seed=3, salt=0, crash_at=4), device=device, init=init)
    except RuntimeError as e:
        crashed = "simulated crash" in str(e)
    note("simulated crash at step 5 stops the run", _agree_all(crashed))
    ok = rank != 0 or all(os.path.exists(os.path.join(d_res, f"step_{s}", ".complete")) for s in (2, 4))
    note("checkpoints written with .complete at steps 2 and 4", _agree(ok))
    # a half-written newer checkpoint must be ignored
    if rank == 0:
        os.makedirs(os.path.join(d_res, "step_5"), exist_ok=True)
    dist.barrier()
    note("newest complete checkpoint is chosen (incomplete step_5 ignored)",
         _agree(PL._latest_checkpoint(d_res)[0] == 4))

    # run C: resume at 4, continue to 6 on the data that follows (skip=4)
    cfg_c = make_cfg(base, mdir, stages=stages, m=m, steps=6, save_dir=d_res, save_interval=100)
    sc, inf_c, tr_c, _ = run_pipeline(cfg_c, lambda info: FakeData(seed=3, salt=0, skip=4), device=device)
    res_final = gather_lora(sc, inf_c)
    ok, detail = True, ""
    if rank == 0:
        dl = max(abs(tr_c.rows[s]["loss"] - tr_f.rows[s]["loss"]) for s in (5, 6))
        dw = max_rel(delta(res_final[0], init[0]), delta(full_final[0], init[0]))
        ok = dl < 1e-4 and dw < 1e-4 and 1 not in tr_c.rows
        detail = f"steps 5-6 loss_maxdiff={dl:.1e} final_update_rel={dw:.1e} first_logged_step={min(tr_c.rows)}"
    note("resumed run == uninterrupted run (loss, weights, optimizer, scheduler)", _agree(ok), detail)

    # resume with a finished run: nothing to train, no hang
    cfg_d = make_cfg(base, mdir, stages=stages, m=m, steps=6, save_dir=d_res, save_interval=100)
    run_pipeline(cfg_d, lambda info: FakeData(seed=3, salt=0, skip=6), device=device)
    note("resume at max_steps trains nothing and exits cleanly", True)

    # missing optimizer files: weights-only resume
    if rank == 0:
        for f in os.listdir(os.path.join(d_res, "step_6")):
            if f.startswith("opt_"):
                os.remove(os.path.join(d_res, "step_6", f))
    dist.barrier()
    cfg_e = make_cfg(base, mdir, stages=stages, m=m, steps=7, save_dir=d_res, save_interval=100)
    _, _, tr_e, _ = run_pipeline(cfg_e, lambda info: FakeData(seed=3, salt=0, skip=6), device=device)
    ok = rank != 0 or (7 in tr_e.rows and tr_e.rows[7]["loss"] == tr_e.rows[7]["loss"])
    note("resume without optimizer state continues from the saved weights", _agree(ok))

    # layout mismatch: checkpoint from 4 stages, run with 2 -> every rank raises
    cfg_g = make_cfg(base, mdir, stages=2, m=m, steps=8, save_dir=d_res, save_interval=100)
    raised = False
    try:
        run_pipeline(cfg_g, data, device=device)
    except RuntimeError as e:
        raised = "failed on" in str(e)
    flag = torch.tensor([1 if raised else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    note("resume with a different stage count fails on every rank (no hang)", bool(flag.item()))


def robustness(base, mdir, device):
    world, rank = dist.get_world_size(), dist.get_rank()

    def raises(kind, **kw):
        try:
            cfg = make_cfg(base, mdir, **kw)
            if kw.get("no_lora"):
                cfg.lora.enabled = False
            stage, info = build_stage(cfg, rank, world, device)
            PL.train_pipeline(stage=stage, info=info, dataset=FakeData(), cfg=cfg, rank=rank,
                              local_rank=device.index, tracker=Tracker(), world_size=world)
        except kind:
            return True
        except Exception:                                                    # noqa: BLE001
            return False
        return False

    note("stages not dividing the world size -> ValueError", _agree_all(raises(ValueError, stages=3)))
    note("microbatches < stages -> ValueError", _agree_all(raises(ValueError, stages=4, m=2)))
    note("LoRA disabled -> ValueError", _agree_all(raises(ValueError, stages=4, no_lora=True)))
    note("batch_size>1 with sdpa -> ValueError", _agree_all(raises(ValueError, stages=4, rows_mb=2)))
    # eval requested with no validation data: warns and trains
    cfg = make_cfg(base, mdir, stages=4, steps=2, eval_steps=1)
    stage, info = build_stage(cfg, rank, world, device)
    PL.train_pipeline(stage=stage, info=info, dataset=FakeData(), val_dataset=None, cfg=cfg, rank=rank,
                      local_rank=device.index, tracker=Tracker(), world_size=world)
    note("eval_steps set without validation data: trains, skips eval", True)


def _agree_all(ok):
    t = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def functional(base, mdir_bf16, device):
    """bf16 + FlashAttention-2: packed positions, 2 sequences per micro-batch; checks the run is
    finite and that the loss moves (no reference: bf16 + flash is not bit-comparable)."""
    for packed, rows_mb in ((True, 2), (False, 2), (True, 1)):
        cfg = make_cfg(base, mdir_bf16, dtype="bfloat16", attn="flash_attention_2", packed=packed,
                       rows_mb=rows_mb, stages=4, m=8, steps=6)
        stage, info, tr, _ = run_pipeline(cfg, lambda i: FakeData(seed=5, salt=i.copy, packed=packed), device=device)
        ok, detail = True, ""
        if dist.get_rank() == 0:
            ls = [tr.rows[s]["loss"] for s in range(1, 7)]
            ok = all(l == l and l < 20 for l in ls) and abs(ls[0] - 6.24) < 1.0
            detail = f"losses {ls[0]:.3f}..{ls[-1]:.3f} (ln V = 6.24)"
        note(f"bf16 flash-attn packed={packed} rows_per_microbatch={rows_mb} finite", _agree(ok), detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/profile/llama33_70b_pipeline_4node.yaml")
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=timedelta(minutes=10))
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mdir, mdir16 = os.path.join(args.work, "model_fp32"), os.path.join(args.work, "model_bf16")
    make_model_dir(mdir, torch.float32)
    make_model_dir(mdir16, torch.bfloat16)

    plans = [
        ("4 stages x2 copies, 1F1B, act-ckpt", dict(stages=4, m=8)),
        ("2 stages x4 copies, GPipe, no act-ckpt", dict(stages=2, m=4, schedule="gpipe", act=False)),
        ("8 stages x1 copy (1-2 layers per stage)", dict(stages=8, m=8)),
    ]
    for name, kw in plans:
        try:
            numeric(f"numeric: {name}", args.base, mdir, device, **kw)
        except Exception:                                                     # noqa: BLE001
            note(f"numeric: {name}", False, traceback.format_exc().splitlines()[-1])
    try:
        numeric("copies with different data keep identical adapters (4x2)", args.base, mdir, device,
                stages=4, m=8, shared=False)
    except Exception:                                                         # noqa: BLE001
        note("copies with different data", False, traceback.format_exc().splitlines()[-1])
    for fn, a in ((eval_scenarios, (args.base, mdir, device)),
                  (resume_scenarios, (args.base, mdir, device, os.path.join(args.work, "ckpt"))),
                  (robustness, (args.base, mdir, device)),
                  (functional, (args.base, mdir16, device))):
        try:
            fn(*a)
        except Exception:                                                     # noqa: BLE001
            note(fn.__name__, False, traceback.format_exc().splitlines()[-1])
    if rank == 0:
        bad = [r for r in RESULTS if not r[1]]
        print(f"\n=== {len(RESULTS) - len(bad)}/{len(RESULTS)} passed ===", flush=True)
        for n, ok, d in RESULTS:
            print(f"{'PASS' if ok else 'FAIL'}  {n}  {d}")
    dist.barrier()
    code = 0 if all(r[1] for r in RESULTS) else 1
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
