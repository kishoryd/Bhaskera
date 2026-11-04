"""
bhaskera.distributed.pipeline
=============================
strategy: pipeline -- keep the frozen model on the GPUs instead of resending it.

Why: under FSDP every GPU receives the whole frozen model twice per step (about 280 GB
for Llama-3.3-70B). On these nodes (one 100 Gb/s link, no GPUDirect) that is ~47 s per
step at any node count, so FSDP falls from ~34% MFU at 8 nodes to 7-27% at 32-64. Here
the decoder is split by layers into `stages` slices, one per GPU, and the slices never
move: only activations cross a stage boundary (~67 MB per 4,096-token sequence for
Llama-70B) and only LoRA gradients are averaged across pipeline copies. Measured with
the prototype scripts/pp_lora_bench.py: 31% (Llama-70B, 4 stages) and 27% (Mistral-123B,
8 stages) MFU, flat from 8 to 64 nodes.

Rank layout: rank = copy * stages + stage, so a pipeline is `stages` consecutive ranks;
Ray Train numbers the GPUs of a node consecutively, so a 4-stage pipeline spans 2 nodes.
build_stage() logs and warns if a pipeline spans more nodes than that.

Supported: Llama/Mistral-style decoders (model.embed_tokens, model.layers, model.norm,
model.rotary_emb, lm_head). Each stage reads only its own tensors from the safetensors
checkpoint, so host RAM and load time scale with the stage, not the model.
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh

from bhaskera.distributed.moe_routing import make_routing_recompute_stable, recompute_contexts
from bhaskera.introspect import _EXPERT_LEAF_NAMES, introspect_model

logger = logging.getLogger(__name__)

# Checkpoint layouts the per-stage loader cannot read; failing early beats a KeyError mid-load.
_UNSUPPORTED_TYPES = {
    "gpt_oss": "its experts are stored fused and MXFP4-quantized (mlp.experts.gate_up_proj_blocks/_scales), "
               "not as one tensor per expert",
}

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def split_layers(n_layers: int, stages: int) -> list[list[int]]:
    """Equal split; any remainder goes to the middle stages (the first stage also runs the
    embedding and the last the norm + LM head)."""
    base, extra = divmod(n_layers, stages)
    sizes = [base + (1 if 0 < s <= extra else 0) for s in range(stages)]
    sizes[-1] += n_layers - sum(sizes)
    out, start = [], 0
    for size in sizes:
        out.append(list(range(start, start + size)))
        start += size
    return out


def canonical_name(name: str, layer_ids: list[int] | None = None) -> str:
    """Stage-local (and wrapper-decorated) parameter name -> the unsplit model's name,
    which is also the safetensors key."""
    n = name.replace("._checkpoint_wrapped_module", "").replace(".base_layer", "")
    if layer_ids is not None:
        if n.startswith("layers."):
            j, rest = n[len("layers."):].split(".", 1)
            n = f"model.layers.{layer_ids[int(j)]}.{rest}"
        elif n.startswith(("embed_tokens.", "norm.")):
            n = "model." + n
    return n


def switch_aux_loss(logits: torch.Tensor, top_k: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Load-balancing loss of ONE MoE layer, E * sum_e f_e * P_e (Switch Transformer; the same
    quantity as transformers' load_balancing_loss_func on a single layer's router logits).
    f_e is the share of top-k assignments that went to expert e (no gradient) and P_e the mean
    router probability of e; `mask` (1 = real token) leaves padding out of both."""
    num_experts = logits.shape[-1]
    probs = F.softmax(logits.float(), dim=-1)
    picked = F.one_hot(probs.topk(top_k, dim=-1).indices, num_experts).sum(dim=1).float()
    if mask is None:
        f, p = picked.mean(0), probs.mean(0)
    else:
        m = mask.reshape(-1, 1).float()
        denom = m.sum().clamp(min=1.0)
        f, p = (picked * m).sum(0) / denom, (probs * m).sum(0) / denom
    return num_experts * (f * p).sum()


class _AuxGrad(torch.autograd.Function):
    """Identity on the stage output whose backward also hands a gradient of 1 to `aux`, so a
    stage-local loss term trains through the pipeline schedule (which only differentiates the
    last stage's loss). `aux` must already be scaled by its loss weight."""

    @staticmethod
    def forward(ctx, x, aux):
        ctx.aux_shape, ctx.aux_dtype = aux.shape, aux.dtype
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad, torch.ones(ctx.aux_shape, dtype=ctx.aux_dtype, device=grad.device)


class Stage(nn.Module):
    """A contiguous slice of the decoder: [embed] layers [norm]. The LM head lives here on
    the last stage but is applied inside the loss (fused with cross-entropy).

    Position ids do not travel through the pipeline: the loop hands every stage the
    step's per-micro-batch position ids up front (set_positions), and forward() takes them
    in micro-batch order, which is the order every schedule runs forwards in. Activation
    checkpointing recomputes from the saved layer inputs, so backward needs no lookup.
    """

    def __init__(self, full: nn.Module, layer_ids: list[int], first: bool, last: bool):
        super().__init__()
        inner = full.model
        self.layer_ids = list(layer_ids)
        self.embed_tokens = inner.embed_tokens if first else None
        self.layers = nn.ModuleList(inner.layers[i] for i in layer_ids)
        self.norm = inner.norm if last else None
        self.lm_head = full.lm_head if last else None
        self.rotary = None                   # built on the GPU (inv_freq is a buffer, not a weight)
        self._positions: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] | None = None
        self._next = 0
        self._aux: dict | None = None         # MoE load-balancing loss settings (enable_aux)
        self._logits: list[torch.Tensor] = []
        self._collecting = False
        self.aux_acc = torch.zeros(())        # sum over the step's micro-batches of per-layer aux (no weight)

    def set_positions(self, chunks: list[torch.Tensor], masks: list[torch.Tensor] | None = None) -> None:
        self._positions, self._masks, self._next = chunks, masks, 0

    def enable_aux(self, gates: list[nn.Module], top_k: int, scale: float, device) -> None:
        """MoE: add the load-balancing loss of this stage's layers to training. `scale` is
        aux_loss_weight / (micro-batches * MoE layers in the whole model), so the sum over all
        stages, micro-batches and layers is weight * the mean per-layer aux loss."""
        self._aux = {"top_k": int(top_k), "scale": float(scale)}
        self.aux_acc = torch.zeros((), device=device)
        for g in gates:
            g.register_forward_hook(self._collect_logits)

    def _collect_logits(self, module, args, out) -> None:
        if self._collecting:          # not during activation-checkpoint recompute
            self._logits.append(out)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        if self.embed_tokens is not None:
            x = self.embed_tokens(x.reshape(1, -1))      # [mb, S] ids -> [1, mb*S, H]
        mask = None
        if pos is not None:                              # eval path: caller supplies positions
            pass
        elif torch.is_grad_enabled():
            pos = self._positions[self._next]
            mask = self._masks[self._next] if self._masks else None
            self._next += 1
        else:
            # PipelineStage runs one no-grad forward before the first step to learn tensor
            # shapes; it must not consume a micro-batch's position ids (every micro-batch has
            # the same shape, so the first entry serves).
            pos = self._positions[0]
        pe = self.rotary(x, pos)
        train_aux = self._aux is not None and torch.is_grad_enabled()
        if train_aux:
            self._logits, self._collecting = [], True
        for layer in self.layers:
            x = layer(x, position_ids=pos, position_embeddings=pe)
        self._collecting = False
        x = self.norm(x) if self.norm is not None else x
        if train_aux and self._logits:
            aux = sum(switch_aux_loss(l, self._aux["top_k"], mask) for l in self._logits)
            self.aux_acc += aux.detach()
            x = _AuxGrad.apply(x, aux * self._aux["scale"])
        self._logits = []
        return x


@dataclass
class PipelineInfo:
    stages: int
    copies: int
    stage: int
    copy: int
    first: bool
    last: bool
    pp_group: object
    dp_group: object
    stage0_rank: int          # global rank of this pipeline's first stage (broadcast source)
    layer_ids: list[int]
    total_params: int         # whole model, for the FLOPs estimate
    num_layers: int
    hidden_size: int
    lora_targets: list[str]
    flops_params: int = 0     # parameters that run per token (active experts only), for MFU
    is_moe: bool = False
    moe_layers: int = 0       # MoE layers in the whole model


def _load_stage_weights(stage: Stage, ckpt_dir: str, layer_ids: list[int], tie: bool) -> None:
    from safetensors import safe_open

    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = "model.safetensors"
        with safe_open(os.path.join(ckpt_dir, single), framework="pt") as f:
            weight_map = {k: single for k in f.keys()}  # noqa: SIM118 -- safe_open is not a dict
    by_file: dict[str, list] = {}
    for name, p in stage.named_parameters():
        key = canonical_name(name, layer_ids)
        if key not in weight_map and key == "lm_head.weight" and tie:
            key = "model.embed_tokens.weight"
        if key not in weight_map:
            raise KeyError(f"pipeline stage: {key} (from {name}) not in {ckpt_dir}")
        by_file.setdefault(weight_map[key], []).append((key, p))
    with torch.no_grad():
        for fname, items in by_file.items():
            with safe_open(os.path.join(ckpt_dir, fname), framework="pt", device="cpu") as f:
                for key, p in items:
                    p.copy_(f.get_tensor(key).to(dtype=p.dtype))


def _log_placement(pp_group, info_rank: int, stages: int) -> None:
    """Warn when a pipeline spans more nodes than needed (inter-node hops are slower)."""
    hosts: list = [None] * dist.get_world_size(pp_group)
    dist.all_gather_object(hosts, socket.gethostname(), group=pp_group)
    nodes = len(set(hosts))
    per_node = max(1, torch.cuda.device_count())
    if info_rank == 0:
        logger.info(f"pipeline placement: stages on {hosts} ({nodes} nodes)")
    if nodes > math.ceil(stages / per_node):
        logger.warning(f"pipeline spans {nodes} nodes; {math.ceil(stages / per_node)} would do "
                       "(ranks of a node are not consecutive)")


def build_stage(cfg, rank: int, world_size: int, device: torch.device) -> tuple[Stage, PipelineInfo]:
    """Build this rank's stage: real weights for its slice only, LoRA, activation checkpointing."""
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoConfig, AutoModelForCausalLM

    pcfg = cfg.training.distributed.pipeline
    if str(getattr(cfg.model, "quantization", "none") or "none").lower() != "none":
        raise ValueError("strategy: pipeline keeps the base model in bf16 and does not support "
                         "model.quantization; use strategy: ddp for 4-bit QLoRA")
    stages = int(pcfg.stages)
    if stages < 2 or world_size % stages:
        raise ValueError(f"pipeline.stages={stages} must be >= 2 and divide world size {world_size}")
    if int(pcfg.microbatches) < stages:
        raise ValueError(f"pipeline.microbatches={pcfg.microbatches} must be >= stages={stages}")
    copies = world_size // stages
    stage_idx, copy_idx = rank % stages, rank // stages
    first, last = stage_idx == 0, stage_idx == stages - 1

    name = cfg.model.name
    mcfg = AutoConfig.from_pretrained(name, trust_remote_code=cfg.model.trust_remote_code)
    if getattr(mcfg, "model_type", "") in _UNSUPPORTED_TYPES:
        raise ValueError(f"strategy: pipeline does not support model_type={mcfg.model_type!r}: "
                         f"{_UNSUPPORTED_TYPES[mcfg.model_type]}; use strategy: fsdp")
    mesh = init_device_mesh("cuda", (copies, stages), mesh_dim_names=("dp", "pp"))
    dtype = _DTYPES.get(getattr(cfg.model, "dtype", "bfloat16"), torch.bfloat16)
    attn = cfg.model.attn_impl or "flash_attention_2"
    with torch.device("meta"):
        full = AutoModelForCausalLM.from_config(mcfg, attn_implementation=attn, dtype=dtype,
                                                trust_remote_code=cfg.model.trust_remote_code)
    for attr in ("embed_tokens", "layers", "norm", "rotary_emb"):
        if not hasattr(getattr(full, "model", None), attr):
            raise ValueError(f"pipeline strategy supports Llama/Mistral-style decoders; "
                             f"{type(full).__name__}.model has no '{attr}'")
    total = sum(p.numel() for p in full.parameters())
    prof = introspect_model(full)
    from bhaskera.trainer.loop import _flops_params
    flops_params = _flops_params(full, prof, total)
    targets = list(cfg.lora.target_modules or ["auto"])
    if targets == ["auto"]:
        targets = list(prof.lora_targets)
    is_moe = bool(prof.is_moe and prof.num_experts > 0)
    is_gate = lambda n, m: (isinstance(m, nn.Linear) and n.endswith("gate") and m.out_features == prof.num_experts)
    moe_layers = sum(1 for n, m in full.named_modules() if is_gate(n, m)) if is_moe else 0
    layer_ids = split_layers(mcfg.num_hidden_layers, stages)[stage_idx]
    rotary_cls = type(full.model.rotary_emb)

    stage = Stage(full, layer_ids, first, last)
    del full
    if is_moe:   # recompute under activation checkpointing must reuse the forward's routing
        make_routing_recompute_stable(stage)
    stage.to_empty(device=device)
    _load_stage_weights(stage, name, layer_ids, bool(getattr(mcfg, "tie_word_embeddings", False)))
    stage.rotary = rotary_cls(config=mcfg, device=device)

    if not cfg.lora.enabled:
        raise ValueError("pipeline strategy is LoRA-only (it keeps the base frozen and resident)")
    lkw = {}
    if is_moe and not getattr(cfg.lora, "include_experts", False):
        # PEFT matches by module name, so gate_proj/up_proj/down_proj would also hit every
        # expert; keep experts frozen like the FSDP path (models/lora.py).
        lkw["exclude_modules"] = rf"(.*\.)?({'|'.join(sorted(_EXPERT_LEAF_NAMES))})\.\d+\..*"
    lcfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                      target_modules=targets, bias="none", **lkw)
    stage = inject_adapter_in_model(lcfg, stage)
    for n, p in stage.named_parameters():
        p.requires_grad = "lora_" in n
    if is_moe and int(getattr(prof, "experts_per_token", 0)) > 0:
        moe_cfg = getattr(cfg, "moe", None)
        weight = float(getattr(moe_cfg, "aux_loss_weight", 0.01)) if moe_cfg else 0.01
        gates = [m for n, m in stage.named_modules() if is_gate(n, m)]
        if weight > 0:      # on every stage, even one without MoE layers: the loop all-reduces the metric
            stage.enable_aux(gates, prof.experts_per_token,
                             weight / (int(pcfg.microbatches) * max(1, moe_layers)), device)
    if pcfg.activation_checkpointing:
        ckw = {"context_fn": recompute_contexts} if is_moe else {}
        for j in range(len(stage.layers)):
            stage.layers[j] = checkpoint_wrapper(stage.layers[j], checkpoint_impl=CheckpointImpl.NO_REENTRANT, **ckw)

    pp_group, dp_group = mesh["pp"].get_group(), mesh["dp"].get_group()
    _log_placement(pp_group, rank, stages)
    trainable = sum(p.numel() for p in stage.parameters() if p.requires_grad)
    if rank == 0:
        logger.info(f"pipeline: {stages} stages x {copies} copies | layers per stage "
                    f"{[len(x) for x in split_layers(mcfg.num_hidden_layers, stages)]} | "
                    f"stage 0 LoRA params {trainable:,} | targets {targets}")
    info = PipelineInfo(stages=stages, copies=copies, stage=stage_idx, copy=copy_idx, first=first,
                        last=last, pp_group=pp_group, dp_group=dp_group,
                        stage0_rank=copy_idx * stages, layer_ids=layer_ids, total_params=total,
                        num_layers=mcfg.num_hidden_layers, hidden_size=mcfg.hidden_size,
                        lora_targets=targets, flops_params=flops_params, is_moe=is_moe,
                        moe_layers=moe_layers)
    return stage, info
