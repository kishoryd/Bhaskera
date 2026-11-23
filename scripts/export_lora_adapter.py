"""
Export the LoRA adapter from a Bhaskera training checkpoint as a standard PEFT adapter.

Bhaskera checkpoints (distributed/checkpoint.py) are sharded torch.distributed.checkpoint
directories holding the whole wrapped model. This reads only the LoRA tensors from
<checkpoint>/model, writes adapter_model.safetensors + adapter_config.json, and the result
loads with plain `PeftModel.from_pretrained(base_model, adapter_dir)` (no Bhaskera needed).
Single process, CPU only; memory use is the adapter size, not the model size.

  python scripts/export_lora_adapter.py --checkpoint <save_dir>/step_0000030 \\
         --config <training yaml> --out <adapter dir>
  (--checkpoint may also be a save_dir: the highest completed step is used)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
import yaml

_STEP = re.compile(r"step_(\d+)$")


def latest_checkpoint(path: str) -> str:
    if _STEP.search(path.rstrip("/")):
        return path
    done = [d for d in os.listdir(path)
            if _STEP.search(d) and os.path.exists(os.path.join(path, d, ".complete"))]
    if not done:
        raise FileNotFoundError(f"no completed step_* checkpoint under {path}")
    return os.path.join(path, max(done, key=lambda d: int(_STEP.search(d).group(1))))


def read_lora_tensors(model_dir: str) -> dict[str, torch.Tensor]:
    """Every tensor whose key contains lora_, read straight from the DCP shards."""
    import torch.distributed.checkpoint as dcp

    reader = dcp.FileSystemReader(model_dir)
    meta = reader.read_metadata().state_dict_metadata
    keys = [k for k in meta if "lora_" in k]
    if not keys:
        raise RuntimeError(f"no lora_ tensors in {model_dir}: was this run trained with lora.enabled?")
    state = {k: torch.empty(meta[k].size, dtype=meta[k].properties.dtype) for k in keys}
    dcp.load(state, storage_reader=reader)      # one process, no process group: loads whole tensors
    # save_checkpoint writes {"model": state_dict}, so every key carries a "model." prefix
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}


def to_peft_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """model-state-dict names -> PEFT adapter-file names: the adapter name is dropped
    (`...lora_A.default.weight` -> `...lora_A.weight`); other wrappers are kept verbatim
    (the module path is the same in the base model PEFT wraps at load time)."""
    out = {}
    for k, v in state.items():
        k = k.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")
        k = re.sub(r"\.lora_([AB])\.[^.]+\.weight$", r".lora_\1.weight", k)
        out[k] = v.contiguous()
    return out


def target_module_paths(peft_state: dict[str, torch.Tensor]) -> list[str]:
    """Full module paths (relative to the wrapped model) that carry adapters."""
    paths = set()
    for k in peft_state:
        m = re.match(r"(?:base_model\.model\.)?(.+)\.lora_[AB]\.weight$", k)
        if m:
            paths.add(m.group(1))
    return sorted(paths)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="the training yaml (for r, alpha, dropout, base model)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    lora, base = cfg.get("lora", {}), cfg["model"]["name"]
    ckpt = latest_checkpoint(a.checkpoint)
    meta_path = os.path.join(ckpt, "meta.json")
    step = json.load(open(meta_path))["step"] if os.path.exists(meta_path) else -1
    print(f"checkpoint {ckpt} (step {step})")

    state = to_peft_keys(read_lora_tensors(os.path.join(ckpt, "model")))
    targets = target_module_paths(state)
    n_params = sum(v.numel() for v in state.values())
    bad = [k for k, v in state.items() if not torch.isfinite(v.float()).all()]
    if bad:
        sys.exit(f"non-finite LoRA tensors: {bad[:3]}")
    print(f"{len(state)} tensors, {n_params:,} parameters, {len(targets)} adapted modules")

    from peft import LoraConfig
    from safetensors.torch import save_file

    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in state.items()},
              os.path.join(a.out, "adapter_model.safetensors"))
    LoraConfig(r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
               lora_dropout=float(lora.get("dropout", 0.0)), bias="none",
               target_modules=targets,
               base_model_name_or_path=base).save_pretrained(a.out)
    json.dump({"step": step, "checkpoint": ckpt, "tensors": len(state), "parameters": n_params,
               "modules": len(targets)}, open(os.path.join(a.out, "export_info.json"), "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
