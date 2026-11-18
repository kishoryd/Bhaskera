"""Convert an MXFP4-quantized checkpoint (gpt-oss) to a plain bf16 checkpoint, one tensor at a time.

Why: A100s have no MXFP4 kernels, so transformers dequantizes MXFP4 experts to bf16 *in host RAM* while
loading. For gpt-oss-120b that is ~234 GB of anonymous memory in one process -- more than a 187 GB node --
so FSDP/HSDP/DDP runs are killed by Ray's memory monitor before training starts. A bf16 checkpoint on disk
is instead memory-mapped (file-backed pages the kernel can drop), the same way the 470 GB Qwen3-235B loads.

Dequantization uses transformers' own convert_moe_packed_tensors, i.e. exactly what from_pretrained computes.
Peak host RAM is one expert tensor (a few GB). Runs on GPU if one is visible (faster), else CPU.

Usage: python scripts/convert_mxfp4_to_bf16.py <mxfp4_dir> <out_dir>
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.integrations.mxfp4 import convert_moe_packed_tensors


def main(src: str, dst: str) -> int:
    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wmap = index["weight_map"]
    handles: dict = {}

    def get(key):
        f = wmap[key]
        if f not in handles:
            handles[f] = safe_open(os.path.join(src, f), framework="pt", device="cpu")
        return handles[f].get_tensor(key)

    new_map, total = {}, 0
    for shard in sorted(set(wmap.values())):
        out: dict = {}
        for key in [k for k, f in wmap.items() if f == shard]:
            if key.endswith("_scales"):
                continue                                   # consumed with its _blocks partner
            if key.endswith("_blocks"):
                base = key[: -len("_blocks")]
                t = convert_moe_packed_tensors(get(key), get(base + "_scales"), dtype=torch.bfloat16).cpu()
                key = base
            else:
                t = get(key)
                if t.is_floating_point() and t.dtype != torch.bfloat16:
                    t = t.to(torch.bfloat16)
            out[key] = t.contiguous()
            new_map[key] = shard
            total += t.numel() * t.element_size()
        save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
        print(f"{shard}: {len(out)} tensors", flush=True)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for f in os.listdir(src):                               # tokenizer, chat template, generation config
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f not in ("config.json",
                                                                            "model.safetensors.index.json"):
            shutil.copy2(p, dst)
    print(f"done: {len(new_map)} tensors, {total / 2**30:.1f} GiB -> {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
