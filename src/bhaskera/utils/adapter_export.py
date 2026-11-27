"""Export the trained LoRA adapter in the standard PEFT format.

The checkpoints ExaAI writes while training are sharded and only another ExaAI run can read them. The
adapter written here is a plain ``adapter_model.safetensors`` plus ``adapter_config.json``: anyone can load it
on top of the base model with ``PeftModel.from_pretrained(base_model, folder)``, merge it, serve it, or
register it in the MLflow model registry (see ``MLflowLogger.log_adapter``).

``export_adapter`` is collective: every rank must call it, because a sharded (FSDP) model has to be gathered
first. Only rank 0 writes files. Only the small trainable adapter weights are gathered, never the base model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

_CARD = """# LoRA adapter

Trained with ExaAI (powered by Bhaskera). Load it on top of the base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base}")
model = PeftModel.from_pretrained(base, "<this folder>")
```

Base model: `{base}`
Trained for {steps} steps.
"""


def _unwrap(model: nn.Module) -> nn.Module:
    """The PEFT model inside a DistributedDataParallel wrapper (FSDP2 models are not wrapped)."""
    from torch.nn.parallel import DistributedDataParallel
    return model.module if isinstance(model, DistributedDataParallel) else model


def _is_peft(model: nn.Module) -> bool:
    return hasattr(model, "peft_config") and bool(model.peft_config)


def export_adapter(model: nn.Module, out_dir: str | Path, *, rank: int = 0,
                   base_model: Optional[str] = None, steps: Optional[int] = None) -> Optional[Path]:
    """Write the adapter to ``out_dir`` (rank 0) and return the folder, or None when nothing was written.

    Returns None on every rank when the model has no LoRA adapter, and on ranks other than 0.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    peft_model = _unwrap(model)
    if not _is_peft(peft_model):
        if rank == 0:
            logger.warning("adapter export skipped: the model has no LoRA adapter")
        return None

    # Gathers a sharded model (a collective call) and offloads to CPU; frozen base weights stay out.
    full = get_model_state_dict(
        peft_model, options=StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True))

    out: Optional[Path] = None
    if rank == 0:
        from peft.utils import get_peft_model_state_dict
        from safetensors.torch import save_file

        adapter = get_peft_model_state_dict(peft_model, state_dict=full)
        if not adapter:
            logger.warning("adapter export skipped: no adapter weights found")
        else:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            tensors = {k: v.detach().to("cpu").contiguous() for k, v in adapter.items()}
            save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
            config = peft_model.peft_config.get("default") or next(iter(peft_model.peft_config.values()))
            if base_model and not getattr(config, "base_model_name_or_path", None):
                config.base_model_name_or_path = base_model
            config.save_pretrained(str(out))
            base = getattr(config, "base_model_name_or_path", None) or base_model or "<base model>"
            (out / "README.md").write_text(_CARD.format(base=base, steps=steps if steps is not None else "?"))
            logger.info(f"adapter exported: {out} ({len(tensors)} tensors)")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return out
