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
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh

from bhaskera.introspect import introspect_model

logger = logging.getLogger(__name__)

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
        self._next = 0

    def set_positions(self, chunks: list[torch.Tensor]) -> None:
        self._positions, self._next = chunks, 0

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        if self.embed_tokens is not None:
            x = self.embed_tokens(x.reshape(1, -1))      # [mb, S] ids -> [1, mb*S, H]
        if pos is not None:                              # eval path: caller supplies positions
            pass
        elif torch.is_grad_enabled():
            pos = self._positions[self._next]
            self._next += 1
        else:
            # PipelineStage runs one no-grad forward before the first step to learn tensor
            # shapes; it must not consume a micro-batch's position ids (every micro-batch has
            # the same shape, so the first entry serves).
            pos = self._positions[0]
        pe = self.rotary(x, pos)
        for layer in self.layers:
            x = layer(x, position_ids=pos, position_embeddings=pe)
        return self.norm(x) if self.norm is not None else x


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
    stages = int(pcfg.stages)
    if stages < 2 or world_size % stages:
        raise ValueError(f"pipeline.stages={stages} must be >= 2 and divide world size {world_size}")
    if int(pcfg.microbatches) < stages:
        raise ValueError(f"pipeline.microbatches={pcfg.microbatches} must be >= stages={stages}")
    copies = world_size // stages
    mesh = init_device_mesh("cuda", (copies, stages), mesh_dim_names=("dp", "pp"))
    stage_idx, copy_idx = rank % stages, rank // stages
    first, last = stage_idx == 0, stage_idx == stages - 1

    name = cfg.model.name
    mcfg = AutoConfig.from_pretrained(name, trust_remote_code=cfg.model.trust_remote_code)
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
    targets = list(cfg.lora.target_modules or ["auto"])
    if targets == ["auto"]:
        targets = list(introspect_model(full).lora_targets)
    layer_ids = split_layers(mcfg.num_hidden_layers, stages)[stage_idx]
    rotary_cls = type(full.model.rotary_emb)

    stage = Stage(full, layer_ids, first, last)
    del full
    stage.to_empty(device=device)
    _load_stage_weights(stage, name, layer_ids, bool(getattr(mcfg, "tie_word_embeddings", False)))
    stage.rotary = rotary_cls(config=mcfg, device=device)

    if not cfg.lora.enabled:
        raise ValueError("pipeline strategy is LoRA-only (it keeps the base frozen and resident)")
    lcfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                      target_modules=targets, bias="none")
    stage = inject_adapter_in_model(lcfg, stage)
    for n, p in stage.named_parameters():
        p.requires_grad = "lora_" in n
    if pcfg.activation_checkpointing:
        for j in range(len(stage.layers)):
            stage.layers[j] = checkpoint_wrapper(stage.layers[j], checkpoint_impl=CheckpointImpl.NO_REENTRANT)

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
                        lora_targets=targets)
    return stage, info
