# models

Loads the model, applies LoRA or QLoRA, and finds the parts of any architecture that LoRA and the parallel strategies need.

## What you control

In the `model` section:

| Key | Meaning |
|---|---|
| `name` | A Hugging Face id or a local folder (compute nodes have no internet, so use a folder) |
| `dtype` | `bfloat16` (default), `float16`, `float32` or `auto` |
| `attn_impl` | `flash_attention_2`, `sdpa`, `eager`, or unset |
| `quantization` | `none` or `qlora` (4-bit frozen base model) |
| `use_liger_kernel` | Fused Triton kernels: much less memory, a little faster |
| `trust_remote_code` | Needed for models that ship their own code, such as BharatGen Param |
| `kind`, `freeze_encoders` | Multimodal models: see `docs/multimodal/README.md` |

In the `lora` section:

| Key | Meaning |
|---|---|
| `enabled`, `r`, `alpha`, `dropout` | The usual LoRA settings |
| `target_modules` | `[auto]` finds the right layers for the architecture, or list names yourself |
| `include_experts` | MoE: also put adapters on the expert layers |
| `freeze_router` | MoE: keep routers frozen (default on; training them under LoRA is unstable) |
| `modules_to_save` | Layers trained in full next to the adapters |
| `encoders` | Multimodal: also put adapters on the vision or audio tower |

To load a model that is not on Hugging Face, register a loader and set `model.name` to its name:

```python
from bhaskera.models import register_model

@register_model("my-model")
def load(cfg, device):
    ...  # return an nn.Module
```

## What ExaAI does itself

- **Loading.** `AutoModelForCausalLM` with low host memory use (or the image-text class for multimodal models). Under FSDP, rank 0 loads the real weights and the other ranks build an empty model with the same structure, including tied embeddings, so every rank agrees on the layout.
- **Introspection** (`introspect.py`). Finds the decoder layer class, expert modules, routers and LoRA targets from the model's structure, not its name. A router is a parameter-bearing sibling of an `experts` list, so SwiGLU's `gate_proj` is not mistaken for one.
- **Liger.** Patches the model for its type (Llama, Mistral, Qwen, Gemma, Phi and others) before LoRA is applied and before sharding. If a model is not supported, training continues on the normal kernels.
- **LoRA dtype.** Under FSDP the adapter weights are cast to the model's dtype, because FSDP needs one dtype within a shard group. Under DDP they stay in float32, so the optimizer moments are not squeezed into bf16.
- **Logging.** After applying LoRA, the trainable share of parameters is logged; typical runs are 0.1% to 1%.
