# Multimodal fine-tuning (image, video, audio)

Bhaskera fine-tunes models that read images, video and audio as well as text, with the
same strategies and cluster setup as text models (FSDP2 or DDP, LoRA, Ray on Slurm).
Output is text; speech generation (Omni "talker") is not trained.

| Input | Model family | Loaded as | Tested with |
|---|---|---|---|
| image | Qwen3-VL (2B to 32B dense) | `AutoModelForImageTextToText` | Qwen3-VL-2B / 8B-Instruct |
| video | Qwen3-VL (same model, same vision tower) | same | Qwen3-VL-2B-Instruct |
| audio (+ image, video) | Qwen2.5-Omni, Qwen3-Omni | the thinker class only | Qwen2.5-Omni-3B |
| other | any model with a `vision_config` / `audio_config` | `config.architectures[0]` | not tested |

## 1. Data format

One JSON object per line. `content` is always a list of parts, also for plain-text turns
(Arrow needs one schema for the column):

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "image", "image": "/data/pokemon/00001.png"},
    {"type": "text",  "text": "Describe this Pokemon in one short sentence."}]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "a green dragon with wings"}]}]}
```

`{"type": "video", "video": path}` and `{"type": "audio", "audio": path}` work the same way;
a row may mix several items and types. Relative paths are resolved against
`data.media_root`. Videos are decoded with PyAV, audio with soundfile (both in the venv).

`scripts/prepare_multimodal_samples.py` builds three ready sample sets in this format:

| Set | Rows (train / val) | Task |
|---|---|---|
| `pokemon` (image) | 750 / 83 | caption a Pokemon picture |
| `moving_square` (video, synthetic) | 540 / 60 | say which colour square moves in which direction |
| `librispeech` (audio) | 66 / 7 | transcribe speech |

## 2. Settings

```yaml
model:
  kind: multimodal            # or auto (default): picked when the config has an encoder
  freeze_encoders: [all]      # or [vision], [audio], []: frozen towers get no gradients
data:
  modalities: [text, image]   # any of text, image, video, audio
  media_root: null            # prefix for relative paths
  loss_on: assistant          # or all (media placeholders are never trained on)
  image_max_pixels: 1048576   # ~1M px = at most 1024 visual tokens per image
  image_min_pixels: null
  video_fps: 4                # frames sampled per second
  video_max_frames: 16
  video_max_pixels: null      # pixel budget for a whole clip
  audio_max_seconds: 30
  image_workers: 8            # threads loading media per GPU worker
  image_prefetch_batches: 4   # batches prepared ahead of the GPU
lora:
  encoders: []                # e.g. [vision] to also put adapters on the vision tower
training.distributed.fsdp:
  shard_encoder_blocks: true  # every encoder block is its own FSDP unit
  checkpoint_encoders: false  # activation checkpointing for encoder blocks
```

Checked before anything starts (`check_multimodal_settings`): the pipeline strategy,
`runtime_packing` and QLoRA are not supported with multimodal models yet.

## 3. Run

```bash
# 1. tokenise once (CPU; decodes media only to count their tokens)
PYTHONPATH=src python -m bhaskera.launcher.tokenize --config configs/multimodal/tokenize_vision.yaml --split both
# 2. put the two printed cache paths into the training config, then
sbatch scripts/submit_multimodal.sh configs/multimodal/fsdp.yaml
```

`submit_multimodal.sh` defaults to 5 nodes on `gpu-small` (that partition's QOS minimum:
5 nodes, 10 GPUs, 240 CPUs). `BHASKERA_MM_GPU_TEST=1` runs the 2-GPU FSDP2 check first.

| Config | Model | Data |
|---|---|---|
| `configs/multimodal/fsdp.yaml` (also `ddp.yaml`, `hsdp.yaml`) | Qwen3-VL-2B | image |

For video, set `data.modalities: [text, image, video]`; for audio, use Qwen2.5-Omni with `[text, audio]`. The tokenize
configs for both are `configs/multimodal/tokenize_vision.yaml` and `tokenize_audio.yaml`.

## 4. How it works

**Model** (`models/multimodal.py`). An *encoder* is a tower (vision, audio) of identical
blocks. `ENCODER_SPECS` names, per modality, the config key that declares it and the module
names it goes by; `find_encoders` returns each tower and its blocks. Adding a modality is one
row there plus nothing in the trainer. Omni models load their thinker
(`model_class_and_config`); the checkpoint's `thinker.` prefix is handled by the sharded
loader.

**Sharding** (`distributed/fsdp.py`). Order of `fully_shard`: experts, decoder layers,
every encoder block, root. The towers' small layers (patch embedding, mergers) stay in the
root unit. LoRA excludes every encoder unless it is listed in `lora.encoders`.

**Data** (`data/multimodal.py`), two stages, both through the model's own processor so the
number of placeholder tokens always matches what the encoder produces:

1. *Tokenise* (`bhaskera-tokenize`): render the chat template, let the processor expand each
   media placeholder, mask the loss to assistant turns, drop rows whose truncation would cut a
   media span, pad. The cache holds token ids plus the prompt text and the media paths (never
   pixels). Media settings are part of the cache key.
2. *Train* (Ray `collate_fn`, in the prefetch threads): load the batch's media, run the
   processor for the encoder inputs (`pixel_values`, `image_grid_thw`, `pixel_values_videos`,
   `video_grid_thw`, `input_features`, `feature_attention_mask`), compare placeholder counts
   with the cache (a mismatch means media settings changed: re-tokenise), move to GPU.

**Every rank runs every encoder.** Encoder blocks are FSDP units, so a rank that skips an
encoder while others run it would hang in the all-gather. Qwen3-VL also encodes images and
video in separate calls. When a rank's batch has no item for a modality in
`data.modalities`, the collator adds one tiny fake item whose placeholder tokens sit in
padding (attention mask 0, label -100). The encoder runs; nothing reaches the loss. The
tests check that the loss is unchanged to 1e-4.

**Metrics.** Tokens/s counts real tokens including media placeholders. MFU counts the
language model only: encoder parameters run once per image or clip, not per token, so they
are left out of the per-token FLOPs (a slight under-estimate).

## 5. Measured (PARAM Rudra, 5 nodes x 2 A100 80GB, `gpu-small`, 2026-11-22)

LoRA r=16 on the language model, encoders frozen, FSDP2 FULL_SHARD, flash-attention 2,
Liger off (see below). Eval accuracy is next-token accuracy on the assistant tokens of the
held-out split.

| Run (job) | Data | Steps | Train loss | Eval loss | Eval acc | Step time | Mem / GPU |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B, image (130401) | Pokemon 750 / 83 | 60 | 5.59 → ~0.7 | 1.22 → 0.909 | 70.4% → 76.2% | 1.85 s | 12.5 GiB |
| Qwen3-VL-2B, video (130404) | moving square 540 / 60 | 60 | 1.13 → 0.000 | 0.0002 → 0.0000 | 100% | 1.6 s | 5.6 GiB |
| Qwen2.5-Omni-3B, audio (130407) | LibriSpeech 66 / 7 | 40 | 0.67 → 0.00 | 0.198 → 0.168 (step 20) → 0.185 | 96.2% → 97.1% | 3.0 s | 5.6 GiB |
| Qwen3-VL-8B, image (130408) | Pokemon 750 / 83 | 60 | queued: `gpu-small` had no free 5-node slot (estimated start 2026-11-23 04:40) | | | | |

- Video: every clip has the same prompt and one of 16 colour x direction answers, so 100%
  held-out accuracy is only reachable by reading the frames. A pipeline check, not a quality
  benchmark.
- Audio: 66 training clips, so the model memorises them (train loss ~0); eval loss is best at
  step 20.
- Media loading keeps up: data wait is 0 s except the first batch of each epoch (2-5 s).
- Each run, including Ray start-up, model load and 6 evaluations, took under 7 minutes.

**Problems found on the cluster and fixed**

| Symptom | Cause | Fix |
|---|---|---|
| All ranks hung in the weight broadcast before step 1 | Qwen3-VL-2B ties lm_head to the embeddings; rank 0 (real load) was tied, ranks > 0 (meta build) were not, so they held one tensor more | `tie_weights()` after the meta build (`models/loader.py`); a cross-rank tensor check now fails at once instead of hanging (`distributed/fsdp.py`) |
| `stdlib.h: No such file or directory` in the first backward | compute nodes have no C headers, so Triton cannot build launchers for Liger kernels not already in `~/.triton/cache` | `use_liger_kernel: false` in the multimodal configs |
| `__init__() got an unexpected keyword argument 'trust_remote_code'` on ranks > 0 (Omni) | concrete classes' `_from_config` forwards extra kwargs to `__init__` | drop the argument for concrete classes |

## 6. Not done yet

- Liger kernels for multimodal runs (needs C headers on the compute nodes, or a pre-built
  Triton cache).

- Packing of multimodal rows (3-D position ids), pipeline strategy, QLoRA.
- Vision-tower LoRA is wired (`lora.encoders`) but not benchmarked.
- MoE variants (Qwen3-VL-MoE, Qwen3-Omni-MoE) load through the same code but are untested.
- Merged-adapter export for multimodal models (save the processor next to the adapter).
