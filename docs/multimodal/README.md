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
python -m bhaskera.launcher.tokenize --config configs/multimodal/tokenize_vision.yaml --split both
# 2. put the two printed cache paths into the training config, then submit it like any other recipe
./submit.sh configs/multimodal/fsdp.yaml <nodes> <hours>
```

Multimodal jobs go through the same `submit.sh` as text jobs: the partition, GPUs per node and network come from the cluster profile, and
the checks run first. Video and audio need PyAV and soundfile (`pip install av soundfile`; the central install has them). The home page's
**Start a job** works for these recipes too, and a multimodal run that stops is continued with `./submit.sh resume --run <run id>`.

Starters, one per input type, in `configs/multimodal/`:

| Input | Training config | Tokenize config | Model |
|---|---|---|---|
| Image | `fsdp.yaml` (also `ddp.yaml`, `hsdp.yaml`) | `tokenize_vision.yaml` | Qwen3-VL-2B |
| Video | `video.yaml` | `tokenize_vision.yaml` | Qwen3-VL-2B |
| Audio | `audio.yaml` | `tokenize_audio.yaml` | Qwen2.5-Omni-3B |

A row may mix types: list them in `data.modalities`.

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

## 5. Measured (PARAM Rudra, 5 nodes × 2 A100 80 GB)

LoRA r=16 on the language model, encoders frozen, FSDP2, flash-attention 2, Liger off. Eval accuracy is next-token accuracy on the
assistant tokens of the held-out split.

| Run | Data | Steps | Eval loss | Eval accuracy | Step time | Memory per GPU |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B, image | Pokemon, 750 / 83 | 60 | 1.22 → 0.909 | 70.4% → 76.2% | 1.85 s | 12.5 GiB |
| Qwen3-VL-2B, video | moving square, 540 / 60 | 60 | 0.0002 → 0.0000 | 100% | 1.6 s | 5.6 GiB |
| Qwen2.5-Omni-3B, audio | LibriSpeech, 66 / 7 | 40 | 0.198 → 0.185 | 96.2% → 97.1% | 3.0 s | 5.6 GiB |

The video set has the same prompt for every clip and 16 possible answers, so 100% is reachable only by reading the frames: it checks the
pipeline, not quality. The audio set is tiny, so the model memorises it. Media loading keeps up: data wait is zero except for the first
batch of each epoch. Each run, with Ray start-up, model load and six evaluations, took under seven minutes.

Liger is off for these models because compute nodes without C headers cannot build Triton launchers for kernels missing from the cache.

## 6. Limits

- Liger kernels (they need C headers on the compute nodes, or a pre-built Triton cache).
- Packing of multimodal rows (3-D position ids), pipeline strategy, QLoRA.
- Vision-tower LoRA is wired (`lora.encoders`) but not benchmarked.
- MoE variants (Qwen3-VL-MoE, Qwen3-Omni-MoE) load through the same code but are untested.
- Merged-adapter export for multimodal models (save the processor next to the adapter).
