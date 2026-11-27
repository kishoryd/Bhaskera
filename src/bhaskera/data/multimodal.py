"""
bhaskera.data.multimodal
========================
Data path for image, video and audio inputs (Qwen3-VL, Qwen2.5-Omni and similar).

Input rows are chat conversations whose message content is a list of parts:

    {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": "/data/pokemon/0001.png"},
            {"type": "text",  "text": "Describe this Pokemon."}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "a green dragon with wings"}]}]}

`video` and `audio` parts work the same way ({"type": "video", "video": path},
{"type": "audio", "audio": path}). Relative paths are resolved against
data.media_root. Keep `content` a list of parts in every message (also plain text
turns), so Parquet/Arrow sees one schema for the column.

Two stages, both driven by the model's own Hugging Face processor so that the
number of placeholder tokens always matches what the encoder produces:

1. Tokenisation (`MultimodalTokenizerActor`, run once by bhaskera-tokenize):
   render the chat template, let the processor expand every media placeholder,
   mask the loss to assistant tokens, drop rows whose truncation would cut a
   media span, pad to seq_len. The cache holds input_ids / attention_mask /
   labels plus `text` (the rendered prompt, placeholders not expanded) and
   `media` (JSON list of {type, path}). Pixels and waveforms are never stored.

2. Training (`MultimodalCollator`, the Ray iterator's collate_fn): load the media
   of each batch, run the same processor on `text` + media to get the encoder
   inputs (pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw,
   input_features, feature_attention_mask, ...), check the placeholder counts
   against the cached ids, and move everything to the GPU.

FSDP / DDP uniformity: an encoder is a set of FSDP units, so every rank must run it
the same number of times per step or the all-gathers hang. When a rank's batch has
no item for a modality listed in data.modalities, the collator inserts one tiny
"fake" item whose placeholder tokens sit in padding (attention_mask 0, labels
-100): the encoder runs, its output reaches no real token and no loss term.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

MEDIA_TYPES = ("image", "video", "audio")
_PART_KEYS = {"image": ("image", "path", "url", "image_url"),
              "video": ("video", "path", "url", "video_url"),
              "audio": ("audio", "path", "url", "audio_url")}
_MEDIA_COLS = ("text", "media")
_TEXT_ONLY_KEYS = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}


# ── options ─────────────────────────────────────────────────────────────

def media_options(cfg) -> dict:
    """Every setting that changes how media are turned into tokens (hashed into the cache key)."""
    d = cfg.data
    return {
        "modalities": sorted(set(getattr(d, "modalities", None) or ["text"])),
        "media_root": getattr(d, "media_root", None) or getattr(d, "image_root", None),
        "image_min_pixels": getattr(d, "image_min_pixels", None),
        "image_max_pixels": getattr(d, "image_max_pixels", None),
        "video_fps": getattr(d, "video_fps", None),
        "video_max_frames": getattr(d, "video_max_frames", None),
        "video_max_pixels": getattr(d, "video_max_pixels", None),
        "audio_max_seconds": getattr(d, "audio_max_seconds", None),
        "loss_on": getattr(d, "loss_on", "assistant") or "assistant",
        "messages_field": (getattr(d, "format_options", None) or {}).get("messages_field", "messages"),
    }


def is_multimodal_data(cfg) -> bool:
    return any(m in MEDIA_TYPES for m in (getattr(cfg.data, "modalities", None) or ["text"]))


def load_processor(model_name: str, opts: dict, trust_remote_code: bool = False):
    """The model's processor with ExaAI's resolution / frame / audio limits applied.
    Used by both tokenisation and training so the two always agree."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    ip = getattr(proc, "image_processor", None)
    if ip is not None and (opts.get("image_min_pixels") or opts.get("image_max_pixels")):
        _set_pixel_bounds(ip, opts.get("image_min_pixels"), opts.get("image_max_pixels"))
    vp = getattr(proc, "video_processor", None)
    if vp is not None:
        if opts.get("video_max_pixels"):
            _set_pixel_bounds(vp, None, opts["video_max_pixels"])
        if opts.get("video_fps"):
            vp.fps = float(opts["video_fps"])
        if opts.get("video_max_frames"):
            vp.max_frames = int(opts["video_max_frames"])
        if hasattr(vp, "do_sample_frames"):
            vp.do_sample_frames = True
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return proc


def _set_pixel_bounds(p, min_px, max_px) -> None:
    size = dict(getattr(p, "size", None) or {})
    if min_px:
        size["shortest_edge"] = int(min_px)
        if hasattr(p, "min_pixels"):
            p.min_pixels = int(min_px)
    if max_px:
        size["longest_edge"] = int(max_px)
        if hasattr(p, "max_pixels"):
            p.max_pixels = int(max_px)
    if size:
        p.size = size


# ── rows and media ──────────────────────────────────────────────────────

def _plain(x: Any) -> Any:
    """numpy / Arrow values -> plain Python, dropping None-valued keys (Arrow fills missing
    struct fields with None, and chat templates test `'image' in part`)."""
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items() if v is not None}
    if isinstance(x, np.generic):
        return x.item()
    return x


def normalize_messages(raw: Any, media_root: Optional[str]) -> tuple[list[dict], list[dict]]:
    """Messages with every content as a list of parts and media paths resolved.
    Returns (messages, media) where media is the ordered list of {type, path}."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    messages = _plain(raw)
    media: list[dict] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}]
            continue
        parts = []
        for part in content:
            kind = part.get("type", "text")
            if kind in MEDIA_TYPES:
                path = next((part[k] for k in _PART_KEYS[kind] if part.get(k)), None)
                if not path:
                    raise ValueError(f"{kind} part without a path: {part}")
                if media_root and not os.path.isabs(path) and "://" not in path:
                    path = os.path.join(media_root, path)
                media.append({"type": kind, "path": path})
                parts.append({"type": kind, kind: path})
            else:
                parts.append({"type": "text", "text": part.get("text", "")})
        msg["content"] = parts
    return messages, media


def _load_audio(path: str, sampling_rate: int, max_seconds: Optional[float]) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sampling_rate:      # linear resample; fine for speech, avoids a librosa dependency
        n = int(round(len(wav) * sampling_rate / sr))
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)
    if max_seconds:
        wav = wav[: int(max_seconds * sampling_rate)]
    return wav


def load_media(items: list[dict], proc, opts: dict) -> dict[str, list]:
    """Decode media for the processor: PIL images, video paths (the video processor decodes and
    samples frames itself), mono float32 waveforms at the feature extractor's rate."""
    out: dict[str, list] = {"images": [], "videos": [], "audio": []}
    for it in items:
        if it["type"] == "image":
            from PIL import Image
            with Image.open(it["path"]) as im:
                out["images"].append(im.convert("RGB"))
        elif it["type"] == "video":
            out["videos"].append(it["path"])
        elif it["type"] == "audio":
            sr = int(getattr(getattr(proc, "feature_extractor", None), "sampling_rate", 16000))
            out["audio"].append(_load_audio(it["path"], sr, opts.get("audio_max_seconds")))
    return out


def run_processor(proc, texts: list[str], media: dict[str, list], return_tensors: str = "pt", padding=False):
    kwargs: dict[str, Any] = {"text": texts, "return_tensors": return_tensors, "padding": padding}
    if media.get("images"):
        kwargs["images"] = media["images"]
    if media.get("videos"):
        kwargs["videos"] = media["videos"]
    if media.get("audio"):
        kwargs["audio"] = media["audio"]
    return proc(**kwargs)


def placeholder_ids(proc) -> dict[str, int]:
    """Token id of each media placeholder the processor expands ({"image": 151655, ...})."""
    tok = proc.tokenizer
    out = {}
    for kind in MEDIA_TYPES:
        token = getattr(proc, f"{kind}_token", None)
        if token:
            tid = tok.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid != tok.unk_token_id:
                out[kind] = tid
    return out


def _media_marker_ids(proc) -> set[int]:
    """Placeholder ids plus the start/end markers around them: never trained on."""
    tok = proc.tokenizer
    ids = set(placeholder_ids(proc).values())
    for t in ("<|vision_start|>", "<|vision_end|>", "<|audio_bos|>", "<|audio_eos|>",
              "<|audio_start|>", "<|audio_end|>", "<|vision_bos|>", "<|vision_eos|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid != tok.unk_token_id:
            ids.add(tid)
    return ids


def assistant_label_mask(ids: np.ndarray, tok) -> Optional[np.ndarray]:
    """True on tokens of assistant turns (ChatML: after '<|im_start|>assistant\\n' up to and
    including '<|im_end|>'). None when the template is not ChatML."""
    start = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    if not start or not isinstance(end, int) or end == tok.unk_token_id:
        return None
    mask = np.zeros(len(ids), dtype=bool)
    n, i = len(start), 0
    ids_l = ids.tolist()
    while i <= len(ids_l) - n:
        if ids_l[i:i + n] == start:
            j = i + n
            while j < len(ids_l) and ids_l[j] != end:
                j += 1
            mask[i + n: min(j + 1, len(ids_l))] = True
            i = j + 1
        else:
            i += 1
    return mask


# ── stage 1: tokenisation ───────────────────────────────────────────────

class MultimodalTokenizerActor:
    """Ray Data map_batches callable: chat rows with media -> fixed-length token rows."""

    def __init__(self, model_name: str, seq_len: int, opts: dict, trust_remote_code: bool = False):
        self.proc = load_processor(model_name, opts, trust_remote_code)
        self.tok = self.proc.tokenizer
        self.seq_len = int(seq_len)
        self.opts = opts
        self.markers = _media_marker_ids(self.proc)
        self.placeholders = set(placeholder_ids(self.proc).values())
        self.allowed = set(opts.get("modalities") or ["text"])
        self.dropped = 0

    def _row(self, raw_messages) -> Optional[dict]:
        messages, media = normalize_messages(raw_messages, self.opts.get("media_root"))
        bad = {m["type"] for m in media} - self.allowed
        if bad:
            raise ValueError(f"row has {sorted(bad)} but data.modalities is {sorted(self.allowed)}")
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # "pt": fast image processors return nothing else
        enc = run_processor(self.proc, [text], load_media(media, self.proc, self.opts))
        ids = enc["input_ids"][0].numpy().astype(np.int64)

        train = None if self.opts.get("loss_on") == "all" else assistant_label_mask(ids, self.tok)
        if train is None:
            train = np.ones(len(ids), dtype=bool)
        train &= ~np.isin(ids, list(self.markers))

        if len(ids) > self.seq_len:
            if np.isin(ids[self.seq_len:], list(self.placeholders)).any() or not train[: self.seq_len].any():
                return None          # truncation would cut a media span or leave nothing to learn
            ids, train = ids[: self.seq_len], train[: self.seq_len]
        if not train.any():
            return None

        n = len(ids)
        pad = self.tok.pad_token_id
        input_ids = np.full(self.seq_len, pad, dtype=np.int32)
        input_ids[:n] = ids
        attn = np.zeros(self.seq_len, dtype=np.int32)
        attn[:n] = 1
        labels = np.full(self.seq_len, -100, dtype=np.int32)
        labels[:n][train] = ids[train]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels,
                "text": text, "media": json.dumps(media)}

    def __call__(self, batch: dict) -> dict:
        field = self.opts.get("messages_field", "messages")
        if field not in batch:
            raise KeyError(f"multimodal rows need a '{field}' column (data.format_options.messages_field); "
                           f"got {sorted(batch)}")
        rows = [r for r in (self._row(m) for m in batch[field]) if r is not None]
        dropped = len(batch[field]) - len(rows)
        if dropped:
            self.dropped += dropped
            logger.warning(f"multimodal tokenizer: dropped {dropped} row(s) (longer than seq_len={self.seq_len} "
                           f"through a media span, or no assistant tokens); {self.dropped} so far")
        if not rows:
            z = np.zeros((0, self.seq_len), dtype=np.int32)
            return {"input_ids": z, "attention_mask": z, "labels": z,
                    "text": np.array([], dtype=object), "media": np.array([], dtype=object)}
        return {k: (np.stack([r[k] for r in rows]) if k not in _MEDIA_COLS
                    else np.array([r[k] for r in rows], dtype=object))
                for k in ("input_ids", "attention_mask", "labels", "text", "media")}


class MultimodalTokenizerFactory:
    """Lazy per-process wrapper so Ray pickles settings, not a live processor."""

    def __init__(self, model_name: str, seq_len: int, opts: dict, trust_remote_code: bool = False):
        self.args = (model_name, seq_len, opts, trust_remote_code)
        self._actor: Optional[MultimodalTokenizerActor] = None

    def __call__(self, batch: dict) -> dict:
        if self._actor is None:
            self._actor = MultimodalTokenizerActor(*self.args)
        return self._actor(batch)


def apply_multimodal_tokenizer(ds, cfg):
    """Ray Data pipeline for multimodal rows (counterpart of tokenize._apply_map_batches)."""
    num_workers = getattr(cfg.data, "num_workers", 4)
    factory = MultimodalTokenizerFactory(cfg.model.name, cfg.data.seq_len, media_options(cfg),
                                         getattr(cfg.model, "trust_remote_code", False))
    ds = ds.repartition(max(num_workers * 2, 1))
    return ds.map_batches(factory, batch_format="numpy",
                          batch_size=min(getattr(cfg.data, "tokenize_batch_size", 32), 32),
                          num_cpus=1, concurrency=num_workers)


# ── stage 2: training-time collation ────────────────────────────────────

def _rows(col) -> np.ndarray:
    a = np.asarray(col)
    if a.dtype == object:
        a = np.stack([np.asarray(r) for r in col])
    return a


class MultimodalCollator:
    """collate_fn for Ray's iter_torch_batches: cached token rows + media -> model inputs on GPU."""

    def __init__(self, cfg, device, media_dtype=None):
        import torch

        self.torch = torch
        self.opts = media_options(cfg)
        self.proc = load_processor(cfg.model.name, self.opts, getattr(cfg.model, "trust_remote_code", False))
        self.tok = self.proc.tokenizer
        self.device = device
        self.media_dtype = media_dtype
        self.placeholders = placeholder_ids(self.proc)
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(getattr(cfg.data, "image_workers", 4) or 4)))
        # Encoders every rank must run each step: vision (images and/or video, which the
        # model encodes in separate calls) and audio.
        self.required = [m for m in MEDIA_TYPES if m in set(self.opts["modalities"])]
        self._fake: dict[str, tuple] = {}
        self._fake_count: dict[str, int] = {}
        # Ray runs collate_fn in several prefetch threads at once, and the processor's tokenizer
        # (Rust) is not safe for concurrent use ("RuntimeError: Already borrowed"). Decoding media
        # stays parallel; every processor / tokenizer call goes through this lock.
        self._proc_lock = threading.Lock()

    # Fake items: smallest input each processor accepts, placeholder expanded by the processor itself.
    def _fake_item(self, kind: str):
        if kind not in self._fake:
            from PIL import Image
            if kind == "image":
                media = {"images": [Image.new("RGB", (64, 64))]}
            elif kind == "video":
                media = {"videos": [np.zeros((4, 64, 64, 3), dtype=np.uint8)]}
            else:
                sr = int(getattr(getattr(self.proc, "feature_extractor", None), "sampling_rate", 16000))
                media = {"audio": [np.zeros(sr, dtype=np.float32)]}
            token = getattr(self.proc, f"{kind}_token")
            if kind == "video":   # raw frames: nothing to sample, no file to read
                enc = self.proc(text=[token], videos=media["videos"], return_tensors="pt", do_sample_frames=False)
            else:
                enc = run_processor(self.proc, [token], media)
            ids = enc["input_ids"][0]
            extra = {k: v for k, v in enc.items() if k not in _TEXT_ONLY_KEYS}
            self._fake[kind] = (ids, extra)
        return self._fake[kind]

    def __call__(self, batch: dict) -> dict:
        torch = self.torch
        input_ids = torch.as_tensor(_rows(batch["input_ids"]), dtype=torch.long)
        attention_mask = torch.as_tensor(_rows(batch["attention_mask"]), dtype=torch.long)
        labels = torch.as_tensor(_rows(batch["labels"]), dtype=torch.long)
        texts = [str(t) for t in batch["text"]]
        items = [json.loads(m) if m else [] for m in batch["media"]]

        loaded = list(self.pool.map(lambda it: load_media(it, self.proc, self.opts), items))
        media = {k: [x for d in loaded for x in d[k]] for k in ("images", "videos", "audio")}
        with self._proc_lock:
            enc = run_processor(self.proc, texts, media, padding=True)

        # The processor must expand placeholders exactly as at tokenisation time.
        for kind, tid in self.placeholders.items():
            got = (enc["input_ids"] == tid).sum(dim=1)
            want = (input_ids == tid).sum(dim=1)
            if not torch.equal(got, want):
                raise ValueError(
                    f"{kind} placeholder count differs from the tokenized cache ({got.tolist()} vs "
                    f"{want.tolist()}): media settings (pixels, fps, frames) changed since bhaskera-tokenize; "
                    "re-tokenize")

        extra = {k: v for k, v in enc.items() if k not in _TEXT_ONLY_KEYS}

        present = {"image": bool(media["images"]), "video": bool(media["videos"]), "audio": bool(media["audio"])}
        fakes = [k for k in self.required if not present[k]]
        if fakes:
            input_ids, attention_mask, labels, extra = self._add_fakes(fakes, input_ids, attention_mask,
                                                                       labels, extra)
        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, **extra}
        return {k: self._to_device(v) for k, v in out.items()}

    def _add_fakes(self, kinds, input_ids, attention_mask, labels, extra):
        torch = self.torch
        used = 0   # tail positions of row 0 already holding a fake item (rows are right-padded)
        for kind in kinds:
            with self._proc_lock:
                ids, fake_extra = self._fake_item(kind)
            n = self._fake_count[kind] = self._fake_count.get(kind, 0) + 1
            if n == 1 or n % 100 == 0:
                logger.info(f"multimodal collator: fake {kind} item added to keep encoders in step ({n} so far)")
            k = ids.numel()
            free = int((attention_mask[0] == 0).sum()) - used
            if free < k:   # no room in row 0's padding: widen the batch with padding
                grow = k - free
                pad = self.tok.pad_token_id
                b = input_ids.size(0)
                input_ids = torch.cat([input_ids, torch.full((b, grow), pad, dtype=torch.long)], 1)
                attention_mask = torch.cat([attention_mask, torch.zeros((b, grow), dtype=torch.long)], 1)
                labels = torch.cat([labels, torch.full((b, grow), -100, dtype=torch.long)], 1)
            L = input_ids.size(1)
            input_ids[0, L - used - k: L - used] = ids
            attention_mask[0, L - used - k: L - used] = 0
            labels[0, L - used - k: L - used] = -100
            used += k
            for key, val in fake_extra.items():
                extra[key] = val if key not in extra else torch.cat([extra[key], val], 0)
        return input_ids, attention_mask, labels, extra

    def _to_device(self, v):
        torch = self.torch
        if not isinstance(v, torch.Tensor):
            return v
        if v.is_floating_point() and self.media_dtype is not None:
            v = v.to(self.media_dtype)
        return v.to(self.device, non_blocking=True)


def batch_media_count(batch: dict) -> int:
    """Number of images + videos + audio clips encoded in a collated batch (fakes included)."""
    n = 0
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in batch:
            n += int(batch[key].shape[0])
    if "feature_attention_mask" in batch:
        n += int(batch["feature_attention_mask"].shape[0])
    return n
