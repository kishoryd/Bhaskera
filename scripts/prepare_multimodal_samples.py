"""
Build small chat-format sample datasets for multimodal fine-tuning, one per modality.

  image  Pokemon BLIP captions (Hugging Face parquet)  -> describe the Pokemon
  video  synthetic clips of a coloured square moving    -> say the colour and the direction
  audio  LibriSpeech dummy (Hugging Face parquet)        -> transcribe the speech

Each writes <out>/<name>/{media files, train.jsonl, val.jsonl} in the format
data/multimodal.py reads (messages with image / video / audio parts, absolute paths).

Usage:
  python scripts/prepare_multimodal_samples.py image --src <pokemon parquet dir> --out <dir>
  python scripts/prepare_multimodal_samples.py video --out <dir> [--clips 600]
  python scripts/prepare_multimodal_samples.py audio --src <librispeech parquet dir> --out <dir>
  python scripts/prepare_multimodal_samples.py mixed --samples <dir> --out <dir>        # image+2-image+video+text
  python scripts/prepare_multimodal_samples.py omni_mixed --samples <dir> --out <dir>   # audio+image+both+text
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random

import numpy as np


def _msg(role: str, *parts: dict) -> dict:
    return {"role": role, "content": list(parts)}


def _text(t: str) -> dict:
    return {"type": "text", "text": t}


def _write_split(rows: list[dict], out_dir: str, val_frac: float, seed: int = 0) -> None:
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"{out_dir}/{name}.jsonl: {len(part)} rows")


def _parquet_rows(src: str):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {src}")
    for f in files:
        yield from pq.read_table(f).to_pylist()


def build_image(src: str, out: str, val_frac: float) -> None:
    from PIL import Image
    d = os.path.join(out, "pokemon"); os.makedirs(os.path.join(d, "images"), exist_ok=True)
    rows = []
    for i, r in enumerate(_parquet_rows(src)):
        path = os.path.abspath(os.path.join(d, "images", f"{i:05d}.png"))
        if not os.path.exists(path):
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(path)
        rows.append({"messages": [
            _msg("user", {"type": "image", "image": path}, _text("Describe this Pokemon in one short sentence.")),
            _msg("assistant", _text(r["text"].strip())),
        ]})
    _write_split(rows, d, val_frac)


_COLOURS = {"red": (230, 40, 40), "green": (40, 200, 60), "blue": (50, 90, 230), "yellow": (240, 220, 40)}
_DIRS = {"left to right": (1, 0), "right to left": (-1, 0), "top to bottom": (0, 1), "bottom to top": (0, -1)}


def _clip(colour, direction, n_frames=16, size=128, box=28, rng=None) -> np.ndarray:
    dx, dy = _DIRS[direction]
    frames = np.full((n_frames, size, size, 3), 235, dtype=np.uint8)
    off = rng.integers(-20, 21)
    span = size - box
    for t in range(n_frames):
        prog = int(span * t / (n_frames - 1))
        x = prog if dx > 0 else (span - prog if dx < 0 else span // 2 + off)
        y = prog if dy > 0 else (span - prog if dy < 0 else span // 2 + off)
        x, y = int(np.clip(x, 0, span)), int(np.clip(y, 0, span))
        frames[t, y:y + box, x:x + box] = _COLOURS[colour]
    return frames


def _write_mp4(frames: np.ndarray, path: str, fps: int = 8) -> None:
    import av
    with av.open(path, "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.bit_rate = 800_000
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def build_video(out: str, clips: int, val_frac: float) -> None:
    d = os.path.join(out, "moving_square"); os.makedirs(os.path.join(d, "videos"), exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(clips):
        colour = list(_COLOURS)[i % len(_COLOURS)]
        direction = list(_DIRS)[(i // len(_COLOURS)) % len(_DIRS)]
        path = os.path.abspath(os.path.join(d, "videos", f"{i:05d}.mp4"))
        if not os.path.exists(path):
            _write_mp4(_clip(colour, direction, rng=rng), path)
        rows.append({"messages": [
            _msg("user", {"type": "video", "video": path}, _text("What moves in this video, and in which direction?")),
            _msg("assistant", _text(f"A {colour} square moves from {direction}.")),
        ]})
    _write_split(rows, d, val_frac)


def build_audio(src: str, out: str, val_frac: float) -> None:
    import soundfile as sf
    d = os.path.join(out, "librispeech"); os.makedirs(os.path.join(d, "audio"), exist_ok=True)
    rows = []
    for i, r in enumerate(_parquet_rows(src)):
        path = os.path.abspath(os.path.join(d, "audio", f"{i:05d}.wav"))
        if not os.path.exists(path):
            wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            sf.write(path, wav, sr)
        text = r.get("text") or r.get("transcription") or ""
        rows.append({"messages": [
            _msg("user", {"type": "audio", "audio": path}, _text("Transcribe this audio.")),
            _msg("assistant", _text(text.strip().capitalize())),
        ]})
    _write_split(rows, d, val_frac)


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _content(row: dict, role: str) -> list[dict]:
    return next(m["content"] for m in row["messages"] if m["role"] == role)


def _text_rows(n: int, seed: int) -> list[dict]:
    """Text-only chat rows (arithmetic and echo): rows with no media at all."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        if rng.random() < 0.5:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            q, ans = f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."
        else:
            w = rng.choice(["river", "lantern", "orchid", "granite", "violin", "harbor", "meadow", "copper"])
            q, ans = f"Repeat this word: {w}", f"{w}"
        rows.append({"messages": [_msg("user", _text(q)), _msg("assistant", _text(ans))]})
    return rows


def build_mixed(samples: str, out: str, val_frac: float) -> None:
    """Qwen3-VL mixed set: single-image rows, two-image rows, video rows and text-only rows
    shuffled together, so batches regularly miss a modality (exercises the fake-item path)."""
    d = os.path.join(out, "mixed_vl"); os.makedirs(d, exist_ok=True)
    pk = _read_jsonl(os.path.join(samples, "pokemon", "train.jsonl"))
    vid = _read_jsonl(os.path.join(samples, "moving_square", "train.jsonl"))
    rows = [r for r in pk[:300]]
    for a, b in zip(pk[300:420], pk[420:540]):            # two images in one row
        ia, ib = _content(a, "user")[0], _content(b, "user")[0]
        ca, cb = _content(a, "assistant")[0]["text"], _content(b, "assistant")[0]["text"]
        rows.append({"messages": [_msg("user", ia, ib, _text("Describe the first Pokemon, then the second.")),
                                  _msg("assistant", _text(f"First: {ca}. Second: {cb}."))]})
    rows += vid[:220] + _text_rows(220, 1)
    _write_split(rows, d, val_frac)


def build_omni_mixed(samples: str, out: str, val_frac: float) -> None:
    """Omni mixed set: audio-only, image-only, image+audio in one row, and text-only rows."""
    d = os.path.join(out, "mixed_omni"); os.makedirs(d, exist_ok=True)
    pk = _read_jsonl(os.path.join(samples, "pokemon", "train.jsonl"))
    au = _read_jsonl(os.path.join(samples, "librispeech", "train.jsonl")) + \
        _read_jsonl(os.path.join(samples, "librispeech", "val.jsonl"))
    rows = [r for r in au]                                        # audio only (73)
    rows += pk[:150]                                              # image only
    for i, a in enumerate(au):                                    # image + audio in one row
        p = pk[150 + i]
        rows.append({"messages": [
            _msg("user", _content(p, "user")[0], _content(a, "user")[0],
                 _text("Describe the picture, then transcribe the audio.")),
            _msg("assistant", _text(f"Picture: {_content(p, 'assistant')[0]['text']}. "
                                    f"Audio: {_content(a, 'assistant')[0]['text']}"))]})
    rows += _text_rows(100, 2)
    _write_split(rows, d, val_frac)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["image", "video", "audio", "mixed", "omni_mixed"])
    p.add_argument("--src", help="source parquet directory (image, audio)")
    p.add_argument("--samples", help="directory with the pokemon / moving_square / librispeech sets (mixed, omni_mixed)")
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=600, help="video: number of synthetic clips")
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    if a.kind == "image":
        build_image(a.src, a.out, a.val_frac)
    elif a.kind == "video":
        build_video(a.out, a.clips, a.val_frac)
    elif a.kind == "mixed":
        build_mixed(a.samples or a.out, a.out, a.val_frac)
    elif a.kind == "omni_mixed":
        build_omni_mixed(a.samples or a.out, a.out, a.val_frac)
    else:
        build_audio(a.src, a.out, a.val_frac)


if __name__ == "__main__":
    main()
