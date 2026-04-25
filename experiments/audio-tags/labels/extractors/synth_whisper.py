"""Synthesize paired voiced + whispered audio via Fish S2 Pro.

For each text prompt, generates two WAVs at the Fish native rate
(44.1kHz mono): one normal, one with `[whisper]` tag prefix.

Empirical Fish S2 Pro behavior (verified 2026-04-25):
- Plain `[whisper]` tag produces correct whisper acoustics ~50% of the
  time — when it works, low-freq (80-300 Hz) energy drops from 55% to
  ~3% and high-freq (2-4 kHz) noise rises from <1% to ~19%. Real
  spectral inversion, not just quiet voice.
- `[whisper in small voice]` is even more extreme but often near-silent.
- `[softly]`, `[quiet]`, `[whispered voice]`: just produce quiet voiced.
- About 50% of `[whisper]` generations come back as quiet-voiced, which
  is unusable. We filter post-hoc.

Quality filter (only kept pairs go in the manifest):
- voiced:    LF% (80-300Hz) >= 30  AND  RMS >= 0.04
- whispered: LF% < 12  AND  HF% (2-4kHz) >= 5  AND  RMS in [0.005, 0.10]

Streams via the Fish OpenAI shim on :8093. Concurrent, resumable
(skips audio_paths already on disk), with progress logging.

Output layout:
  /mnt/data/audio-tags/synth/
    voiced/    <prompt_id>.wav
    whispered/ <prompt_id>.wav
    manifest.jsonl    one row per pair, with quality flags
    rejected.jsonl    pairs that failed filter (kept on disk for inspection)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

FISH_URL = "http://localhost:8093/v1/audio/speech"
FISH_MODEL = "fish-s2-pro"
DEFAULT_VOICES = ["default"]


def _clean_transcript(t: str) -> str:
    """Strip lowercase normalization artifacts; capitalize first letter
    so Fish doesn't read it as a continuation."""
    t = re.sub(r"\s+", " ", t.strip())
    return t[0].upper() + t[1:] if t else t


def _word_count(t: str) -> int:
    return len(t.split())


def load_prompts(attr_path: Path, n: int, seed: int = 42,
                 min_words: int = 5, max_words: int = 18) -> list[dict]:
    """Sample N prompts from an attribute jsonl, in the desired length band."""
    rng = random.Random(seed)
    rows: list[dict] = []
    with attr_path.open() as f:
        for line in f:
            d = json.loads(line)
            t = d.get("transcript", "").strip()
            wc = _word_count(t)
            if min_words <= wc <= max_words:
                rows.append({
                    "prompt_id": Path(d["audio_path"]).stem,
                    "transcript": _clean_transcript(t),
                    "source_audio": d["audio_path"],
                })
    rng.shuffle(rows)
    return rows[:n]


async def _synth(client: httpx.AsyncClient, sem: asyncio.Semaphore,
                 text: str, voice: str, out_path: Path) -> dict:
    async with sem:
        try:
            r = await client.post(
                FISH_URL,
                json={
                    "model": FISH_MODEL,
                    "input": text,
                    "voice": voice,
                    "response_format": "wav",
                },
                timeout=120,
            )
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return {"ok": True, "bytes": len(r.content)}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}


def spectral_metrics(path: Path) -> dict:
    """RMS + low/high frequency energy share. Used as a Fish-output
    quality filter to reject non-whispered whispers and low-quality
    voiced samples."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    rms = float(np.sqrt(np.mean(wav**2)))
    n = 1 << int(np.ceil(np.log2(max(len(wav), 1))))
    fft = np.abs(np.fft.rfft(wav, n=n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= 80) & (freqs <= 4000)
    low = (freqs >= 80) & (freqs <= 300)
    high = (freqs >= 2000) & (freqs <= 4000)
    band_e = float(fft[band].sum()) + 1e-12
    return {
        "rms": rms,
        "lf_pct": 100.0 * float(fft[low].sum()) / band_e,
        "hf_pct": 100.0 * float(fft[high].sum()) / band_e,
        "duration_s": len(wav) / sr,
    }


def is_clean_voiced(m: dict) -> bool:
    return m["lf_pct"] >= 30.0 and m["rms"] >= 0.04 and m["duration_s"] >= 1.0


def is_clean_whispered(m: dict) -> bool:
    return (
        m["lf_pct"] < 12.0
        and m["hf_pct"] >= 5.0
        and 0.005 <= m["rms"] <= 0.10
        and m["duration_s"] >= 1.0
    )


async def _process_pair(client, sem, prompt: dict, voice: str,
                        voiced_dir: Path, whispered_dir: Path):
    pid = prompt["prompt_id"]
    text = prompt["transcript"]
    voiced_path = voiced_dir / f"{pid}.wav"
    whispered_path = whispered_dir / f"{pid}.wav"
    results = {"prompt_id": pid, "voice": voice, "text": text,
               "voiced_path": str(voiced_path),
               "whispered_path": str(whispered_path)}

    if not voiced_path.exists():
        r = await _synth(client, sem, text, voice, voiced_path)
        results["voiced"] = r
    else:
        results["voiced"] = {"ok": True, "skipped": True}

    if not whispered_path.exists():
        # Empirical: plain `[whisper]` is ignored on ~80% of LibriSpeech-
        # length prompts. `[whisper in small voice]` is much more
        # reliable per single-prompt testing (LF dropped to 0.4%).
        r = await _synth(client, sem, f"[whisper in small voice] {text}", voice, whispered_path)
        results["whispered"] = r
    else:
        results["whispered"] = {"ok": True, "skipped": True}

    # Spectral QC
    if voiced_path.exists():
        results["voiced_metrics"] = spectral_metrics(voiced_path)
        results["voiced_clean"] = is_clean_voiced(results["voiced_metrics"])
    if whispered_path.exists():
        results["whispered_metrics"] = spectral_metrics(whispered_path)
        results["whispered_clean"] = is_clean_whispered(results["whispered_metrics"])

    results["pair_clean"] = bool(results.get("voiced_clean")) and bool(results.get("whispered_clean"))
    return results


async def _run(prompts: list[dict], voices: list[str], out_root: Path,
               concurrency: int):
    voiced_dir = out_root / "voiced"
    whispered_dir = out_root / "whispered"
    voiced_dir.mkdir(parents=True, exist_ok=True)
    whispered_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.jsonl"

    rejected_path = out_root / "rejected.jsonl"
    sem = asyncio.Semaphore(concurrency)
    n_ok = n_err = n_clean = n_dirty = 0
    t0 = time.time()
    last_log = t0

    async with httpx.AsyncClient() as client:
        coros = [
            _process_pair(client, sem, p, voices[i % len(voices)],
                          voiced_dir, whispered_dir)
            for i, p in enumerate(prompts)
        ]
        with manifest.open("a") as mf, rejected_path.open("a") as rf:
            for fut in asyncio.as_completed(coros):
                res = await fut
                if res["voiced"].get("ok") and res["whispered"].get("ok"):
                    n_ok += 1
                else:
                    n_err += 1
                if res.get("pair_clean"):
                    mf.write(json.dumps(res) + "\n")
                    n_clean += 1
                else:
                    rf.write(json.dumps(res) + "\n")
                    n_dirty += 1
                now = time.time()
                if now - last_log >= 30:
                    done = n_ok + n_err
                    rate = done / max(now - t0, 1e-3)
                    eta = (len(prompts) - done) / max(rate, 1e-3)
                    keep_pct = 100 * n_clean / max(n_clean + n_dirty, 1)
                    print(f"  [{done}/{len(prompts)}]  ok={n_ok} err={n_err}  "
                          f"clean={n_clean} ({keep_pct:.0f}%)  "
                          f"{rate:.2f} pairs/s  eta {eta/3600:.2f} h",
                          flush=True)
                    mf.flush(); rf.flush()
                    last_log = now

    elapsed = time.time() - t0
    print(f"Done. ok={n_ok} err={n_err} clean={n_clean} dirty={n_dirty} "
          f"elapsed {elapsed/3600:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-attrs",
                    default="/mnt/data/salm-duplex/data/train-clean-360-attributes.jsonl",
                    help="LibriSpeech attribute jsonl to sample prompts from")
    ap.add_argument("--n", type=int, default=5000, help="number of prompt pairs")
    ap.add_argument("--out", default="/mnt/data/audio-tags/synth/")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES),
                    help="comma-separated Fish voice IDs")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    print(f"Loading prompts from {args.source_attrs}", flush=True)
    prompts = load_prompts(Path(args.source_attrs), args.n, seed=args.seed)
    print(f"  selected {len(prompts)} prompts ({args.n} requested)", flush=True)
    print(f"  voices: {voices}", flush=True)

    asyncio.run(_run(prompts, voices, Path(args.out), args.concurrency))


if __name__ == "__main__":
    main()
