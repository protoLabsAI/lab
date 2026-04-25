"""Slice DailyTalk multi-turn audio cuts into per-utterance clips
with emotion / act / speaker labels mapped to our taxonomy.

Source: /mnt/data/salm-duplex/manifests/dailytalk/{train,val}.jsonl.gz
Source audio: /mnt/data/salm-duplex/data/dailytalk-audio/DailyTalk/concatenated/*.wav (44.1 kHz)
Output:
  /mnt/data/audio-tags/dailytalk/utterances/<cut_id>-<sup_id>.wav  (16 kHz mono)
  /mnt/data/audio-tags/dailytalk/manifest.jsonl                   (one row per utterance)
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# DailyTalk emotion → our mood_class (taxonomy v0.1)
EMOTION_MAP: dict[str, str] = {
    "no emotion": "neutral",
    "happiness": "excited",
    "surprise": "excited",
    "sadness": "sad",
    "anger": "tense",
    "disgust": "tense",
    "fear": "tense",
}

# All DailyTalk: conversational, indoor_quiet
SPEECH_STYLE = "conversational"
ENVIRONMENT = "indoor_quiet"
VOICE_QUALITY = "voiced"
SR = 16000


def process(manifest_path: Path, out_dir: Path, manifest_out: Path,
            min_dur: float = 0.5, max_dur: float = 30.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    n_total = n_kept = 0
    t0 = time.time()
    cut_audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    with manifest_out.open("w") as out_f:
        with gzip.open(manifest_path, "rt") as f:
            for line in f:
                d = json.loads(line)
                cut_id = d["id"]
                src = d["recording"]["sources"][0]["source"]
                # Cache the cut audio so we don't re-read for each supervision
                if cut_id not in cut_audio_cache:
                    if len(cut_audio_cache) > 8:  # tiny LRU
                        cut_audio_cache.pop(next(iter(cut_audio_cache)))
                    wav, sr = sf.read(src, dtype="float32")
                    if wav.ndim > 1:
                        wav = wav.mean(axis=1)
                    if sr != SR:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                    cut_audio_cache[cut_id] = (wav, SR)
                wav, sr = cut_audio_cache[cut_id]

                for sup in d.get("supervisions", []):
                    n_total += 1
                    dur = sup["duration"]
                    if not (min_dur <= dur <= max_dur):
                        continue
                    start = sup["start"]
                    s0 = int(start * sr)
                    s1 = int((start + dur) * sr)
                    clip = wav[s0:s1]
                    if len(clip) < int(min_dur * sr):
                        continue

                    sup_id = sup["id"]
                    out_path = out_dir / f"{sup_id}.wav"
                    sf.write(out_path, clip, sr)

                    cust = sup.get("custom") or {}
                    emotion = cust.get("emotion", "no emotion")
                    row = {
                        "audio_path": str(out_path),
                        "transcript": sup.get("text", ""),
                        "duration": float(dur),
                        "speaker": sup.get("speaker"),
                        "dataset_source": "dailytalk",
                        "mood_class": EMOTION_MAP.get(emotion, "neutral"),
                        "speech_style": SPEECH_STYLE,
                        "environment": ENVIRONMENT,
                        "voice_quality": VOICE_QUALITY,
                        "raw_emotion": emotion,
                        "act": cust.get("act"),
                    }
                    out_f.write(json.dumps(row) + "\n")
                    n_kept += 1

                if n_total % 2000 == 0 and n_total > 0:
                    rate = n_total / max(time.time() - t0, 1e-3)
                    print(f"  {n_total} processed  ({rate:.1f}/s)", flush=True)

    print(f"Done. {n_kept}/{n_total} kept (dur {min_dur}-{max_dur}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default="/mnt/data/salm-duplex/manifests/dailytalk/train.jsonl.gz")
    ap.add_argument("--out-dir",
                    default="/mnt/data/audio-tags/dailytalk/utterances/")
    ap.add_argument("--manifest-out",
                    default="/mnt/data/audio-tags/dailytalk/manifest.jsonl")
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()

    process(Path(args.manifest), Path(args.out_dir), Path(args.manifest_out),
            args.min_duration, args.max_duration)


if __name__ == "__main__":
    main()
