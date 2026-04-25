"""MELD audio → labels.

Source CSV columns:
  Sr No., Utterance, Speaker, Emotion, Sentiment, Dialogue_ID,
  Utterance_ID, Season, Episode, StartTime, EndTime
Audio filename pattern: dia<Dialogue_ID>_utt<Utterance_ID>.flac

MELD emotion classes: anger, disgust, fear, joy, neutral, sadness, surprise.
Map to our mood_class taxonomy:

Speech_style = conversational, environment = indoor_noisy (TV background
applause/music/laughter is part of MELD audio).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MOOD_MAP: dict[str, str] = {
    "anger":    "tense",
    "disgust":  "tense",
    "fear":     "tense",
    "joy":      "excited",
    "neutral":  "neutral",
    "sadness":  "sad",
    "surprise": "excited",
}

# Friends speaker → likely gender (best-effort; main cast)
GENDER_MAP: dict[str, str] = {
    "Rachel": "female", "Monica": "female", "Phoebe": "female",
    "Joey": "male", "Chandler": "male", "Ross": "male",
}

SPEECH_STYLE = "conversational"
ENVIRONMENT = "indoor_noisy"
VOICE_QUALITY = "voiced"


def process(csv_path: Path, audio_dir: Path, split_name: str,
            manifest_out: Path) -> int:
    n_kept = n_missing = 0
    with manifest_out.open("a") as out:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.flac"
                audio_path = audio_dir / fname
                if not audio_path.exists():
                    n_missing += 1
                    continue
                emotion = r["Emotion"].strip().lower()
                row = {
                    "audio_path": str(audio_path),
                    "dataset_source": "meld",
                    "split": split_name,
                    "transcript": r["Utterance"],
                    "speaker": r["Speaker"],
                    "speaker_gender": GENDER_MAP.get(r["Speaker"], "unknown"),
                    "mood_class": MOOD_MAP.get(emotion, "neutral"),
                    "speech_style": SPEECH_STYLE,
                    "environment": ENVIRONMENT,
                    "voice_quality": VOICE_QUALITY,
                    "raw_emotion": emotion,
                    "raw_sentiment": r.get("Sentiment", "").lower(),
                    "season": r.get("Season"),
                    "episode": r.get("Episode"),
                }
                out.write(json.dumps(row) + "\n")
                n_kept += 1
    print(f"  {split_name}: {n_kept} kept, {n_missing} audio missing")
    return n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/data/datasets/MELD_audio/")
    ap.add_argument("--manifest-out",
                    default="/mnt/data/audio-tags/meld/manifest.jsonl")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.manifest_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # truncate

    total = 0
    # MELD has train/dev/test; train.csv lives at the toplevel typically
    for split, csv_name in [("train", "train.csv"), ("dev", "dev.csv"),
                             ("test", "test.csv")]:
        csv_path = root / csv_name
        audio_dir = root / split
        if not csv_path.exists() or not audio_dir.exists():
            print(f"Skipping {split}: csv={csv_path.exists()} audio_dir={audio_dir.exists()}")
            continue
        total += process(csv_path, audio_dir, split, out)
    print(f"Total: {total} rows")


if __name__ == "__main__":
    main()
