"""RAVDESS speech subset → labels.

RAVDESS filename convention (7 hyphen-separated fields):
  modality - vocal_channel - emotion - intensity - statement - repetition - actor
  03 (audio-only) — keep
  vocal: 01=speech, 02=song; we keep speech only (1440 files)
  emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
  intensity: 01=normal, 02=strong
  actor: odd=male, even=female (24 actors)

Output: a manifest jsonl with audio_path, mood_class, gender, intensity,
plus voice_quality=voiced, speech_style=dramatic, environment=indoor_quiet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EMOTION_NAMES: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# RAVDESS emotion → our mood_class (taxonomy v0.1)
MOOD_MAP: dict[str, str] = {
    "01": "neutral",
    "02": "calm_positive",
    "03": "excited",        # happy
    "04": "sad",
    "05": "tense",          # angry
    "06": "tense",          # fearful
    "07": "tense",          # disgust
    "08": "excited",        # surprised
}

SPEECH_STYLE = "dramatic"      # acted
ENVIRONMENT = "indoor_quiet"   # studio
VOICE_QUALITY = "voiced"


def parse_filename(name: str) -> dict | None:
    parts = name.replace(".wav", "").split("-")
    if len(parts) != 7:
        return None
    modality, vocal, emotion, intensity, statement, repetition, actor = parts
    if modality != "03" or vocal != "01":
        return None
    actor_id = int(actor)
    return {
        "modality": modality,
        "vocal": vocal,
        "emotion": emotion,
        "emotion_name": EMOTION_NAMES.get(emotion, "unknown"),
        "intensity": "strong" if intensity == "02" else "normal",
        "statement": statement,
        "repetition": int(repetition),
        "actor_id": actor_id,
        "speaker_gender": "female" if actor_id % 2 == 0 else "male",
    }


def process(audio_dir: Path, manifest_out: Path):
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    n_kept = n_skipped = 0
    with manifest_out.open("w") as f:
        for p in sorted(audio_dir.glob("*.wav")):
            meta = parse_filename(p.name)
            if meta is None:
                n_skipped += 1
                continue
            row = {
                "audio_path": str(p),
                "dataset_source": "ravdess",
                "mood_class": MOOD_MAP[meta["emotion"]],
                "speaker_gender": meta["speaker_gender"],
                "speaker_id": f"ravdess-{meta['actor_id']:02d}",
                "speech_style": SPEECH_STYLE,
                "environment": ENVIRONMENT,
                "voice_quality": VOICE_QUALITY,
                "raw_emotion": meta["emotion_name"],
                "intensity": meta["intensity"],
                "volume": "loud" if meta["intensity"] == "strong" else "normal",
            }
            f.write(json.dumps(row) + "\n")
            n_kept += 1
    print(f"Kept {n_kept} speech files, skipped {n_skipped} (song / non-audio)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="/mnt/data/datasets/ravdess-mahia/audios/")
    ap.add_argument("--manifest-out",
                    default="/mnt/data/audio-tags/ravdess/manifest.jsonl")
    args = ap.parse_args()
    process(Path(args.audio_dir), Path(args.manifest_out))


if __name__ == "__main__":
    main()
