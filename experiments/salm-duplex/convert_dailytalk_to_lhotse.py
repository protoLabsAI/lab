"""
Convert DailyTalk dataset to Lhotse CutSet with duplex format.

DailyTalk has 2,541 dialogues, alternating speaker turns (speaker 0/1).
We create one Cut per DIALOGUE (concatenated audio of all turns), with
SupervisionSegments tagged with speaker role for SALM-Duplex DataModule.

Output format follows what DuplexS2SDataset expects:
- Cuts with multiple supervisions (one per turn)
- Each supervision has speaker = "user" or "agent"
- Supervisions span their respective turn time ranges
- target_audio recording field for the agent's audio (separate from source)

For monologue → duplex conversion:
- speaker 0 → "user"
- speaker 1 → "agent"

Usage:
    python convert_dailytalk_to_lhotse.py \
        --metadata /mnt/data/salm-duplex/data/dailytalk-audio/DailyTalk/dailytalk/metadata.json \
        --audio-dir /mnt/data/salm-duplex/data/dailytalk-audio/DailyTalk/dailytalk/data \
        --output-dir /mnt/data/salm-duplex/manifests/dailytalk \
        --split 0.95
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import soundfile as sf
import numpy as np
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource


def load_dailytalk(metadata_path: str, audio_dir: str) -> dict:
    """Load DailyTalk metadata and group by dialogue."""
    with open(metadata_path) as f:
        md = json.load(f)

    dialogues = defaultdict(list)
    # Metadata is nested: md[outer_key][inner_key] = utterance dict
    for outer_key, group in md.items():
        if not isinstance(group, dict):
            continue
        for inner_key, entry in group.items():
            if not isinstance(entry, dict):
                continue
            dialog_idx = entry.get("dialog_idx")
            if dialog_idx is None:
                continue
            dialogues[dialog_idx].append({
                "utterance_idx": entry["utterance_idx"],
                "speaker": entry["speaker"],
                "text": entry["text"],
                "emotion": entry.get("emotion", ""),
                "act": entry.get("act", ""),
            })

    # Sort each dialogue by utterance order
    for dialog_idx in dialogues:
        dialogues[dialog_idx].sort(key=lambda x: x["utterance_idx"])

    return dialogues


def build_dialogue_cut(dialog_idx: int, turns: list[dict], audio_dir: Path) -> MonoCut | None:
    """Concatenate dialogue turns into a single MonoCut with per-turn supervisions."""
    audio_segments = []
    supervisions = []
    cumulative_duration = 0.0
    sample_rate = None

    for turn in turns:
        wav_path = audio_dir / str(dialog_idx) / f"{turn['utterance_idx']}_{turn['speaker']}_d{dialog_idx}.wav"
        if not wav_path.exists():
            return None

        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
        except Exception:
            return None

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            return None  # mixed sample rates, skip

        duration = len(audio) / sr
        speaker_role = "user" if turn["speaker"] == 0 else "agent"

        supervisions.append(SupervisionSegment(
            id=f"dailytalk-{dialog_idx}-{turn['utterance_idx']}",
            recording_id=f"dailytalk-{dialog_idx}",
            start=cumulative_duration,
            duration=duration,
            text=turn["text"],
            speaker=speaker_role,
            language="en",
            custom={"emotion": turn.get("emotion", ""), "act": turn.get("act", "")},
        ))

        audio_segments.append(audio)
        cumulative_duration += duration

    if not audio_segments:
        return None

    # Concatenate audio
    full_audio = np.concatenate(audio_segments)
    cut_id = f"dailytalk-{dialog_idx}"

    # Save concatenated audio
    concat_dir = audio_dir.parent.parent / "concatenated"
    concat_dir.mkdir(exist_ok=True)
    concat_path = concat_dir / f"{cut_id}.wav"
    if not concat_path.exists():
        sf.write(str(concat_path), full_audio, sample_rate)

    recording = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source=str(concat_path))],
        sampling_rate=sample_rate,
        num_samples=len(full_audio),
        duration=cumulative_duration,
    )

    # For duplex S2S, we also need a target_audio recording (same audio for monologue-style training,
    # the dataset will resample to target_sample_rate during training)
    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=cumulative_duration,
        channel=0,
        recording=recording,
        supervisions=supervisions,
        custom={
            "target_audio": recording,  # same audio used as target
            "num_turns": len(turns),
        },
    )

    return cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    print(f"Loading DailyTalk metadata from {args.metadata}...")
    dialogues = load_dailytalk(args.metadata, str(audio_dir))
    print(f"Loaded {len(dialogues)} dialogues")

    print("Building cuts...")
    cuts = []
    skipped = 0
    for dialog_idx, turns in dialogues.items():
        cut = build_dialogue_cut(dialog_idx, turns, audio_dir)
        if cut is None:
            skipped += 1
            continue
        cuts.append(cut)
        if len(cuts) % 200 == 0:
            print(f"  Built {len(cuts)} cuts...")

    print(f"Built {len(cuts)} cuts (skipped {skipped})")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(cuts)
    split_idx = int(len(cuts) * args.split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cuts = CutSet.from_cuts(cuts[:split_idx])
    val_cuts = CutSet.from_cuts(cuts[split_idx:])

    train_path = output_dir / "train.jsonl.gz"
    val_path = output_dir / "val.jsonl.gz"

    train_cuts.to_jsonl(str(train_path))
    val_cuts.to_jsonl(str(val_path))

    print(f"Train: {len(train_cuts)} cuts → {train_path}")
    print(f"Val:   {len(val_cuts)} cuts → {val_path}")

    # Stats
    total_duration = sum(c.duration for c in train_cuts) + sum(c.duration for c in val_cuts)
    print(f"Total duration: {total_duration/3600:.1f} hours")
    avg_turns = sum(len(c.supervisions) for c in train_cuts) / len(train_cuts) if len(train_cuts) > 0 else 0
    print(f"Avg turns per dialogue: {avg_turns:.1f}")


if __name__ == "__main__":
    main()
