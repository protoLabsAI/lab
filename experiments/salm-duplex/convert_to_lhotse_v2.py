"""
Convert described JSONL to Lhotse CutSet with description as supervision text.

v2: Uses the LLM-generated description as the training target instead of transcript.
Keeps transcript in custom fields for reference.

Usage:
    python convert_to_lhotse_v2.py \
        --input /mnt/data/salm-duplex/data/subset-31k-described.jsonl \
               /mnt/data/salm-duplex/data/train-clean-360-described.jsonl \
               /mnt/data/salm-duplex/data/train-other-500-described.jsonl \
        --output-dir /mnt/data/salm-duplex/manifests/full-960h-described \
        --split 0.95
"""

import argparse
import json
import gzip
import random
from pathlib import Path

import soundfile as sf
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource


PROMPT_POOL = [
    "What can you hear from this audio?",
    "Describe what you hear in this audio.",
    "Describe the speech in this audio clip.",
    "What is being said and how does the speaker sound?",
    "Analyze this audio recording.",
]


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            desc = r.get("description", "")
            if not desc or desc.startswith("[ERROR"):
                continue
            records.append(r)
    return records


def record_to_cut(record: dict, idx: int) -> MonoCut:
    audio_path = record["audio_path"]
    transcript = record["transcript"]
    description = record["description"]
    attrs = record.get("attributes", {})
    duration = attrs.get("duration", 0)

    if duration <= 0:
        try:
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            duration = 1.0

    cut_id = f"libri-desc-{idx:07d}"

    recording = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source=audio_path)],
        sampling_rate=16000,
        num_samples=int(duration * 16000),
        duration=duration,
    )

    # Use DESCRIPTION as supervision text (training target)
    # Pick a random prompt to vary the instruction
    prompt = random.choice(PROMPT_POOL)

    supervision = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0.0,
        duration=duration,
        text=description,  # <-- description as target, not transcript
        language="en",
    )

    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
        custom={
            "transcript": transcript,  # keep for reference
            "prompt": prompt,
            "attributes": attrs,
        },
    )
    return cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_records = []
    for inp in args.input:
        records = load_records(inp)
        print(f"Loaded {len(records)} records from {inp}")
        all_records.extend(records)

    print(f"Total: {len(all_records)} records")

    random.seed(args.seed)
    random.shuffle(all_records)

    split_idx = int(len(all_records) * args.split)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, records in [("train", train_records), ("val", val_records)]:
        cuts = []
        skipped = 0
        for i, r in enumerate(records):
            try:
                cut = record_to_cut(r, i)
                cuts.append(cut)
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  Skip: {e}")

        cutset = CutSet.from_cuts(cuts)
        out_path = output_dir / f"{name}.jsonl.gz"
        cutset.to_jsonl(str(out_path))
        print(f"Wrote {len(cuts)} cuts to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
