"""
Convert our described JSONL files to Lhotse CutSet manifests for NeMo SpeechLM2.

Input: JSONL with {audio_path, transcript, description, attributes}
Output: Lhotse CutSet JSONL manifests

Usage:
    python convert_to_lhotse.py \
        --input /mnt/data/salm-duplex/data/subset-31k-described.jsonl \
        --output /mnt/data/salm-duplex/manifests/train-31k.jsonl.gz \
        --split 0.95
"""

import argparse
import json
import gzip
from pathlib import Path
import random

import soundfile as sf
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import AudioSource


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # Skip errored descriptions
            if r.get("description", "").startswith("[ERROR"):
                continue
            records.append(r)
    return records


def record_to_cut(record: dict, idx: int) -> MonoCut:
    """Convert a single record to a Lhotse MonoCut."""
    audio_path = record["audio_path"]
    transcript = record["transcript"]
    description = record.get("description", transcript)
    attrs = record.get("attributes", {})
    duration = attrs.get("duration", 0)

    # If duration not in attributes, read from file
    if duration <= 0:
        try:
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            duration = 1.0  # fallback

    cut_id = f"libri-{idx:07d}"

    recording = Recording(
        id=cut_id,
        sources=[AudioSource(type="file", channels=[0], source=audio_path)],
        sampling_rate=16000,
        num_samples=int(duration * 16000),
        duration=duration,
    )

    supervision = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0.0,
        duration=duration,
        text=transcript,
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
            "description": description,
            "attributes": attrs,
        },
    )
    return cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="Input JSONL file(s)")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifests")
    parser.add_argument("--split", type=float, default=0.95, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load all records
    all_records = []
    for inp in args.input:
        records = load_records(inp)
        print(f"Loaded {len(records)} records from {inp}")
        all_records.extend(records)

    print(f"Total: {len(all_records)} records")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_records)

    split_idx = int(len(all_records) * args.split)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    # Convert to cuts
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
                if skipped <= 5:
                    print(f"  Skipped: {e}")

        cutset = CutSet.from_cuts(cuts)
        out_path = output_dir / f"{name}.jsonl.gz"
        cutset.to_jsonl(str(out_path))
        print(f"Wrote {len(cuts)} cuts to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
