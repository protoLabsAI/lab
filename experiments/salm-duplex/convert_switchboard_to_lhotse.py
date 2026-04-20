"""
Convert Switchboard (hhoangphuoc/switchboard HF mirror) to Lhotse CutSet.

Source format: parquet shards with schema {audio: {bytes, path}, sampling_rate, transcript}
- path examples:
    sw02440B_400303_401980375.wav   → call=sw02440, channel=B, ts1=400303, ts2=401980375
    sw02189A_253389375_254574375.wav

Grouping:
- One cut per call_id (sw02440)
- Sort utterances by filename (monotonic within a call)
- Channel A → user, Channel B → agent
- Concatenate in chronological order with small silence gap
- Drop calls shorter than MIN_CALL_DURATION (filters out near-empty calls)

Usage:
    python convert_switchboard_to_lhotse.py \
        --parquet-dir /mnt/pool/datasets/switchboard/data \
        --audio-out   /mnt/pool/salm-duplex/audio/switchboard \
        --manifest-out /mnt/data/salm-duplex/manifests/phase3/switchboard
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from phase3_common import (
    TARGET_SAMPLE_RATE,
    Utterance,
    build_dialogue_cut,
    decode_audio_bytes,
    report_stats,
    resample_if_needed,
    split_and_save,
)


MIN_CALL_DURATION = 30.0      # seconds; drop calls with less total speech than this
MAX_CALL_DURATION = 1800.0    # seconds (30 min); skip outliers that blow up memory
PATH_RE = re.compile(r"^(sw\d+)([AB])_(\d+)_(\d+)\.wav$")


def parse_path(path: str) -> tuple[str, str, int] | None:
    """Returns (call_id, channel, start_ts) or None if unparseable."""
    m = PATH_RE.match(path)
    if not m:
        return None
    call_id, channel, ts1, _ts2 = m.groups()
    return call_id, channel, int(ts1)


def iter_parquet_rows(parquet_dir: Path) -> list[dict]:
    """Yield every row from all parquet files in parquet_dir (train/val/test)."""
    rows: list[dict] = []
    for pq_path in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(str(pq_path))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["audio", "sampling_rate", "transcript"])
            d = tbl.to_pydict()
            for i in range(len(d["audio"])):
                rows.append({
                    "audio_bytes": d["audio"][i]["bytes"],
                    "path": d["audio"][i]["path"],
                    "sr": d["sampling_rate"][i],
                    "text": d["transcript"][i] or "",
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--audio-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-calls", type=int, default=None, help="limit for smoke test")
    args = ap.parse_args()

    print(f"Loading Switchboard rows from {args.parquet_dir}...")
    rows = iter_parquet_rows(args.parquet_dir)
    print(f"  {len(rows):,} utterances across all splits")

    # Group by call_id, remember (channel, start_ts) for sorting
    calls: dict[str, list] = defaultdict(list)
    skipped_unparseable = 0
    for r in rows:
        parsed = parse_path(r["path"])
        if parsed is None:
            skipped_unparseable += 1
            continue
        call_id, channel, start_ts = parsed
        calls[call_id].append((start_ts, channel, r))
    print(f"  {len(calls):,} unique calls (skipped {skipped_unparseable} unparseable)")

    if args.max_calls:
        call_ids = sorted(calls.keys())[:args.max_calls]
        calls = {cid: calls[cid] for cid in call_ids}
        print(f"  smoke test: limiting to {len(calls)} calls")

    print("Building cuts...")
    cuts = []
    skipped_short = skipped_long = skipped_decode = 0

    for n, (call_id, entries) in enumerate(sorted(calls.items()), 1):
        entries.sort(key=lambda e: (e[0], e[1]))
        utterances: list[Utterance] = []
        total_dur = 0.0

        for start_ts, channel, r in entries:
            try:
                audio, sr = decode_audio_bytes(r["audio_bytes"])
            except Exception:
                skipped_decode += 1
                continue
            audio, sr = resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
            if len(audio) == 0:
                continue
            role = "user" if channel == "A" else "agent"
            utt_id = r["path"].removesuffix(".wav")
            text = r["text"].strip()
            utterances.append(Utterance(
                id=utt_id,
                speaker_role=role,
                text=text,
                audio=audio,
                sample_rate=sr,
            ))
            total_dur += len(audio) / sr

        if not utterances:
            skipped_decode += 1
            continue
        if total_dur < MIN_CALL_DURATION:
            skipped_short += 1
            continue
        if total_dur > MAX_CALL_DURATION:
            skipped_long += 1
            continue

        cut = build_dialogue_cut(
            cut_id=f"switchboard-{call_id}",
            utterances=utterances,
            output_audio_dir=args.audio_out / "concatenated",
            dataset_name="switchboard",
        )
        if cut is not None:
            cuts.append(cut)
        if n % 50 == 0:
            print(f"  processed {n}/{len(calls)} calls, {len(cuts)} cuts so far")

    print(f"\nBuilt {len(cuts)} cuts")
    print(f"  skipped: {skipped_short} too short, {skipped_long} too long, {skipped_decode} decode/empty")
    report_stats("all", cuts)

    n_train, n_val = split_and_save(
        cuts,
        output_manifest_dir=args.manifest_out,
        train_ratio=args.train_ratio,
        seed=args.seed,
        export_shar=True,
    )
    print(f"\nManifest: {args.manifest_out}")
    print(f"  train: {n_train} cuts  |  val: {n_val} cuts")


if __name__ == "__main__":
    main()
