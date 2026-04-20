"""
Convert AMI Meeting Corpus (edinburghcstr/ami HF mirror, ihm variant) to Lhotse CutSet.

Source format: parquet with schema {meeting_id, audio_id, text, audio:{bytes,path},
begin_time, end_time, microphone_id, speaker_id}

AMI meetings have 4-5 speakers. SALM-Duplex expects 2-role (user/agent).
Strategy: for each meeting, pick the TOP-2 most-prolific speakers by total speech
duration, map them to user/agent (first → user, second → agent), drop utterances
from other speakers entirely. This yields cleaner 2-party conversations.

Usage:
    python convert_ami_to_lhotse.py \
        --parquet-dir /mnt/pool/datasets/ami/ihm \
        --audio-out    /mnt/pool/salm-duplex/audio/ami \
        --manifest-out /mnt/data/salm-duplex/manifests/phase3/ami
"""

from __future__ import annotations

import argparse
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


MIN_MEETING_DURATION = 60.0     # drop meetings shorter than 1 minute of speech
MAX_MEETING_DURATION = 3600.0   # cap cut length at 1 hour (AMI meetings go ~30-90 min)


def iter_parquet_rows(parquet_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for pq_path in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(str(pq_path))
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg).to_pydict()
            n = len(tbl["meeting_id"])
            for i in range(n):
                rows.append({
                    "meeting_id": tbl["meeting_id"][i],
                    "audio_id": tbl["audio_id"][i],
                    "speaker_id": tbl["speaker_id"][i],
                    "begin_time": float(tbl["begin_time"][i]),
                    "end_time": float(tbl["end_time"][i]),
                    "text": tbl["text"][i] or "",
                    "audio_bytes": tbl["audio"][i]["bytes"],
                })
    return rows


def pick_top_two_speakers(entries: list[dict]) -> dict[str, str] | None:
    """Return {speaker_id: role} for the 2 most-prolific speakers. None if <2 speakers."""
    totals: dict[str, float] = defaultdict(float)
    for e in entries:
        totals[e["speaker_id"]] += max(0.0, e["end_time"] - e["begin_time"])
    if len(totals) < 2:
        return None
    ranked = sorted(totals.items(), key=lambda x: -x[1])
    return {ranked[0][0]: "user", ranked[1][0]: "agent"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", type=Path, required=True)
    ap.add_argument("--audio-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-meetings", type=int, default=None, help="smoke-test limit")
    args = ap.parse_args()

    print(f"Loading AMI rows from {args.parquet_dir}...")
    rows = iter_parquet_rows(args.parquet_dir)
    print(f"  {len(rows):,} utterance rows")

    meetings: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        meetings[r["meeting_id"]].append(r)
    print(f"  {len(meetings):,} unique meetings")

    if args.max_meetings:
        meeting_ids = sorted(meetings.keys())[:args.max_meetings]
        meetings = {m: meetings[m] for m in meeting_ids}
        print(f"  smoke test: limiting to {len(meetings)} meetings")

    print("Building cuts...")
    cuts = []
    skipped_few_speakers = skipped_short = skipped_decode = 0

    for n, (meeting_id, entries) in enumerate(sorted(meetings.items()), 1):
        role_map = pick_top_two_speakers(entries)
        if role_map is None:
            skipped_few_speakers += 1
            continue
        kept = [e for e in entries if e["speaker_id"] in role_map]
        kept.sort(key=lambda e: e["begin_time"])

        utterances: list[Utterance] = []
        total_dur = 0.0
        for e in kept:
            if total_dur >= MAX_MEETING_DURATION:
                break
            try:
                audio, sr = decode_audio_bytes(e["audio_bytes"])
            except Exception:
                skipped_decode += 1
                continue
            audio, sr = resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
            if len(audio) == 0:
                continue
            utterances.append(Utterance(
                id=e["audio_id"],
                speaker_role=role_map[e["speaker_id"]],
                text=e["text"].strip(),
                audio=audio,
                sample_rate=sr,
            ))
            total_dur += len(audio) / sr

        if total_dur < MIN_MEETING_DURATION:
            skipped_short += 1
            continue
        if not utterances:
            skipped_decode += 1
            continue

        cut = build_dialogue_cut(
            cut_id=f"ami-{meeting_id}",
            utterances=utterances,
            output_audio_dir=args.audio_out / "concatenated",
            dataset_name="ami",
        )
        if cut is not None:
            cuts.append(cut)
        if n % 10 == 0:
            print(f"  processed {n}/{len(meetings)} meetings, {len(cuts)} cuts so far")

    print(f"\nBuilt {len(cuts)} cuts")
    print(f"  skipped: {skipped_few_speakers} <2 speakers, {skipped_short} too short, {skipped_decode} decode/empty")
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
