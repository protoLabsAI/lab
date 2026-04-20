"""
Convert Emilia-YODAS (amphion/Emilia-Dataset) English subset to Lhotse CutSet.

Source format: 1362 tars at /mnt/pool/datasets/emilia-yodas/Emilia-YODAS/EN/
Each tar contains pairs of {video_id}_W{utt_idx}.{json,mp3}
JSON schema: {text, duration, speaker, language, dnsmos, phone_count, _id}
MP3 is the utterance audio.

Key property: tars are alphabetically sharded by video_id, so all utterances for a
given video live in exactly one tar. We process one tar at a time, group by video_id.

Strategy:
- Multi-speaker videos (≥2 unique speakers) → dialogue cut, map the top-2 speakers
  to user/agent by total speaking time.
- Single-speaker videos → skipped by default (less duplex value). Can be included
  with --include-solo for scale.
- Filter by --min-dnsmos (default 3.0) for audio quality.

Tars stream through one at a time and extracted audio is kept under
audio-out/{video_id}/ for future use. You can delete these after SHAR export.

Usage:
    python convert_emilia_to_lhotse.py \
        --tar-dir      /mnt/pool/datasets/emilia-yodas/Emilia-YODAS/EN \
        --audio-out    /mnt/pool/salm-duplex/audio/emilia \
        --manifest-out /mnt/data/salm-duplex/manifests/phase3/emilia \
        --max-tars 10                # optional: limit for first pass
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

from phase3_common import (
    TARGET_SAMPLE_RATE,
    Utterance,
    build_dialogue_cut,
    decode_audio_bytes,
    report_stats,
    resample_if_needed,
    split_and_save,
)


FILENAME_RE = re.compile(r"^(EN_[^_]+)_W(\d+)\.(json|mp3)$")

MIN_VIDEO_DURATION = 30.0
MAX_VIDEO_DURATION = 1800.0  # 30 min cap per video; longer videos are split into 30-min chunks


def parse_member_name(name: str) -> tuple[str, int, str] | None:
    """name like 'EN_tKvmUvxYZXI_W000006.json' → (video_id, utt_idx, ext)."""
    base = name.rsplit("/", 1)[-1]
    m = FILENAME_RE.match(base)
    if not m:
        return None
    video_id, utt_idx, ext = m.groups()
    return video_id, int(utt_idx), ext


def process_tar(
    tar_path: Path,
    audio_out: Path,
    min_dnsmos: float,
    include_solo: bool,
) -> tuple[list, dict]:
    """Extract + group one tar into a list of MonoCuts. Returns (cuts, stats)."""
    # First pass: read all JSON metadata + MP3 bytes into memory per video
    # (one tar ~1.1GB; 450 videos; each video ~2-5MB; tractable)
    videos: dict[str, dict[int, dict]] = defaultdict(dict)
    stats = {"tars_opened": 1, "members": 0, "videos_seen": 0,
             "multi_speaker_kept": 0, "solo_kept": 0, "solo_skipped": 0,
             "low_dnsmos": 0, "decode_errors": 0, "short": 0, "long": 0,
             "duration_cut_hours": 0.0}

    try:
        tf = tarfile.open(str(tar_path), "r|")  # streaming read
    except Exception as e:
        print(f"  ERROR opening {tar_path.name}: {e}")
        return [], stats

    with tf:
        for member in tf:
            stats["members"] += 1
            parsed = parse_member_name(member.name)
            if parsed is None:
                continue
            video_id, utt_idx, ext = parsed
            buf = tf.extractfile(member)
            if buf is None:
                continue
            data = buf.read()
            entry = videos[video_id].setdefault(utt_idx, {})
            if ext == "json":
                try:
                    entry["meta"] = json.loads(data.decode("utf-8"))
                except Exception:
                    pass
            elif ext == "mp3":
                entry["mp3"] = data

    stats["videos_seen"] = len(videos)
    cuts: list = []

    for video_id, utts in videos.items():
        ordered = sorted(utts.items(), key=lambda kv: kv[0])
        # drop utts missing meta or mp3
        ordered = [(i, e) for i, e in ordered if "meta" in e and "mp3" in e]
        if not ordered:
            continue

        # Quality filter
        ordered = [(i, e) for i, e in ordered if e["meta"].get("dnsmos", 0.0) >= min_dnsmos]
        if not ordered:
            stats["low_dnsmos"] += 1
            continue

        speakers_by_time: dict[str, float] = defaultdict(float)
        for _, e in ordered:
            speakers_by_time[e["meta"].get("speaker", "unknown")] += float(e["meta"].get("duration", 0.0))
        unique_speakers = sorted(speakers_by_time.keys())

        if len(unique_speakers) >= 2:
            top2 = sorted(speakers_by_time.items(), key=lambda x: -x[1])[:2]
            role_map = {top2[0][0]: "user", top2[1][0]: "agent"}
            ordered = [(i, e) for i, e in ordered if e["meta"].get("speaker") in role_map]
            is_solo = False
        elif include_solo:
            role_map = {unique_speakers[0]: "user"}  # alternate manually later
            is_solo = True
        else:
            stats["solo_skipped"] += 1
            continue

        utterances: list[Utterance] = []
        total_dur = 0.0
        video_audio_dir = audio_out / video_id
        for utt_idx, e in ordered:
            if total_dur >= MAX_VIDEO_DURATION:
                break
            try:
                audio, sr = decode_audio_bytes(e["mp3"])
            except Exception:
                stats["decode_errors"] += 1
                continue
            audio, sr = resample_if_needed(audio, sr, TARGET_SAMPLE_RATE)
            if len(audio) == 0:
                continue
            speaker = e["meta"].get("speaker", "unknown")
            if is_solo:
                # Alternate user/agent across utterances to simulate turn-taking
                role = "user" if utt_idx % 2 == 0 else "agent"
            else:
                role = role_map[speaker]
            utterances.append(Utterance(
                id=f"{video_id}_W{utt_idx:06d}",
                speaker_role=role,
                text=e["meta"].get("text", "").strip(),
                audio=audio,
                sample_rate=sr,
            ))
            total_dur += len(audio) / sr

        if total_dur < MIN_VIDEO_DURATION:
            stats["short"] += 1
            continue
        if not utterances:
            continue

        cut = build_dialogue_cut(
            cut_id=f"emilia-{video_id}",
            utterances=utterances,
            output_audio_dir=video_audio_dir,
            dataset_name="emilia",
        )
        if cut is not None:
            cuts.append(cut)
            stats["duration_cut_hours"] += total_dur / 3600
            if is_solo:
                stats["solo_kept"] += 1
            else:
                stats["multi_speaker_kept"] += 1

    return cuts, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-dir", type=Path, required=True)
    ap.add_argument("--audio-out", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tars", type=int, default=None, help="limit number of tars processed")
    ap.add_argument("--min-dnsmos", type=float, default=3.0, help="quality filter")
    ap.add_argument("--include-solo", action="store_true", help="include single-speaker videos as alternating user/agent")
    args = ap.parse_args()

    tars = sorted(args.tar_dir.glob("EN-B*.tar"))
    if args.max_tars:
        tars = tars[:args.max_tars]
    print(f"Processing {len(tars)} tars from {args.tar_dir}")

    all_cuts: list = []
    agg_stats = defaultdict(float)

    for n, tar_path in enumerate(tars, 1):
        print(f"[{n}/{len(tars)}] {tar_path.name}")
        cuts, stats = process_tar(tar_path, args.audio_out, args.min_dnsmos, args.include_solo)
        all_cuts.extend(cuts)
        for k, v in stats.items():
            agg_stats[k] += v
        print(f"   +{len(cuts)} cuts  (multi={stats['multi_speaker_kept']} solo={stats['solo_kept']}"
              f" skipped_solo={stats['solo_skipped']} short={stats['short']})")

    print(f"\n=== Aggregate stats ===")
    for k, v in sorted(agg_stats.items()):
        if isinstance(v, float) and not k.endswith("hours"):
            print(f"  {k}: {int(v):,}")
        elif k.endswith("hours"):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v:,}")

    print(f"\nTotal cuts: {len(all_cuts)}")
    report_stats("all", all_cuts)

    n_train, n_val = split_and_save(
        all_cuts,
        output_manifest_dir=args.manifest_out,
        train_ratio=args.train_ratio,
        seed=args.seed,
        export_shar=True,
    )
    print(f"\nManifest: {args.manifest_out}")
    print(f"  train: {n_train} cuts  |  val: {n_val} cuts")


if __name__ == "__main__":
    main()
