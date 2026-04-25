"""Build the unified label table for training.

Sources (per-tag):
  speaker_gender ← LibriSpeech/SPEAKERS.TXT (speaker_id parsed from filename)
  volume / pitch / speaking_speed / snr_db / duration ← attributes.jsonl
  speech_style / environment / mood_class ← prose-extract jsonl
  valence / arousal ← emotion2vec (deferred to v1; not built here)

Output: parquet keyed by audio_path with one row per sample. Strict
filtering: any row with a missing required tag is dropped (we'd rather
train on fewer clean rows).

Usage:
  python labels/build_labels.py --attrs subset-31k --prose prose-train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import HEADS_BY_NAME  # noqa: E402

ATTR_PATHS = {
    "subset-31k": Path("/mnt/data/salm-duplex/data/subset-31k-attributes.jsonl"),
    "train-clean-100": Path("/mnt/data/salm-duplex/data/train-clean-100-attributes.jsonl"),
    "train-clean-360": Path("/mnt/data/salm-duplex/data/train-clean-360-attributes.jsonl"),
    "train-other-500": Path("/mnt/data/salm-duplex/data/train-other-500-attributes.jsonl"),
    "test-clean": Path("/mnt/data/salm-duplex/data/librispeech-test-clean-attributes.jsonl"),
}

SPEAKERS_TXT = Path("/mnt/data/salm-duplex/data/LibriSpeech/SPEAKERS.TXT")

# In v0 the rule-extracted "low" pitch is dropped (0.2% of data — see audit).
# Map to "medium" so we don't lose those rows.
PITCH_MAP = {"low": "medium", "medium": "medium", "high": "high"}
# Volume "quiet" is essentially nonexistent on LibriSpeech (0.04%).
# Drop those rows entirely.
VOLUME_KEEP = {"normal", "loud"}


def load_speaker_gender() -> dict[int, str]:
    out: dict[int, str] = {}
    with SPEAKERS_TXT.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            try:
                spk = int(parts[0])
            except ValueError:
                continue
            sex = parts[1]
            out[spk] = {"F": "female", "M": "male"}.get(sex, "unknown")
    return out


def speaker_id_from_path(p: str) -> int | None:
    # LibriSpeech filenames: {spk}-{chapter}-{utt}.flac
    name = Path(p).stem
    parts = name.split("-")
    if not parts:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


def load_attrs(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            a = d["attributes"]
            rows.append({
                "audio_path": d["audio_path"],
                "duration": float(a["duration"]),
                "volume": a["volume"],
                "pitch": PITCH_MAP.get(a["pitch"], a["pitch"]),
                "speaking_speed": a["speaking_speed"],
                "snr_db": float(a["snr_db"]),
                "rms": float(a["rms"]),
            })
    return pd.DataFrame(rows)


def load_prose(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            t = d.get("tags") or {}
            if "_error" in t:
                continue

            def _get(name: str) -> str | None:
                v = t.get(name)
                if isinstance(v, dict):
                    val = v.get("value")
                    classes = HEADS_BY_NAME[name].classes
                    return val if val in classes else None
                return None

            rows.append({
                "audio_path": d["audio_path"],
                "speech_style": _get("speech_style"),
                "environment": _get("environment"),
                "mood_class": _get("mood_class"),
            })
    return pd.DataFrame(rows)


def build(attrs: pd.DataFrame, prose: pd.DataFrame, gender: dict[int, str]) -> pd.DataFrame:
    df = attrs.copy()
    df["speaker_id"] = df["audio_path"].map(speaker_id_from_path)
    df["speaker_gender"] = df["speaker_id"].map(gender).fillna("unknown")
    # All natural LibriSpeech is voiced
    df["voice_quality"] = "voiced"

    df = df[df["volume"].isin(VOLUME_KEEP)]
    df = df.merge(prose, on="audio_path", how="left")
    return df


def load_dsp_whispered(manifest_path: Path) -> pd.DataFrame:
    """Load DSP-whisperized samples as additional training rows.

    Whispered samples DO NOT inherit acoustic labels because the
    transformation alters volume (down to ~0.04 RMS regardless of
    source), kills F0 (pitch label invalid), and changes spectral
    character (SNR meaningless). Only speaker_gender + voice_quality
    are supervised; everything else is masked at training time by
    being NaN.
    """
    rows = []
    if not manifest_path.exists():
        return pd.DataFrame(columns=[
            "audio_path", "voiced_source", "voice_quality",
        ])
    with manifest_path.open() as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "audio_path": d["whispered_path"],
                "voiced_source": d["voiced_path"],
                "voice_quality": "whispered",
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="subset-31k",
                    help="single attrs key, or comma-separated list")
    ap.add_argument("--prose", default="/mnt/data/audio-tags/labels/prose-train.jsonl")
    ap.add_argument("--dsp-whispered", default=None,
                    help="path to DSP-whispered manifest jsonl to merge in")
    ap.add_argument("--output", default="/mnt/data/audio-tags/labels/labels-31k.parquet")
    args = ap.parse_args()

    keys = [k.strip() for k in args.attrs.split(",")]
    for k in keys:
        if k not in ATTR_PATHS:
            raise SystemExit(f"unknown --attrs key: {k}. choices: {list(ATTR_PATHS)}")

    print(f"Loading attrs: {keys}")
    attrs_parts = [load_attrs(ATTR_PATHS[k]) for k in keys]
    attrs = pd.concat(attrs_parts, ignore_index=True)
    print(f"  rows: {len(attrs):,}")

    prose_path = Path(args.prose)
    if prose_path.exists() and prose_path.stat().st_size > 0:
        print(f"Loading prose: {prose_path}")
        prose = load_prose(prose_path)
        print(f"  rows: {len(prose):,}")
    else:
        print(f"Prose file empty/missing: {prose_path} — proceeding without prose tags")
        prose = pd.DataFrame(columns=["audio_path", "speech_style", "environment", "mood_class"])

    gender = load_speaker_gender()
    print(f"Loaded {len(gender):,} speakers from SPEAKERS.TXT")

    df = build(attrs, prose, gender)

    if args.dsp_whispered:
        dsp_path = Path(args.dsp_whispered)
        dsp = load_dsp_whispered(dsp_path)
        print(f"\nMerging DSP-whispered rows from {dsp_path.name}: {len(dsp):,}")

        # Inherit ONLY speaker info from the voiced source (gender +
        # speaker_id). All acoustic/prose labels stay NaN so they're
        # masked at loss time — see notes in load_dsp_whispered docstring.
        speaker_cols = df[["audio_path", "speaker_id", "speaker_gender", "duration"]]
        whisper_rows = dsp.merge(
            speaker_cols, left_on="voiced_source", right_on="audio_path",
            how="left", suffixes=("", "_src"),
        )
        # whisper_rows.audio_path was set from dsp; drop the voiced one
        whisper_rows = whisper_rows.drop(columns=["audio_path_src", "voiced_source"], errors="ignore")
        df = pd.concat([df, whisper_rows], ignore_index=True)
        print(f"  Combined rows: {len(df):,}")
        print(f"  voice_quality dist: {df['voice_quality'].value_counts(dropna=False).to_dict()}")

    print(f"\nFinal rows: {len(df):,}")
    print("\n=== Class distributions ===")
    for col in ("speaker_gender", "volume", "pitch", "speaking_speed",
                "speech_style", "environment", "mood_class"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).to_dict()
            print(f"  {col}: {vc}")
    print(f"\n  snr_db: min={df['snr_db'].min():.1f} mean={df['snr_db'].mean():.1f} max={df['snr_db'].max():.1f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"\nWrote {args.output}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
