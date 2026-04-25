"""Build the v4 unified training parquet by merging:
  - existing v2 parquet (LibriSpeech voiced + DSP whispered)
  - DailyTalk per-utterance manifest
  - MELD train/dev/test manifest (we hold MELD test out for eval)
  - RAVDESS speech-only manifest

Each new source contributes mood_class / speech_style / environment /
voice_quality / speaker_gender labels with NaN on the LibriSpeech-rule
attributes (volume / pitch / speaking_speed / snr_db). Masking at loss
time handles the absent labels.

Adds a `dataset_source` column for future ablation slicing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULTS = {
    "v2_parquet": "/mnt/data/audio-tags/labels/labels-v2-train.parquet",
    "dailytalk":  "/mnt/data/audio-tags/dailytalk/manifest.jsonl",
    "meld":       "/mnt/data/audio-tags/meld/manifest.jsonl",
    "ravdess":    "/mnt/data/audio-tags/ravdess/manifest.jsonl",
    "out":        "/mnt/data/audio-tags/labels/labels-v4-train.parquet",
    "out_test":   "/mnt/data/audio-tags/labels/labels-v4-test.parquet",
    "test_clean_with_dsp":
                  "/mnt/data/audio-tags/labels/labels-test-clean-with-dsp.parquet",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    args = ap.parse_args()

    # ── Existing v2 train (LibriSpeech voiced + DSP whispered) ──
    v2 = pd.read_parquet(args.v2_parquet)
    v2["dataset_source"] = v2["voice_quality"].map(
        lambda v: "librispeech_dsp_whispered" if v == "whispered" else "librispeech"
    )
    print(f"v2 parquet: {len(v2):,} rows")

    # ── DailyTalk ──
    dt = load_jsonl(Path(args.dailytalk))
    print(f"DailyTalk: {len(dt):,} rows")

    # ── MELD ──
    meld = load_jsonl(Path(args.meld))
    print(f"MELD raw: {len(meld):,} rows")
    # Hold test split out for eval
    meld_test = meld[meld["split"] == "test"].copy() if "split" in meld.columns else pd.DataFrame()
    meld_train = meld[meld["split"] != "test"].copy() if "split" in meld.columns else meld
    print(f"  → MELD train+dev: {len(meld_train):,}  /  test held: {len(meld_test):,}")

    # ── RAVDESS ──
    ravdess = load_jsonl(Path(args.ravdess))
    print(f"RAVDESS: {len(ravdess):,} rows")
    # Hold a small per-actor stratified test (last 4 actors by id for speaker disjoint)
    if "speaker_id" in ravdess.columns:
        actor_ids = sorted(ravdess["speaker_id"].unique())
        held_actors = actor_ids[-4:]
        ravdess_test = ravdess[ravdess["speaker_id"].isin(held_actors)].copy()
        ravdess_train = ravdess[~ravdess["speaker_id"].isin(held_actors)].copy()
        print(f"  → RAVDESS train: {len(ravdess_train):,}  test (last {len(held_actors)} actors): {len(ravdess_test):,}")
    else:
        ravdess_train = ravdess
        ravdess_test = pd.DataFrame()

    # ── Concat (pandas auto-aligns columns; missing → NaN) ──
    train = pd.concat([v2, dt, meld_train, ravdess_train], ignore_index=True, sort=False)
    test = pd.concat([meld_test, ravdess_test], ignore_index=True, sort=False)

    print(f"\nv4 train rows: {len(train):,}")
    print(f"v4 holdout test rows: {len(test):,}")

    print("\n=== Source distribution (train) ===")
    print(train["dataset_source"].value_counts(dropna=False).to_dict())

    print("\n=== mood_class distribution (train, after merge) ===")
    print(train["mood_class"].value_counts(dropna=False).to_dict())

    print("\n=== speech_style distribution (train) ===")
    print(train["speech_style"].value_counts(dropna=False).to_dict())

    print("\n=== environment distribution (train) ===")
    print(train["environment"].value_counts(dropna=False).to_dict())

    print("\n=== voice_quality distribution (train) ===")
    print(train["voice_quality"].value_counts(dropna=False).to_dict())

    print("\n=== speaker_gender distribution (train) ===")
    print(train["speaker_gender"].value_counts(dropna=False).to_dict())

    # Coerce mixed-type columns (e.g. speaker_id has int from LibriSpeech +
    # str from RAVDESS) to string for parquet compatibility.
    for col in ("speaker_id", "speaker", "season", "episode"):
        if col in train.columns:
            train[col] = train[col].astype("string")
        if col in test.columns:
            test[col] = test[col].astype("string")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.out)
    print(f"\nWrote {args.out}")
    if len(test):
        test.to_parquet(args.out_test)
        print(f"Wrote {args.out_test}")


if __name__ == "__main__":
    main()
