"""
Patch existing Emilia Lhotse manifests: re-read each WAV file's actual sample
count, correct the Recording metadata, clamp supervisions, then re-save
JSONL + re-export SHAR.

Run after convert_emilia_to_lhotse.py if SHAR export failed with
AudioLoadingError (MP3 decoder sample-count drift).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import soundfile as sf
from lhotse import CutSet, MonoCut, Recording
from lhotse.audio import AudioSource


def fix_cut(cut: MonoCut) -> MonoCut:
    src = cut.recording.sources[0].source
    info = sf.info(str(src))
    actual_samples = info.frames
    sr = info.samplerate
    actual_duration = actual_samples / sr

    new_rec = Recording(
        id=cut.recording.id,
        sources=cut.recording.sources,
        sampling_rate=sr,
        num_samples=actual_samples,
        duration=actual_duration,
    )
    cut.recording = new_rec
    cut.duration = actual_duration
    if cut.custom and "target_audio" in cut.custom:
        cut.custom["target_audio"] = new_rec

    for sup in cut.supervisions:
        if sup.start + sup.duration > actual_duration:
            sup.duration = max(0.0, actual_duration - sup.start)
    return cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--shar-shard-size", type=int, default=200)
    args = ap.parse_args()

    for split, shar_name in [("train", "shar-train"), ("val", "shar-val")]:
        jsonl = args.manifest_dir / f"{split}.jsonl.gz"
        if not jsonl.exists():
            print(f"[skip] {jsonl} missing")
            continue
        print(f"\n=== {split} ===")
        print(f"loading {jsonl}...")
        cuts = list(CutSet.from_jsonl(str(jsonl)))
        print(f"  {len(cuts):,} cuts")

        fixed = []
        missing = 0
        errors = 0
        for i, c in enumerate(cuts):
            src = c.recording.sources[0].source
            if not Path(src).exists():
                missing += 1
                continue
            try:
                fixed.append(fix_cut(c))
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  fix error on {c.id}: {e}")
            if (i + 1) % 10000 == 0:
                print(f"  processed {i+1:,}/{len(cuts):,}")

        print(f"  fixed={len(fixed):,}  missing_audio={missing}  errors={errors}")

        fixed_cs = CutSet.from_cuts(fixed)
        jsonl.rename(jsonl.with_suffix(".jsonl.gz.orig"))
        fixed_cs.to_jsonl(str(jsonl))
        print(f"  rewrote {jsonl}")

        shar_dir = args.manifest_dir / shar_name
        if shar_dir.exists():
            shutil.rmtree(shar_dir)
        shar_dir.mkdir()
        print(f"  exporting SHAR → {shar_dir}")
        fixed_cs.to_shar(
            output_dir=str(shar_dir),
            fields={"recording": "wav", "target_audio": "wav"},
            shard_size=args.shar_shard_size,
            num_jobs=1,
        )
        print(f"  SHAR done: {len(list(shar_dir.glob('*.tar')))} tars")


if __name__ == "__main__":
    main()
