"""Audit attribute coverage across the LibriSpeech jsonls + describe-prose
manifests. Surfaces class distributions, missing-attr counts, and a
sampling of the prose to gauge what's mineable for the missing tags
(mood / gender / environment).

Run: python labels/audit.py
"""

from __future__ import annotations

import collections
import gzip
import json
import statistics
from pathlib import Path

ATTR_FILES = [
    Path("/mnt/data/salm-duplex/data/train-clean-100-attributes.jsonl"),
    Path("/mnt/data/salm-duplex/data/train-clean-360-attributes.jsonl"),
    Path("/mnt/data/salm-duplex/data/train-other-500-attributes.jsonl"),
    Path("/mnt/data/salm-duplex/data/librispeech-test-clean-attributes.jsonl"),
    Path("/mnt/data/salm-duplex/data/subset-31k-attributes.jsonl"),
]

DESC_TRAIN = Path("/mnt/data/salm-duplex/manifests/full-960h-described/train.jsonl.gz")
DESC_VAL = Path("/mnt/data/salm-duplex/manifests/full-960h-described/val.jsonl.gz")


def _open(p: Path):
    return gzip.open(p, "rt") if p.suffix == ".gz" else p.open()


def audit_attrs(p: Path) -> dict:
    counts: dict[str, collections.Counter] = {
        "volume": collections.Counter(),
        "pitch": collections.Counter(),
        "speaking_speed": collections.Counter(),
    }
    snrs: list[float] = []
    durs: list[float] = []
    n = 0
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            a = d["attributes"]
            for k in counts:
                counts[k][a[k]] += 1
            snrs.append(a["snr_db"])
            durs.append(a["duration"])
            n += 1
    return {
        "n": n,
        "volume": dict(counts["volume"]),
        "pitch": dict(counts["pitch"]),
        "speaking_speed": dict(counts["speaking_speed"]),
        "snr_mean": statistics.fmean(snrs),
        "snr_min": min(snrs),
        "snr_max": max(snrs),
        "dur_mean": statistics.fmean(durs),
        "dur_min": min(durs),
        "dur_max": max(durs),
    }


def sample_descriptions(p: Path, k: int = 8) -> list[str]:
    out: list[str] = []
    with _open(p) as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            d = json.loads(line)
            sup = d.get("supervisions") or []
            if sup:
                out.append(sup[0]["text"])
    return out


def main() -> None:
    print("=== Attribute coverage ===\n")
    totals = collections.Counter()
    for p in ATTR_FILES:
        if not p.exists():
            print(f"missing: {p}")
            continue
        s = audit_attrs(p)
        print(f"--- {p.name}  (n={s['n']:,}) ---")
        print(f"  volume:  {s['volume']}")
        print(f"  pitch:   {s['pitch']}")
        print(f"  speed:   {s['speaking_speed']}")
        print(f"  SNR dB:  min={s['snr_min']:.1f} mean={s['snr_mean']:.1f} max={s['snr_max']:.1f}")
        print(f"  Dur s:   min={s['dur_min']:.1f} mean={s['dur_mean']:.1f} max={s['dur_max']:.1f}")
        print()
        totals["n"] += s["n"]
    print(f"Total samples across files: {totals['n']:,}\n")

    print("=== Description prose samples (for mineability gut-check) ===\n")
    for p in [DESC_TRAIN, DESC_VAL]:
        if not p.exists():
            continue
        print(f"--- {p.name} ---")
        for desc in sample_descriptions(p, k=4):
            print(f"\n{desc[:600]}")
        print()


if __name__ == "__main__":
    main()
