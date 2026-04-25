"""Compile a multi-model comparison table from existing eval JSON files.

Reads the baseline JSON (majority + linear probe) plus per-model
eval JSONs and prints a side-by-side table for the blog.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "labels"))
from taxonomy import HEADS  # noqa: E402

DEFAULT_BASELINES = "/mnt/data/training/audio-tags/v2-whisper/baselines.json"
DEFAULT_EVALS = [
    ("v2",          "/mnt/data/training/audio-tags/v2-whisper/eval-test-clean-with-dsp.json"),
    ("v3-linear",   "/mnt/data/training/audio-tags/v3-linear/eval-test-clean-with-dsp.json"),
    ("v3-balanced", "/mnt/data/training/audio-tags/v3-balanced/eval-test-clean-with-dsp.json"),
]


def _load_baselines(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {"majority": {}, "linear_probe": {}}
    return json.loads(path.read_text())


def _load_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    j = json.loads(path.read_text())
    out = {}
    for k, v in j.get("classification", {}).items():
        out[k] = {"accuracy": v["accuracy"], "f1_macro": v["f1_macro"]}
    for k, v in j.get("regression", {}).items():
        out[k] = {"mae": v["mae"]}
    return out


def _fmt_cls(d: dict | None) -> str:
    if not d:
        return "—"
    return f"{d.get('accuracy', 0)*100:5.1f}/{d.get('f1_macro', 0):.2f}"


def _fmt_reg(d: dict | None) -> str:
    if not d:
        return "—"
    return f"MAE {d.get('mae', 0):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=DEFAULT_BASELINES)
    ap.add_argument("--evals", nargs="*", default=None,
                    help="space-separated 'name=path' pairs; defaults to v2/v3-linear/v3-balanced")
    args = ap.parse_args()

    if args.evals:
        eval_pairs = []
        for entry in args.evals:
            name, _, path = entry.partition("=")
            if not path:
                raise SystemExit(f"--evals entry must be name=path, got {entry!r}")
            eval_pairs.append((name, path))
    else:
        eval_pairs = DEFAULT_EVALS

    baselines = _load_baselines(Path(args.baselines))
    majority = baselines.get("majority", {})
    probe = baselines.get("linear_probe", {})
    model_results = [(name, _load_eval(Path(p))) for name, p in eval_pairs]

    cls_heads = [h for h in HEADS if h.type == "classification"]
    reg_heads = [h for h in HEADS if h.type == "regression"]

    cols = [("majority", majority), ("linear probe", probe)] + model_results
    col_w = max(13, max(len(name) for name, _ in cols))

    print("\n=== Multi-model comparison (acc% / F1 macro) ===\n")
    header = f"{'head':18s}  " + "  ".join(f"{name:>{col_w}}" for name, _ in cols)
    print(header)
    print("─" * len(header))
    for h in cls_heads:
        row = f"{h.name:18s}  "
        row += "  ".join(f"{_fmt_cls(d.get(h.name)):>{col_w}}" for _, d in cols)
        print(row)

    print("\n=== Regression heads (MAE) ===\n")
    for h in reg_heads:
        row = f"{h.name:18s}  "
        row += "  ".join(f"{_fmt_reg(d.get(h.name)):>{col_w}}" for _, d in cols)
        print(row)

    # Markdown version, useful for the blog
    print("\n=== Markdown table ===\n")
    print("| Head | " + " | ".join(name for name, _ in cols) + " |")
    print("|---|" + "|".join(["---:"] * len(cols)) + "|")
    for h in cls_heads:
        cells = [_fmt_cls(d.get(h.name)) for _, d in cols]
        print(f"| {h.name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
