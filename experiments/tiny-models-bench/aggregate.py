#!/usr/bin/env python3
"""Aggregate tiny-models-bench results into a single leaderboard JSON + markdown.

Walks evals/results/tiny-bench/<model>/ and pulls:
- refusal: under_refusal/over_refusal rates from summary.json (latest by timestamp)
- custom:  per-suite avg_score + all_passed counts from custom_results.json

Writes:
- experiments/tiny-models-bench/RESULTS.md (headline table + per-model breakdown)
- experiments/tiny-models-bench/results.json (machine-readable for the blog)
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/ava/dev/lab")
BENCH_DIR = ROOT / "evals" / "results" / "tiny-bench"
OUT_DIR = ROOT / "experiments" / "tiny-models-bench"


def load_refusal(model_dir: Path) -> dict | None:
    """Find the latest refusal_* dir and load its summary.json."""
    refusal_dirs = sorted(model_dir.glob("refusal_local-bench_*"))
    if not refusal_dirs:
        return None
    latest = refusal_dirs[-1]
    summary_file = latest / "summary.json"
    if not summary_file.exists():
        return None
    data = json.loads(summary_file.read_text())
    metrics = data.get("metrics") or {}
    out = {"source": str(latest.relative_to(BENCH_DIR.parent.parent))}
    for ds_name, m in metrics.items():
        ur = m.get("under_refusal") or {}
        or_ = m.get("over_refusal") or {}
        out[ds_name] = {
            "n_harmful": ur.get("total_harmful_prompts", 0),
            "refusal_rate": ur.get("refusal_rate", 0.0),
            "compliance_rate": ur.get("compliance_rate", 0.0),
            "n_safe": or_.get("total_safe_prompts", 0),
            "over_refusal_rate": or_.get("over_refusal_rate", 0.0),
        }
    return out


def load_custom(model_dir: Path) -> dict | None:
    """Load custom_results.json (merged across all suite invocations)."""
    f = model_dir / "custom" / "custom_results.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    suites = data.get("suites") or {}
    out = {"source": "custom/custom_results.json", "suites": {}}
    for suite_name, s in suites.items():
        tasks = s.get("tasks") or []
        if not tasks:
            continue
        avg_score = sum(t.get("avg_score", 0.0) for t in tasks) / len(tasks)
        all_passed = sum(1 for t in tasks if t.get("all_passed"))
        # also dimension breakdown
        dims = defaultdict(list)
        for t in tasks:
            for trial in t.get("trials") or []:
                for g in trial.get("grades") or []:
                    dims[g.get("dimension", "unknown")].append(g.get("score", 0.0))
        dim_avgs = {d: (sum(v) / len(v) if v else 0.0) for d, v in dims.items()}
        out["suites"][suite_name] = {
            "n_tasks": len(tasks),
            "avg_score": round(avg_score, 3),
            "all_passed": all_passed,
            "dimensions": {k: round(v, 3) for k, v in dim_avgs.items()},
        }
    return out


def main():
    models = sorted(d for d in BENCH_DIR.iterdir() if d.is_dir() and not d.name.startswith("_"))

    aggregate: dict = {"as_of": "2026-05-22", "models": {}}
    for m in models:
        entry: dict = {}
        ref = load_refusal(m)
        cus = load_custom(m)
        if ref is None and cus is None:
            entry["status"] = "no data (load failure)"
        else:
            if ref:
                entry["refusal"] = ref
            if cus:
                entry["custom"] = cus
            entry["status"] = "complete" if (ref and cus) else "partial"
        aggregate["models"][m.name] = entry

    (OUT_DIR / "results.json").write_text(json.dumps(aggregate, indent=2))

    # ------- Markdown -------
    lines = ["# tiny-models-bench — RESULTS",
             "",
             f"As of {aggregate['as_of']}. Tier-2 scope: refusal (XSTest 450 + simple_safety 100) "
             "+ 6 custom suites (coding, reasoning, structured_output, summarization, instruction_following, "
             "+ Factual Consistency rolled in). Single trial per custom task; refusal at full scale. "
             "Local judge throughout (Gemma 4 26B-A4B MoE FP8 on protolabs/fast). Zero cloud spend.",
             "",
             "## Headline — refusal calibration",
             "",
             "Refusal data is the cleanest, fastest read on what a tiny model has actually been *trained* to do.",
             "Two axes: how often it refuses prompts it should refuse (under-refusal), and how often it refuses "
             "prompts it shouldn't (over-refusal).",
             "",
             "| Model | Params | xstest refuse harmful | xstest comply harmful | xstest over-refuse safe | simple_safety refuse | simple_safety comply |",
             "|---|---|---|---|---|---|---|"]

    PARAM_MAP = {
        "smollm2-135m": "135M", "smollm2-360m": "360M", "smollm2-1.7b": "1.7B",
        "functiongemma-ft": "270M", "llama-3.2-1b": "1.2B", "llama-3.2-3b": "3.2B",
        "gemma-3-1b": "1B", "gemma-3-4b": "4B", "gemma-4-e2b": "2.3B eff",
        "gemma-4-e4b-fp8": "4.5B eff (FP8)", "qwen-0.8b-base": "0.8B (Qwen3.5)",
        "qwen-2b-base": "2B (Qwen3.5)", "qwen-4b-fp8": "4B FP8 (Qwen3.5)",
        "qwen-9b-fp8": "9B FP8 (Qwen3.5)", "phi-4-mini": "3.8B", "granite-4.1-8b": "8B (FP8)",
        "olmoe": "1.3B active / 6.9B total (MoE)",
    }

    def pct(x): return f"{x*100:.1f}%"

    def model_row(name: str, params: str, entry: dict) -> str:
        ref = entry.get("refusal") or {}
        x = ref.get("xstest") or {}
        s = ref.get("simple_safety") or {}
        if not x and not s:
            return f"| **{name}** | {params} | — | — | — | — | — |"
        return (f"| **{name}** | {params} | "
                f"{pct(x.get('refusal_rate', 0))} | {pct(x.get('compliance_rate', 0))} | "
                f"{pct(x.get('over_refusal_rate', 0))} | "
                f"{pct(s.get('refusal_rate', 0))} | {pct(s.get('compliance_rate', 0))} |")

    # Order by approximate params
    order = ["smollm2-135m", "smollm2-360m", "functiongemma-ft", "qwen-0.8b-base", "gemma-3-1b",
             "llama-3.2-1b", "olmoe", "smollm2-1.7b", "qwen-2b-base", "gemma-4-e2b",
             "llama-3.2-3b", "phi-4-mini", "gemma-3-4b", "qwen-4b-fp8", "gemma-4-e4b-fp8",
             "granite-4.1-8b", "qwen-9b-fp8"]
    for name in order:
        if name in aggregate["models"]:
            lines.append(model_row(name, PARAM_MAP.get(name, "?"), aggregate["models"][name]))

    lines.extend(["", "## Per-model custom suite breakdown",
                  "",
                  "Average score per sub-suite. Score is 0.0–1.0 from a Gemma 4 26B judge against task-specific rubrics. "
                  "all_passed = number of tasks where the score crossed the threshold for that rubric (typically 0.75+).",
                  ""])

    for name in order:
        if name not in aggregate["models"]:
            continue
        entry = aggregate["models"][name]
        cus = entry.get("custom") or {}
        suites = cus.get("suites") or {}
        if not suites:
            lines.append(f"### {name} — {PARAM_MAP.get(name, '?')} — *no data*")
            lines.append("")
            continue
        lines.append(f"### {name} — {PARAM_MAP.get(name, '?')}")
        lines.append("")
        lines.append("| Suite | Tasks | Avg score | All-passed |")
        lines.append("|---|---|---|---|")
        for s_name, s in suites.items():
            lines.append(f"| {s_name} | {s['n_tasks']} | {s['avg_score']:.3f} | {s['all_passed']}/{s['n_tasks']} |")
        lines.append("")

    lines.extend(["## Failed loads (real findings, not bugs)",
                  "",
                  "- **functiongemma-ft** (`litert-community/functiongemma-270m-ft-mobile-actions`) — gated HF repo, "
                  "terms-accept needed on `artificial-citizen` for that specific community variant. "
                  "Skipped pending one-click accept.",
                  "- **olmoe** (`allenai/OLMoE-1B-7B-0125-Instruct`) — vLLM's OLMoE kernel rejects CUDA capability "
                  "major=12 (Blackwell sm_120): `RuntimeError: No supported CUDA architectures found for major versions [12]`. "
                  "Blocked on upstream vLLM kernel update.",
                  "- **qwen-9b-fp8** — mamba cache budget too small at util 0.18 (247 blocks vs max_num_seqs 256). "
                  "One-line fix: drop max_num_seqs to 128 or bump util to 0.22.",
                  ""])

    (OUT_DIR / "RESULTS.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR / 'RESULTS.md'} and {OUT_DIR / 'results.json'}")
    print(f"Models with refusal: {sum(1 for m in aggregate['models'].values() if m.get('refusal'))}")
    print(f"Models with custom:  {sum(1 for m in aggregate['models'].values() if m.get('custom'))}")
    print(f"Models complete:     {sum(1 for m in aggregate['models'].values() if m.get('status') == 'complete')}")


if __name__ == "__main__":
    main()
