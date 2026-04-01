"""A/B Evaluation: Original vs Optimized SOUL.md

Runs the full protoResearcher task suite twice — once with the original
(pre-APO) prompt and once with the Sonnet-optimized prompt — scoring
each response with the Sonnet 4.6 LLM judge.

Usage:
    cd ~/dev/lab
    source ~/.config/gateway/identity
    TOKEN=$(infisical login --method=universal-auth ...)
    export GATEWAY_API_KEY=$(infisical secrets get GATEWAY_API_KEY ...)

    .venv/bin/python experiments/agent-lightning/run_ab_eval.py
    .venv/bin/python experiments/agent-lightning/run_ab_eval.py --tasks simple_question,hf_model_search
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

import llm_judge

RESEARCHER_URL = "http://localhost:7872/api/chat"
SOUL_PATH = Path("/home/ava/dev/protoResearcher/config/SOUL.md")
TASKS_PATH = Path("/home/ava/dev/protoResearcher/evals/tasks.json")
RESULTS_DIR = Path(__file__).parent / "results"


def load_tasks(filter_ids: list[str] | None = None) -> list[dict]:
    tasks = json.loads(TASKS_PATH.read_text())
    if filter_ids:
        tasks = [t for t in tasks if t["id"] in filter_ids]
    return tasks


async def run_task(prompt: str, session_id: str, timeout: float = 180) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                RESEARCHER_URL,
                json={"message": prompt, "session_id": session_id},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"[error: {e}]"


async def evaluate_variant(
    variant_name: str,
    soul_content: str,
    tasks: list[dict],
    judge_client,
) -> dict:
    """Run all tasks with a given SOUL.md variant and judge each response."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {variant_name}")
    print(f"{'='*60}")

    # Write the SOUL.md variant
    SOUL_PATH.write_text(soul_content)
    # Brief pause for the container to pick up the change (bind mount)
    await asyncio.sleep(2)

    results = []
    for task in tasks:
        task_id = task["id"]
        prompt = task["prompt"]

        # Skip empty-input edge case for meaningful comparison
        if task_id == "edge_empty_input":
            print(f"  [{task_id:30s}] SKIP (edge case)")
            continue

        sid = f"ab-{variant_name}-{task_id}-{int(time.time())}"
        print(f"  [{task_id:30s}] Running...", end=" ", flush=True)

        t0 = time.monotonic()
        response = await run_task(prompt, sid)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Judge the response
        score = await llm_judge.judge_response(task, response, client=judge_client)
        print(f"composite={score.composite:.3f}  ({elapsed_ms}ms)")

        results.append({
            "task_id": task_id,
            "category": task["category"],
            "prompt": prompt[:80],
            "response_length": len(response),
            "elapsed_ms": elapsed_ms,
            "judge_score": score.to_dict(),
            "response_preview": response[:500],
        })

    return {
        "variant": variant_name,
        "results": results,
        "summary": _summarize(results),
    }


def _summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    composites = [r["judge_score"]["composite"] for r in results]
    by_cat = {}
    for cat in ("simple", "medium", "complex"):
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cat_composites = [r["judge_score"]["composite"] for r in cat_results]
            by_cat[cat] = {
                "count": len(cat_results),
                "avg_composite": round(sum(cat_composites) / len(cat_composites), 3),
                "avg_elapsed_ms": sum(r["elapsed_ms"] for r in cat_results) // len(cat_results),
            }
    return {
        "total_tasks": len(results),
        "avg_composite": round(sum(composites) / len(composites), 3),
        "min_composite": min(composites),
        "max_composite": max(composites),
        "avg_elapsed_ms": sum(r["elapsed_ms"] for r in results) // len(results),
        "by_category": by_cat,
    }


def _generate_report(original: dict, optimized: dict) -> str:
    """Generate a markdown comparison report."""
    os_ = original["summary"]
    ops = optimized["summary"]
    delta = ops["avg_composite"] - os_["avg_composite"]
    winner = "optimized" if delta > 0.01 else ("original" if delta < -0.01 else "tie")

    lines = [
        "# A/B Eval: Original vs APO-Optimized SOUL.md",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Judge**: Claude Sonnet 4.6 via ava gateway",
        f"**Agent model**: Qwen 27B FP8 TP=2",
        "",
        "## Overall",
        "",
        "| Metric | Original | Optimized | Delta |",
        "|--------|:--------:|:---------:|:-----:|",
        f"| Avg composite | {os_['avg_composite']:.3f} | {ops['avg_composite']:.3f} | {delta:+.3f} |",
        f"| Min composite | {os_['min_composite']:.3f} | {ops['min_composite']:.3f} | {ops['min_composite'] - os_['min_composite']:+.3f} |",
        f"| Avg latency (ms) | {os_['avg_elapsed_ms']} | {ops['avg_elapsed_ms']} | {ops['avg_elapsed_ms'] - os_['avg_elapsed_ms']:+d} |",
        "",
        f"**Winner: {winner}** (composite delta: {delta:+.3f})",
        "",
        "## By Category",
        "",
        "| Category | Original | Optimized | Delta |",
        "|----------|:--------:|:---------:|:-----:|",
    ]

    for cat in ("simple", "medium", "complex"):
        oc = os_.get("by_category", {}).get(cat, {})
        opc = ops.get("by_category", {}).get(cat, {})
        if oc and opc:
            d = opc["avg_composite"] - oc["avg_composite"]
            lines.append(f"| {cat} | {oc['avg_composite']:.3f} | {opc['avg_composite']:.3f} | {d:+.3f} |")

    lines += ["", "## Per-Task Breakdown", "", "| Task | Cat | Original | Optimized | Delta | Winner |", "|------|-----|:--------:|:---------:|:-----:|:------:|"]

    orig_by_task = {r["task_id"]: r for r in original["results"]}
    opt_by_task = {r["task_id"]: r for r in optimized["results"]}

    for task_id in orig_by_task:
        if task_id not in opt_by_task:
            continue
        o = orig_by_task[task_id]["judge_score"]["composite"]
        p = opt_by_task[task_id]["judge_score"]["composite"]
        d = p - o
        cat = orig_by_task[task_id]["category"]
        w = "opt" if d > 0.02 else ("orig" if d < -0.02 else "—")
        lines.append(f"| {task_id} | {cat} | {o:.3f} | {p:.3f} | {d:+.3f} | {w} |")

    lines += [
        "",
        "## Dimension Analysis",
        "",
        "| Dimension | Original | Optimized | Delta |",
        "|-----------|:--------:|:---------:|:-----:|",
    ]

    for dim in ("relevance", "completeness", "structure", "accuracy", "actionability"):
        o_avg = sum(r["judge_score"][dim] for r in original["results"]) / max(len(original["results"]), 1)
        p_avg = sum(r["judge_score"][dim] for r in optimized["results"]) / max(len(optimized["results"]), 1)
        lines.append(f"| {dim} | {o_avg:.3f} | {p_avg:.3f} | {p_avg - o_avg:+.3f} |")

    return "\n".join(lines)


async def run_ab(args):
    tasks = load_tasks(args.tasks.split(",") if args.tasks else None)
    print(f"Tasks: {len(tasks)}")
    print(f"Judge: {llm_judge.JUDGE_MODEL} via gateway")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_soul = (RESULTS_DIR / "original_soul.md").read_text()
    optimized_soul = (RESULTS_DIR / "best_soul.md").read_text()

    # Stash current SOUL.md so we can restore it
    current_soul = SOUL_PATH.read_text()

    judge_client = llm_judge._make_client()

    try:
        # Run original first
        original_results = await evaluate_variant("original", original_soul, tasks, judge_client)

        # Then optimized
        optimized_results = await evaluate_variant("optimized", optimized_soul, tasks, judge_client)
    finally:
        # Restore the SOUL.md that was active before the eval
        SOUL_PATH.write_text(current_soul)

    # Generate report
    report = _generate_report(original_results, optimized_results)

    # Save everything
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": ts,
        "judge_model": llm_judge.JUDGE_MODEL,
        "original": original_results,
        "optimized": optimized_results,
    }
    (RESULTS_DIR / f"ab_eval_{ts}.json").write_text(json.dumps(output, indent=2))
    report_path = RESULTS_DIR / f"ab_eval_{ts}.md"
    report_path.write_text(report)

    print(f"\n{report}")
    print(f"\nResults saved to: {RESULTS_DIR / f'ab_eval_{ts}.json'}")
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="A/B eval: original vs optimized SOUL.md")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs (default: all)")
    args = parser.parse_args()
    asyncio.run(run_ab(args))


if __name__ == "__main__":
    main()
