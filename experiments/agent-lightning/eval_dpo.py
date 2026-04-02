"""Evaluate DPO adapter vs baseline on protoResearcher tasks.

Runs eval tasks twice:
  1. Baseline: vLLM serving Qwen 9B (no adapter)
  2. DPO: vLLM serving Qwen 9B + LoRA adapter

Each response scored by Sonnet 4.6 LLM judge. Generates comparison report.

Usage:
    cd ~/dev/lab
    export GATEWAY_API_KEY=<from-infisical>

    # Full eval (swaps vLLM models automatically)
    .venv/bin/python experiments/agent-lightning/eval_dpo.py

    # Eval only current model (no swap, just score)
    .venv/bin/python experiments/agent-lightning/eval_dpo.py --current-only

    # Specific tasks
    .venv/bin/python experiments/agent-lightning/eval_dpo.py --tasks simple_question,paper_analysis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import llm_judge

RESEARCHER_URL = "http://localhost:7872/api/chat"
TASKS_PATH = Path("/home/ava/dev/protoResearcher/evals/tasks.json")
RESULTS_DIR = Path(__file__).parent / "results"
SWAP_SCRIPT = Path("/home/ava/dev/lab/models/vllm-swap.sh")

BASELINE_MODEL = "qwen-9b"
DPO_ADAPTER = "/mnt/data/training/researcher/dpo-9b-v2"
DPO_MERGED = "/mnt/data/training/researcher/dpo-9b-v2-merged"


def load_tasks(filter_ids: list[str] | None = None) -> list[dict]:
    tasks = json.loads(TASKS_PATH.read_text())
    usable = [t for t in tasks if t["prompt"] and not t["prompt"].startswith("/")]
    if filter_ids:
        usable = [t for t in usable if t["id"] in filter_ids]
    return usable


def swap_model(config: str):
    """Swap vLLM model via swap script."""
    print(f"\n>>> Swapping vLLM to {config}...")
    result = subprocess.run(
        ["bash", str(SWAP_SCRIPT), config],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  Swap failed: {result.stderr}")
        raise RuntimeError(f"vllm-swap failed for {config}")
    print(f"  Model swapped to {config}")
    time.sleep(5)  # Extra settle time


def stop_vllm():
    """Stop any running vLLM process."""
    subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
    # Wait for port to free
    for _ in range(15):
        time.sleep(2)
        r = subprocess.run(["lsof", "-ti", ":8000"], capture_output=True)
        if r.returncode != 0:
            return
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], capture_output=True)
    time.sleep(3)


def start_merged_server():
    """Start vLLM serving the merged DPO model."""
    print(f"\n>>> Starting vLLM with merged DPO model...")
    stop_vllm()
    time.sleep(3)

    vllm_bin = os.path.expanduser("~/dev/vllm-env/bin/vllm")
    cmd = [
        vllm_bin, "serve", DPO_MERGED,
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.85",
        "--reasoning-parser", "qwen3",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    print("  Waiting for merged DPO model to load...", end="", flush=True)
    for _ in range(60):
        time.sleep(3)
        try:
            r = httpx.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200:
                print(" Ready!")
                return proc
        except Exception:
            print(".", end="", flush=True)

    proc.kill()
    raise RuntimeError("vLLM merged DPO model failed to start within 3 minutes")


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


async def evaluate_tasks(
    tasks: list[dict], label: str, judge_client,
) -> list[dict]:
    """Run all tasks and score with LLM judge."""
    results = []
    for task in tasks:
        sid = f"dpo-eval-{label}-{task['id']}-{int(time.time())}"
        print(f"  [{task['id']:30s}] ", end="", flush=True)

        t0 = time.monotonic()
        response = await run_task(task["prompt"], sid)
        elapsed = time.monotonic() - t0

        score = await llm_judge.judge_response(task, response, client=judge_client)
        print(f"composite={score.composite:.3f} ({elapsed:.1f}s)")

        results.append({
            "task_id": task["id"],
            "category": task["category"],
            "composite": score.composite,
            "judge_scores": score.to_dict(),
            "elapsed_s": round(elapsed, 1),
            "response_length": len(response),
            "response_preview": response[:500],
        })

    return results


def summarize(results: list[dict]) -> dict:
    composites = [r["composite"] for r in results]
    by_cat = {}
    for cat in ("simple", "medium", "complex"):
        cat_rs = [r for r in results if r["category"] == cat]
        if cat_rs:
            cat_c = [r["composite"] for r in cat_rs]
            by_cat[cat] = round(sum(cat_c) / len(cat_c), 3)
    return {
        "avg": round(sum(composites) / len(composites), 3) if composites else 0,
        "by_category": by_cat,
        "by_task": {r["task_id"]: r["composite"] for r in results},
    }


def generate_report(baseline: dict, dpo: dict, baseline_results: list, dpo_results: list) -> str:
    delta = dpo["avg"] - baseline["avg"]
    winner = "DPO" if delta > 0.01 else ("Baseline" if delta < -0.01 else "Tie")

    lines = [
        "# DPO Adapter Evaluation",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Base model**: Qwen/Qwen3.5-9B",
        f"**DPO adapter**: {DPO_ADAPTER}",
        f"**Judge**: Claude Sonnet 4.6",
        "",
        "## Overall",
        "",
        f"| Variant | Avg Score |",
        f"|---------|:---------:|",
        f"| Baseline 9B | **{baseline['avg']:.3f}** |",
        f"| 9B + DPO LoRA | **{dpo['avg']:.3f}** |",
        f"| Delta | **{delta:+.3f}** |",
        f"| Winner | **{winner}** |",
        "",
        "## By Category",
        "",
        "| Category | Baseline | DPO | Delta |",
        "|----------|:--------:|:---:|:-----:|",
    ]

    all_cats = sorted(set(list(baseline["by_category"].keys()) + list(dpo["by_category"].keys())))
    for cat in all_cats:
        b = baseline["by_category"].get(cat, 0)
        d = dpo["by_category"].get(cat, 0)
        lines.append(f"| {cat} | {b:.3f} | {d:.3f} | {d - b:+.3f} |")

    lines.extend(["", "## Per Task", "",
                   "| Task | Baseline | DPO | Delta |",
                   "|------|:--------:|:---:|:-----:|"])

    all_tasks = sorted(set(list(baseline["by_task"].keys()) + list(dpo["by_task"].keys())))
    for tid in all_tasks:
        b = baseline["by_task"].get(tid, 0)
        d = dpo["by_task"].get(tid, 0)
        marker = " **" if abs(d - b) > 0.1 else ""
        lines.append(f"| {tid} | {b:.3f} | {d:.3f} | {d - b:+.3f}{marker} |")

    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO adapter vs baseline")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task IDs to evaluate")
    parser.add_argument("--current-only", action="store_true",
                        help="Only evaluate current model (no swap)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Rounds per variant (avg over multiple runs)")
    args = parser.parse_args()

    filter_ids = args.tasks.split(",") if args.tasks else None
    tasks = load_tasks(filter_ids)
    judge_client = llm_judge._make_client()

    print(f"Tasks: {len(tasks)}")
    print(f"Rounds: {args.rounds}")
    print(f"Adapter: {DPO_ADAPTER}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.current_only:
        print("\n=== Evaluating current model ===")
        results = await evaluate_tasks(tasks, "current", judge_client)
        summary = summarize(results)
        print(f"\nAvg composite: {summary['avg']:.3f}")
        for cat, score in summary["by_category"].items():
            print(f"  {cat}: {score:.3f}")
        return

    # --- Baseline run ---
    swap_model(BASELINE_MODEL)
    print("\n=== Baseline (Qwen 9B) ===")
    all_baseline = []
    for rnd in range(args.rounds):
        if args.rounds > 1:
            print(f"\n  Round {rnd + 1}/{args.rounds}")
        results = await evaluate_tasks(tasks, f"baseline-r{rnd}", judge_client)
        all_baseline.extend(results)

    baseline_summary = summarize(all_baseline)
    print(f"\nBaseline avg: {baseline_summary['avg']:.3f}")

    # --- DPO run ---
    merged_proc = start_merged_server()
    try:
        print("\n=== DPO (9B + merged LoRA) ===")
        all_dpo = []
        for rnd in range(args.rounds):
            if args.rounds > 1:
                print(f"\n  Round {rnd + 1}/{args.rounds}")
            results = await evaluate_tasks(tasks, f"dpo-r{rnd}", judge_client)
            all_dpo.extend(results)

        dpo_summary = summarize(all_dpo)
        print(f"\nDPO avg: {dpo_summary['avg']:.3f}")
    finally:
        merged_proc.terminate()
        merged_proc.wait(timeout=10)

    # --- Report ---
    report = generate_report(baseline_summary, dpo_summary, all_baseline, all_dpo)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report_path = RESULTS_DIR / f"dpo_eval_{ts}.md"
    report_path.write_text(report)

    data_path = RESULTS_DIR / f"dpo_eval_{ts}.json"
    data_path.write_text(json.dumps({
        "baseline": {"summary": baseline_summary, "results": all_baseline},
        "dpo": {"summary": dpo_summary, "results": all_dpo},
    }, indent=2, default=str))

    print(f"\n{'='*60}")
    print(report)
    print(f"\nSaved: {report_path}")
    print(f"Data:  {data_path}")

    # Restore daily driver
    swap_model("qwen-27b-int4")


if __name__ == "__main__":
    asyncio.run(main())
