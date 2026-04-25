"""
Planner/executor eval — runs all 20 tasks against 3 baselines and writes results.

Baselines:
  1. planner-only   — 27B thinking model handles everything (:8000)
  2. executor-only  — 35B no-thinking model handles everything (:8002)
  3. planner-executor — dual-model loop (this experiment)

Usage:
  cd experiments/companion-stack/pipes/llm-context/planner-executor
  python eval/run_eval.py [--baseline all|planner|executor|dual] [--task-id mt_001]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# make parent importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from tasks import ALL_TASKS
from models import LoopResult, Plan, PlanStep, ExecutionResult, ReviewDecision
from planner_executor import run as run_dual
from eval.metrics import RunSummary, score_result

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PLANNER_URL   = "http://localhost:8000/v1"
EXECUTOR_URL  = "http://localhost:8002/v1"


# ---------------------------------------------------------------------------
# Single-model baselines
# ---------------------------------------------------------------------------

SINGLE_SYSTEM = """\
You are an agentic assistant. Given a goal, reason through the steps needed \
and use available tools to complete it. When done, provide a concise final answer.
"""


def run_single(task_id: str, goal: str, model: str, base_url: str) -> LoopResult:
    """Run a task with a single model — no planner/executor split."""
    client = OpenAI(base_url=base_url, api_key="none")
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SINGLE_SYSTEM},
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    latency = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    success = len(content.strip()) > 20  # non-empty response = completed

    return LoopResult(
        task_id=task_id,
        goal=goal,
        success=success,
        total_steps=1,
        completed_steps=1 if success else 0,
        replans=0,
        retries=0,
        final_answer=content,
        plan=Plan(goal=goal, steps=[PlanStep(step_id=1, description="single-model response")], success_criteria=""),
        execution_results=[ExecutionResult(step_id=1, status="ok" if success else "error", output=content)],
        review=ReviewDecision(verdict="pass" if success else "abort"),
        plan_latency=0.0,
        execution_latency=latency,
        review_latency=0.0,
        total_latency=latency,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(name: str, tasks: list[dict], fn) -> RunSummary:
    summary = RunSummary(baseline=name)
    all_results = []

    for task in tasks:
        result = fn(task["id"], task["goal"])
        summary.scores.append(score_result(result, task["category"]))
        all_results.append(result.model_dump())

    # persist raw results
    out = RESULTS_DIR / f"{name}_{int(time.time())}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"  Saved raw results → {out}")

    summary.print_summary()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="all",
                        choices=["all", "planner", "executor", "dual"])
    parser.add_argument("--task-id", default=None,
                        help="Run a single task by ID (e.g. mt_001)")
    args = parser.parse_args()

    tasks = ALL_TASKS
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            print(f"Task {args.task_id!r} not found.")
            sys.exit(1)

    summaries: list[RunSummary] = []

    if args.baseline in ("all", "planner"):
        print("\n" + "=" * 60)
        print("BASELINE 1: planner-only (27B thinking, :8000)")
        s = run_baseline(
            "planner-only", tasks,
            lambda tid, goal: run_single(tid, goal, "local", PLANNER_URL),
        )
        summaries.append(s)

    if args.baseline in ("all", "executor"):
        print("\n" + "=" * 60)
        print("BASELINE 2: executor-only (35B no-thinking, :8002)")
        s = run_baseline(
            "executor-only", tasks,
            lambda tid, goal: run_single(tid, goal, "local-voice", EXECUTOR_URL),
        )
        summaries.append(s)

    if args.baseline in ("all", "dual"):
        print("\n" + "=" * 60)
        print("BASELINE 3: planner/executor dual-model loop")
        s = run_baseline("dual", tasks, lambda tid, goal: run_dual(tid, goal))
        summaries.append(s)

    # comparison table
    if len(summaries) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"  {'Baseline':<25} {'Success':>8} {'Avg latency':>14} {'Replans':>9}")
        print(f"  {'-'*25} {'-'*8} {'-'*14} {'-'*9}")
        for s in summaries:
            print(f"  {s.baseline:<25} {s.success_rate:>7.0%} {s.avg_total_latency:>13.2f}s {s.replan_rate:>9.2f}")


if __name__ == "__main__":
    main()
