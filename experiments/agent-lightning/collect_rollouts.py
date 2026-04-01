"""Collect rollouts from protoResearcher for GRPO training.

Runs all eval tasks multiple times, collecting (prompt, response) pairs
scored by the LLM judge. Outputs a HuggingFace-compatible dataset.

Usage:
    cd ~/dev/lab
    source ~/.config/gateway/identity && ...  # set GATEWAY_API_KEY
    .venv/bin/python experiments/agent-lightning/collect_rollouts.py --rounds 5
    .venv/bin/python experiments/agent-lightning/collect_rollouts.py --rounds 10 --output /mnt/data/training/researcher/rollouts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import llm_judge

RESEARCHER_URL = "http://localhost:7872/api/chat"
TASKS_PATH = Path("/home/ava/dev/protoResearcher/evals/tasks.json")
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "rollouts"


def load_tasks() -> list[dict]:
    tasks = json.loads(TASKS_PATH.read_text())
    # Skip empty input edge case and command-style tasks
    return [t for t in tasks if t["prompt"] and not t["prompt"].startswith("/")]


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


async def collect(args):
    tasks = load_tasks()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_client = llm_judge._make_client()

    print(f"Tasks: {len(tasks)}")
    print(f"Rounds: {args.rounds}")
    print(f"Expected rollouts: {len(tasks) * args.rounds}")
    print(f"Output: {output_dir}")
    print()

    rollouts = []

    for rnd in range(1, args.rounds + 1):
        print(f"=== Round {rnd}/{args.rounds} ===")
        for task in tasks:
            tid = task["id"]
            sid = f"rollout-r{rnd}-{tid}-{int(time.time())}"

            print(f"  [{tid:30s}] ", end="", flush=True)
            t0 = time.monotonic()
            response = await run_task(task["prompt"], sid)
            elapsed = time.monotonic() - t0

            # Score with LLM judge
            score = await llm_judge.judge_response(task, response, client=judge_client)
            print(f"composite={score.composite:.3f} ({elapsed:.1f}s, {len(response)} chars)")

            rollouts.append({
                "prompt": task["prompt"],
                "response": response,
                "task_id": tid,
                "category": task["category"],
                "round": rnd,
                "reward": score.composite,
                "judge_scores": score.to_dict(),
                "elapsed_s": round(elapsed, 1),
                "response_length": len(response),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Save as JSONL (easy to load with HF datasets)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"rollouts_{ts}.jsonl"
    with open(jsonl_path, "w") as f:
        for r in rollouts:
            f.write(json.dumps(r) + "\n")

    # Also save summary stats
    rewards = [r["reward"] for r in rollouts]
    summary = {
        "total_rollouts": len(rollouts),
        "rounds": args.rounds,
        "tasks": len(tasks),
        "avg_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "zero_reward_count": sum(1 for r in rewards if r == 0),
        "by_category": {},
    }
    for cat in ("simple", "medium", "complex"):
        cat_rewards = [r["reward"] for r in rollouts if r["category"] == cat]
        if cat_rewards:
            summary["by_category"][cat] = {
                "count": len(cat_rewards),
                "avg_reward": round(sum(cat_rewards) / len(cat_rewards), 3),
            }

    summary_path = output_dir / f"rollouts_{ts}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Collected {len(rollouts)} rollouts")
    print(f"Avg reward: {summary['avg_reward']:.3f}")
    print(f"Zero-reward: {summary['zero_reward_count']}/{len(rollouts)}")
    print(f"Saved to: {jsonl_path}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Collect rollouts for GRPO training")
    parser.add_argument("--rounds", type=int, default=5, help="Number of collection rounds")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output directory")
    args = parser.parse_args()
    asyncio.run(collect(args))


if __name__ == "__main__":
    main()
