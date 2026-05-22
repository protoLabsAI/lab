"""End-to-end smoke test for the RLM scaffolding.

Builds a synthetic long-ish corpus (1000 items), asks a distributional question,
runs the RLM, asserts a sane answer landed in the trajectory, and prints stats.

Usage:
  cd ~/dev/lab
  uv run python experiments/rlm/scripts/smoke.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

# Make the experiments/rlm package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlm import RLM, RLMConfig


def build_corpus(n: int = 1000, seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    departments = ["eng", "sales", "ops", "marketing", "research"]
    return [
        {
            "id": i,
            "department": rng.choice(departments),
            "region": rng.choice(["US", "EU", "APAC"]),
            "tenure_years": rng.randint(0, 15),
            "satisfaction": rng.randint(1, 5),
        }
        for i in range(n)
    ]


def ground_truth_count_eng_us(corpus: list[dict]) -> int:
    return sum(1 for r in corpus if r["department"] == "eng" and r["region"] == "US")


async def main() -> int:
    corpus = build_corpus()
    truth = ground_truth_count_eng_us(corpus)
    print(f"[smoke] corpus size={len(corpus)} ground_truth(eng+US)={truth}")

    cfg = RLMConfig(
        # Tighter budget for a smoke test
        max_steps=8,
        max_wall_seconds=120.0,
        max_tokens=80_000,
    )
    rlm = RLM(cfg)

    query = (
        "How many records in the dataset have department='eng' AND region='US'? "
        "Answer with just the integer."
    )

    print("[smoke] running RLM...")
    traj = await rlm.completion(
        query=query,
        context=corpus,
        context_var="ctx",
        context_meta={"description": "list[dict] employee records"},
    )

    print(f"[smoke] terminated_reason={traj.terminated_reason}")
    print(f"[smoke] final={traj.final!r}")
    print(f"[smoke] totals={traj.totals}")
    print(f"[smoke] turn count={len(traj.turns)} (planner+exec+leaf interleaved)")

    if traj.error:
        print(f"[smoke] ERROR: {traj.error}")
        return 1

    if traj.final is None:
        print("[smoke] FAIL: no final answer")
        return 1

    # Loose check — model may say "42" or "There are 42." — extract any int
    import re

    nums = re.findall(r"\d+", traj.final)
    if nums and int(nums[0]) == truth:
        print(f"[smoke] PASS: model said {nums[0]} == ground truth {truth}")
        return 0
    print(f"[smoke] SOFT-FAIL: model said {traj.final!r}, expected {truth}")
    return 2  # soft-fail — orchestration worked, model was just wrong


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
