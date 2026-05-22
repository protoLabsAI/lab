"""Run LoCoDiff against our RLM.

Usage:
  set -a; source ~/.proto/.env; set +a
  export GATEWAY_API_KEY="$LITELLM_API_KEY"

  # Sanity (3 smallest tasks, concurrency 1):
  uv run python experiments/rlm/eval/run_locodiff.py --n 3 --concurrency 1 --strategy small

  # Real batch (5 per quartile, concurrency 2):
  uv run python experiments/rlm/eval/run_locodiff.py --n 20 --concurrency 2 --strategy stratified
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make package importable when run as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.locodiff import LoCoDiffTask, bucket, list_tasks, load_task, score
from rlm import RLM, RLMConfig

PROMPTS_DIR = Path("/tmp/LoCoDiff-bench/locodiff-250425/prompts")
RESULTS_DIR = Path("/home/ava/dev/lab/experiments/rlm/results")

LOCODIFF_QUERY = (
    "The variable `git_log` holds the output of `git log -p --cc --topo-order --reverse` "
    "for a single file. Reconstruct the EXACT current state of that file by applying every "
    "diff in order. Output the file content verbatim — do not fix bugs, do not add or remove "
    "anything that isn't in the final state. Use Python (parse the diffs, apply hunks; "
    "the `unidiff` library is NOT available — write your own parser or apply line-by-line). "
    "When you have the reconstructed file as a string, store it in a variable and emit "
    "FINAL_VAR(<that_var_name>)."
)


def select_tasks(
    strategy: str, n: int, all_paths: list[Path], only_bucket: str | None = None
) -> list[Path]:
    if only_bucket:
        filtered = [p for p in all_paths if bucket(p.stat().st_size) == only_bucket]
        return sorted(filtered, key=lambda p: p.stat().st_size)[:n]
    if strategy == "small":
        return sorted(all_paths, key=lambda p: p.stat().st_size)[:n]
    if strategy == "stratified":
        per_q = max(1, n // 4)
        by_q: dict[str, list[Path]] = defaultdict(list)
        for p in all_paths:
            by_q[bucket(p.stat().st_size)].append(p)
        out: list[Path] = []
        for q in ("Q1", "Q2", "Q3", "Q4"):
            out.extend(sorted(by_q[q], key=lambda p: p.stat().st_size)[:per_q])
        return out[:n]
    if strategy == "first":
        return sorted(all_paths)[:n]
    raise ValueError(f"unknown strategy: {strategy}")


async def run_one(rlm: RLM, task_path: Path) -> dict:
    task = load_task(task_path)
    t0 = time.perf_counter()
    err: str | None = None
    traj = None
    try:
        traj = await rlm.completion(
            query=LOCODIFF_QUERY,
            context=task.git_log,
            context_var="git_log",
            context_meta={
                "type": "git log text (multi-commit, with diffs)",
                "target_file": task.target_path,
                "expected_output_bytes": task.expected_bytes,
            },
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    wall = time.perf_counter() - t0

    passed = score(traj.final, task.expected) if traj else False

    return {
        "name": task.name,
        "bucket": bucket(task.prompt_bytes),
        "prompt_bytes": task.prompt_bytes,
        "expected_bytes": task.expected_bytes,
        "passed": passed,
        "wall_s": round(wall, 2),
        "tokens": int(traj.totals["tokens"]) if traj else 0,
        "steps": int(traj.totals["steps"]) if traj else 0,
        "leaf_calls": int(traj.totals["leaf_calls"]) if traj else 0,
        "terminated": traj.terminated_reason if traj else "exception",
        "error": err or (traj.error if traj else None),
        "session_id": traj.session_id if traj else None,
        "predicted_bytes": len(traj.final) if (traj and traj.final) else 0,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--strategy", choices=("small", "stratified", "first"), default="small")
    ap.add_argument("--bucket", choices=("Q1", "Q2", "Q3", "Q4"), default=None,
                    help="restrict to one bucket (overrides strategy)")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-wall", type=float, default=600.0)
    ap.add_argument("--max-tokens", type=int, default=400_000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not PROMPTS_DIR.exists():
        print(f"FATAL: {PROMPTS_DIR} not found. Clone LoCoDiff-bench to /tmp first.")
        return 1

    paths = list_tasks(PROMPTS_DIR)
    selected = select_tasks(args.strategy, args.n, paths, only_bucket=args.bucket)
    selector = args.bucket or args.strategy
    print(f"[locodiff] {len(selected)} tasks selected ({selector})")
    for p in selected:
        size_kb = p.stat().st_size / 1024
        print(f"  {bucket(p.stat().st_size)} {size_kb:6.1f} KB  {p.name}")

    cfg = RLMConfig(
        max_steps=args.max_steps,
        max_wall_seconds=args.max_wall,
        max_tokens=args.max_tokens,
    )
    rlm = RLM(cfg)

    sem = asyncio.Semaphore(args.concurrency)

    async def _bounded(p: Path) -> dict:
        async with sem:
            print(f"[locodiff] start {p.name}")
            r = await run_one(rlm, p)
            print(
                f"[locodiff] done  {r['name']}  "
                f"{'PASS' if r['passed'] else 'FAIL'}  "
                f"{r['wall_s']}s  {r['steps']} steps  "
                f"{r['tokens']} tok  {r['leaf_calls']} leaf  "
                f"({r['terminated']})"
            )
            return r

    t0 = time.perf_counter()
    results = await asyncio.gather(*(_bounded(p) for p in selected))
    total_wall = time.perf_counter() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or RESULTS_DIR / f"locodiff-{int(time.time())}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[locodiff] wrote {out_path}")

    # Summary table by bucket
    print("\n=== summary by bucket ===")
    by_b: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_b[r["bucket"]].append(r)
    print(f"{'bkt':4} {'n':>3} {'pass':>6} {'avg_s':>7} {'avg_tok':>8} {'avg_leaf':>9}")
    for b in ("Q1", "Q2", "Q3", "Q4"):
        rs = by_b.get(b, [])
        if not rs:
            continue
        n = len(rs)
        passed = sum(1 for r in rs if r["passed"])
        print(
            f"{b:4} {n:>3} "
            f"{passed}/{n:<3}  "
            f"{sum(r['wall_s'] for r in rs)/n:>7.2f} "
            f"{sum(r['tokens'] for r in rs)/n:>8.0f} "
            f"{sum(r['leaf_calls'] for r in rs)/n:>9.2f}"
        )
    total_pass = sum(1 for r in results if r["passed"])
    print(f"\nOVERALL  {total_pass}/{len(results)}  total wall {total_wall:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
