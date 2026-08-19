#!/usr/bin/env python3
"""Local arena for kaggriculture agents.

Runs seeded, seat-swapped head-to-head games in parallel and reports
wins/ties/losses plus bank statistics.

Usage:
    .venv/bin/python arena.py agents/v1.py starter --games 20
    .venv/bin/python arena.py agents/v2.py agents/v1.py --games 50 --procs 16
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def play_one(args):
    a_path, b_path, seed, swap = args
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    pair = [b_path, a_path] if swap else [a_path, b_path]
    env.run(pair)
    final = env.steps[-1]
    rewards = [s.reward if s.reward is not None else 0.0 for s in final]
    statuses = [s.status for s in final]
    a_idx = 1 if swap else 0
    return {
        "seed": seed,
        "swap": swap,
        "a_bank": rewards[a_idx],
        "b_bank": rewards[1 - a_idx],
        "a_status": statuses[a_idx],
        "b_status": statuses[1 - a_idx],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--json", dest="json_out", help="write per-game results to this path")
    args = ap.parse_args()

    jobs = []
    for i in range(args.games):
        seed = args.seed0 + i
        jobs.append((args.agent_a, args.agent_b, seed, i % 2 == 1))

    results = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = {ex.submit(play_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            outcome = "W" if r["a_bank"] > r["b_bank"] else ("T" if r["a_bank"] == r["b_bank"] else "L")
            print(f"seed={r['seed']} swap={int(r['swap'])} {outcome}  "
                  f"A={r['a_bank']:>10.0f}  B={r['b_bank']:>10.0f}  "
                  f"({r['a_status']}/{r['b_status']})", flush=True)

    wins = sum(1 for r in results if r["a_bank"] > r["b_bank"])
    ties = sum(1 for r in results if r["a_bank"] == r["b_bank"])
    losses = len(results) - wins - ties
    errors = sum(1 for r in results if r["a_status"] != "DONE" or r["b_status"] != "DONE")
    a_banks = sorted(r["a_bank"] for r in results)
    b_banks = sorted(r["b_bank"] for r in results)

    def stats(xs):
        n = len(xs)
        return f"mean={sum(xs)/n:>9.0f}  min={xs[0]:>9.0f}  med={xs[n//2]:>9.0f}  max={xs[-1]:>9.0f}"

    print(f"\n{args.agent_a} vs {args.agent_b}: {len(results)} games")
    print(f"W-T-L: {wins}-{ties}-{losses}  ({100*wins/len(results):.0f}% win)  errors={errors}")
    print(f"A bank: {stats(a_banks)}")
    print(f"B bank: {stats(b_banks)}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"a": args.agent_a, "b": args.agent_b, "results": results}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
