#!/usr/bin/env python3
"""Properly-powered paired A/B against the champion.

Per-game bank SD between different agents is ~$11k, so resolving a $2k effect
at 95% needs >116 paired games. Anything smaller is unresolved, not refuted.

    ./powered_test.py variants/h12.py [n]
"""
import json
import math
import os
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(HERE, ".venv/bin/python")


def run(variant, n=116, seed0=50000, procs=28, opp="agents/v23.py"):
    out = f"/tmp/pt_{os.path.basename(variant).replace('.py','')}.json"
    subprocess.run([PY, os.path.join(HERE, "arena.py"), variant, opp,
                    "--games", str(n), "--procs", str(procs),
                    "--seed0", str(seed0), "--json", out],
                   cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    d = json.load(open(out))
    m = [r["a_bank"] - r["b_bank"] for r in d["results"]]
    n = len(m)
    mean = st.mean(m)
    sd = st.stdev(m)
    sem = sd / math.sqrt(n)
    sigma = abs(mean) / sem if sem else 0
    return {"variant": os.path.basename(variant), "n": n, "mean": mean,
            "sd": sd, "sem": sem, "sigma": sigma,
            "lo": mean - 1.96 * sem, "hi": mean + 1.96 * sem,
            "wins": sum(1 for x in m if x > 0)}


def fmt(r):
    verdict = ("WIN" if r["lo"] > 0 else
               "LOSS" if r["hi"] < 0 else "unresolved")
    return (f"{r['variant']:<12} n={r['n']:>3}  margin {r['mean']:>+8,.0f}  "
            f"95%CI [{r['lo']:>+8,.0f},{r['hi']:>+8,.0f}]  "
            f"{r['sigma']:>4.1f}s  {r['wins']:>3}/{r['n']}  {verdict}")


if __name__ == "__main__":
    v = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 116
    print(fmt(run(v, n)), flush=True)
