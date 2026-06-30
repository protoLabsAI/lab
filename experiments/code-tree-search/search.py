#!/usr/bin/env python3
"""Execution-grounded tree search over a code model, verified by hidden tests.

MARS² (TsinghuaC3I/MARTI) does *learned* multi-agent tree search for code gen — RL-trained,
8x80G+. We can't train Ornith (it's already maximally RL'd, and the rig is 2 cards), but the
*test-time* half of the idea fits us today and is actually GROUNDED where MARS² uses a learned
value: our `code_exec` grader runs the candidate against the hidden tests, so the search is
steered by real pass/fail, not a critic's guess.

Modes (all use the same generation budget knobs so they're comparable):
  greedy1 : one sample, no search   (the pass@1 baseline)
  bestof  : k independent samples, keep the best   (search without refinement)
  tree    : beam search — sample k, score, then expand the top-B partial solutions by showing
            the model its code + the FAILING tests and asking for a fix; keep top-B; repeat.
            This is the lever: execution feedback flows back into generation.

Verdict knob is `solved` (all hidden tests pass). Reports solved-rate, mean best-score, and
generations spent so the lift is always read against its cost.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import yaml
from openai import OpenAI

LAB = os.path.expanduser("~/dev/lab")
sys.path.insert(0, os.path.join(LAB, "evals"))
from graders.code_exec import CodeExecGrader  # noqa: E402


def _client() -> OpenAI:
    # gateway key from evals/.env
    env = {}
    envp = os.path.join(LAB, "evals", ".env")
    if os.path.exists(envp):
        for ln in open(envp):
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.strip().split("=", 1)
                env.setdefault(k, v)
    key = env.get("GATEWAY_API_KEY") or os.environ.get("GATEWAY_API_KEY", "not-needed")
    return OpenAI(base_url=os.environ.get("GATEWAY_BASE", "http://ava:4000/v1"), api_key=key)


def _ask(client, model, prompt, temperature) -> str:
    last = ""
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
            m = r.choices[0].message
            return (m.content or getattr(m, "reasoning_content", "") or "")
        except Exception as e:  # transient gateway/backend 500s shouldn't kill the run
            last = str(e)[:100]
            time.sleep(2 * (attempt + 1))
    print(f"    [ask failed after retries: {last}]", flush=True)
    return ""


def _grade(task, output_text):
    g = task["graders"][0]
    grader = CodeExecGrader(dimension="correctness", tests=g["tests"], entry=g.get("entry"),
                            setup=g.get("setup", ""), timeout=g.get("timeout", 10))
    res = grader.grade({}, {"output": output_text})
    return res.score, res.metadata


def _refine_prompt(task, code, meta):
    fails = [r for r in meta.get("results", []) if not r["passed"]][:4]
    lines = "\n".join(f"  - `{r['test']}`  -> {r['error'] or 'failed'}" for r in fails)
    return (f"{task['prompt']}\n\nYour previous solution:\n```python\n{code}\n```\n\n"
            f"It FAILED these hidden tests:\n{lines}\n\n"
            "Find the bug and return a corrected full solution. Return only the function in a Python code block.")


def solve(client, model, task, mode, k, beam, rounds, temp):
    gens = 0
    # round 0: k initial samples
    cands = []
    for i in range(1 if mode == "greedy1" else k):
        txt = _ask(client, model, task["prompt"], 0.0 if mode == "greedy1" else temp)
        gens += 1
        sc, meta = _grade(task, txt)
        cands.append((sc, txt, meta))
    cands.sort(key=lambda c: c[0], reverse=True)
    best = cands[0][0]
    if mode in ("greedy1", "bestof") or best >= 1.0:
        return {"solved": best >= 1.0, "best": best, "gens": gens, "round": 0}

    # tree: refine the top-beam partial solutions using execution feedback
    frontier = cands[:beam]
    for rd in range(1, rounds + 1):
        children = []
        for sc, txt, meta in frontier:
            if sc >= 1.0:
                return {"solved": True, "best": 1.0, "gens": gens, "round": rd - 1}
            code = meta.get("code", txt)
            rp = _refine_prompt(task, code, meta)
            for _ in range(max(1, k // beam)):
                ctxt = _ask(client, model, rp, temp)
                gens += 1
                csc, cmeta = _grade(task, ctxt)
                children.append((csc, ctxt, cmeta))
        pool = sorted(frontier + children, key=lambda c: c[0], reverse=True)
        frontier = pool[:beam]
        best = max(best, frontier[0][0])
        if best >= 1.0:
            return {"solved": True, "best": 1.0, "gens": gens, "round": rd}
    return {"solved": best >= 1.0, "best": best, "gens": gens, "round": rounds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protolabs/smart")
    ap.add_argument("--tasks", default=os.path.join(LAB, "evals/tasks/coding/hard.yaml"))
    ap.add_argument("--modes", default="greedy1,tree")
    ap.add_argument("--k", type=int, default=4, help="samples per round")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3, help="refinement rounds (tree)")
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    client = _client()
    suite = yaml.safe_load(open(args.tasks))
    tasks = suite["tests"]
    modes = args.modes.split(",")
    print(f"model={args.model}  tasks={len(tasks)}  k={args.k} beam={args.beam} rounds={args.rounds}\n")

    agg = {m: {"solved": 0, "best": 0.0, "gens": 0} for m in modes}
    for t in tasks:
        row = f"{t['id']:22}"
        for m in modes:
            t0 = time.time()
            r = solve(client, args.model, t, m, args.k, args.beam, args.rounds, args.temp)
            agg[m]["solved"] += int(r["solved"]); agg[m]["best"] += r["best"]; agg[m]["gens"] += r["gens"]
            disp = "PASS" if r["solved"] else f"{r['best']:.2f}"
            row += f"  {m}={disp} (g{r['gens']} r{r['round']} {time.time()-t0:.0f}s)"
        print(row)
    n = len(tasks)
    print("\n=== summary ===")
    for m in modes:
        a = agg[m]
        print(f"  {m:8} solved {a['solved']}/{n}  mean-best {a['best']/n:.3f}  total-gens {a['gens']}")


if __name__ == "__main__":
    main()
