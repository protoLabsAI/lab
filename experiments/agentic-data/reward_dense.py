"""Dense, deterministic reward for tau2 sims — recover the partial-credit gradient
that tau2's product-composition throws away.

tau2's official reward AND-composes at TWO levels (both are products of 0/1 booleans):
  - across reward_basis components:  reward = db * env_assertion * action * communicate
  - within a component (per the evaluator code):  env_assertion_reward *= res.reward
so a task with 3 env-assertions where the agent satisfies 2 scores 0.0 — terminal-sparse.

This recomputes each basis component as a FRACTION of its checks met, then aggregates the
basis components as a (weighted) mean. Every input boolean is the same deterministic check
tau2 already computed (env_assertions[].met, action_checks[].action_match, db_check.db_reward,
communicate_checks[].met) — NO LLM in the path. reward_basis is respected: only components
tau2 graded this task on contribute (matches tau2 semantics; no credit for incidental matches).

Two aggregation modes per component:
  - "fraction"       : mean of the check booleans (default; smooth gradient)
  - "monotone_prefix": longest satisfied PREFIX / total (for ordered action chains — resists
                       gaming: can't harvest check k without checks 1..k-1 genuinely present)

tool_type weighting (anti-gaming): write/irreversible actions weigh more than read-only ones,
since reads are cheap to spam. Default WRITE=1.0, READ=0.3, THINK/GENERIC=0.5.

INVARIANT (asserted in the self-test): if tau2's official reward == 1.0, reward_dense == 1.0.
A full success strictly dominates every partial one, so this is safe as PBRS shaping over the
unchanged terminal objective (Phi = reward_dense; keep tau2's 0/1 as the true objective).

  python reward_dense.py <results_dir_or_json>   # self-test + gradient-recovery report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# basis name -> (reward_info field, per-check "met" accessor)
_TOOL_W = {"write": 1.0, "read": 0.3, "think": 0.5, "generic": 0.5}


def _frac(bools: list[float], weights: list[float] | None = None) -> float:
    if not bools:
        return 0.0
    if weights is None:
        return sum(bools) / len(bools)
    tot = sum(weights)
    return sum(b * w for b, w in zip(bools, weights)) / tot if tot else 0.0


def _prefix(bools: list[float]) -> float:
    """longest satisfied prefix / total."""
    if not bools:
        return 0.0
    k = 0
    for b in bools:
        if b >= 1.0:
            k += 1
        else:
            break
    return k / len(bools)


def _component_score(name: str, ri: dict, mode: str) -> float | None:
    """Fraction (or prefix) of checks met for one basis component. None if not present."""
    if name == "DB":
        db = ri.get("db_check") or {}
        return float(db.get("db_reward", 0.0)) if "db_reward" in db else None
    if name == "ENV_ASSERTION":
        ea = ri.get("env_assertions")
        if not ea:
            return None
        bools = [1.0 if a.get("met") else 0.0 for a in ea]
        return _prefix(bools) if mode == "monotone_prefix" else _frac(bools)
    if name == "ACTION":
        ac = ri.get("action_checks")
        if not ac:
            return None
        bools = [1.0 if a.get("action_match") else 0.0 for a in ac]
        if mode == "monotone_prefix":
            return _prefix(bools)
        weights = [_TOOL_W.get((a.get("tool_type") or "generic").lower(), 0.5) for a in ac]
        return _frac(bools, weights)
    if name == "COMMUNICATE":
        cc = ri.get("communicate_checks")
        if not cc:
            return None
        return _frac([1.0 if c.get("met") else 0.0 for c in cc])
    if name == "NL_ASSERTION":
        # LLM-judged — deliberately EXCLUDED from the deterministic reward path.
        return None
    return None


def reward_dense(reward_info: dict, mode: str = "fraction",
                 basis_weights: dict | None = None) -> float | None:
    """Deterministic dense reward in [0,1] for one tau2 sim's reward_info.

    Returns None for ungradable tasks (empty reward_basis) — no gradient, exclude them.
    Respects reward_basis: only graded components contribute.
    """
    basis = reward_info.get("reward_basis") or []
    if not basis:
        return None  # ungradable — no criteria
    scores, weights = [], []
    for comp in basis:
        s = _component_score(comp, reward_info, mode)
        if s is None:
            # basis names a component but its checks are absent -> treat as 0 (unmet)
            s = 0.0
        scores.append(s)
        weights.append((basis_weights or {}).get(comp, 1.0))
    tot = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / tot if tot else None


def _iter_sims(path: str):
    p = Path(path)
    f = p if p.is_file() else next(iter(p.glob("**/results.json")), None)
    d = json.load(open(f))
    yield from (d.get("simulations") if isinstance(d, dict) else d)


def _selftest(path: str):
    import statistics as st
    sims = list(_iter_sims(path))
    graded = 0
    ungradable = 0
    violations = 0            # official==1.0 but dense!=1.0  (invariant breach)
    recovered = 0             # official<1.0 but dense>0      (newly-informative negative)
    dead = 0                  # official<1.0 and dense==0     (still no gradient)
    dense_of_fail = []
    for s in sims:
        ri = s.get("reward_info") or {}
        off = float(ri.get("reward", 0.0))
        dn = reward_dense(ri)
        if dn is None:
            ungradable += 1
            continue
        graded += 1
        assert 0.0 <= dn <= 1.0 + 1e-9, f"out of range: {dn}"
        if off >= 1.0 and abs(dn - 1.0) > 1e-9:
            violations += 1
        if off < 1.0:
            dense_of_fail.append(dn)
            if dn > 1e-9:
                recovered += 1
            else:
                dead += 1
    print(f"sims                : {len(sims)}")
    print(f"gradable            : {graded}   ungradable(empty basis): {ungradable}")
    print(f"INVARIANT breaches  : {violations}   (official==1.0 but dense<1.0 — must be 0)")
    nfail = len(dense_of_fail)
    print(f"failed sims         : {nfail}")
    print(f"  recovered gradient: {recovered}  ({recovered*100//max(nfail,1)}% of failures now score >0)")
    print(f"  still dead-zero   : {dead}")
    if dense_of_fail:
        buckets = {"0":0, "(0,.25]":0, "(.25,.5]":0, "(.5,.75]":0, "(.75,1)":0}
        for v in dense_of_fail:
            if v <= 0: buckets["0"]+=1
            elif v<=.25: buckets["(0,.25]"]+=1
            elif v<=.5: buckets["(.25,.5]"]+=1
            elif v<=.75: buckets["(.5,.75]"]+=1
            else: buckets["(.75,1)"]+=1
        print("  failed-sim dense distribution:")
        for k,c in buckets.items(): print(f"    {k:>9}: {c}")
        print(f"  mean dense over failures: {st.mean(dense_of_fail):.3f}")
    print()
    print("RESULT:", "PASS — invariant holds, gradient recovered" if violations==0 and recovered>0
          else "CHECK — see violations/recovery above")


if __name__ == "__main__":
    _selftest(sys.argv[1] if len(sys.argv) > 1
              else "/mnt/data/datasets/agentic-distill/_raw/tau2-telecom")
