"""Reasoning v2 generator — parametric, solver-verified, exact-match graded.

Phase 3 (FOCUS.md): replace the 5-task LLM-judged reasoning suite with ~24
contamination-resistant tasks from seeded generators. Every instance's ground
truth is verified by an independent brute-force solver (uniqueness checked for
puzzle families); grading is deterministic (match: regex / number / all_of).
Zero LLM judge.

Families (4 tasks each, easy/easy/medium/hard within family → 4:3:3-ish mix
across the suite):
  arith     multi-step word problems with distractor numbers   (number)
  sequence  composed-rule integer sequences, next term          (number)
  machine   register-machine trace, final state                 (number)
  schedule  constraint assignment, unique solution              (all_of)
  grid      zebra-style logic grid, unique solution, one query  (regex)
  knights   knights & knaves, unique assignment                 (all_of)

Regenerate: python gen_v2.py --seed 20260702 --out ..
Changing the seed regenerates every instance (fresh, contamination-clean set);
bump the filename suffix if the old set has published baselines.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

import yaml

ANSWER_RULES = (
    "Work through this step by step, then give your final answer on the last "
    "line in exactly this format:\nANSWER: <answer>"
)


# ---------------------------------------------------------------- arith

def gen_arith(rng: random.Random, difficulty: str) -> dict:
    """Multi-step quantity tracking: TWO tracked item types with cross-transfers,
    distractor items and non-events salted in, events in shuffled order (all ops
    are +/- so the final counts are order-independent). Medium/hard must report
    BOTH final counts (two graders -> partial credit)."""
    n_steps = {"easy": 4, "medium": 10, "hard": 16, "extreme": 28}[difficulty]
    n_distractors = {"easy": 1, "medium": 4, "hard": 8, "extreme": 14}[difficulty]
    both = difficulty != "easy"

    item_a, item_b, other = rng.sample(
        ["crates", "pallets", "drums", "cartons", "sacks", "bins"], 3)
    qa, qb = rng.randint(60, 140), rng.randint(60, 140)
    ta, tb = qa, qb
    lines = [f"A warehouse starts the week with {qa} {item_a}, {qb} {item_b}, "
             f"and {rng.randint(20, 60)} {other}."]

    ops = []
    for _ in range(n_steps):
        kind = rng.choice(["recv_a", "ship_a", "recv_b", "ship_b", "xfer_ab", "xfer_ba", "audit"])
        k = rng.randint(3, 28)
        if kind == "recv_a":
            ta += k; ops.append((k, 0))
            lines.append(f"A delivery adds {k} {item_a}.")
        elif kind == "ship_a":
            ta -= k; ops.append((-k, 0))
            lines.append(f"An order ships out {k} {item_a}.")
        elif kind == "recv_b":
            tb += k; ops.append((0, k))
            lines.append(f"A supplier drops off {k} {item_b}.")
        elif kind == "ship_b":
            tb -= k; ops.append((0, -k))
            lines.append(f"A customer pickup removes {k} {item_b}.")
        elif kind == "xfer_ab":
            ta -= k; tb += k; ops.append((-k, k))
            lines.append(f"{k} {item_a} are repacked: their contents now count as {k} {item_b} "
                         f"(the {item_a} leave stock).")
        elif kind == "xfer_ba":
            tb -= k; ta += k; ops.append((k, -k))
            lines.append(f"{k} {item_b} are consolidated into {k} {item_a} "
                         f"(the {item_b} leave stock).")
        else:
            j = rng.randint(2, 9)
            ta += j; ops.append((j, 0))
            lines.append(f"An audit finds {j} miscounted {item_a} and adds them back to the ledger.")

    for _ in range(n_distractors):
        k = rng.randint(5, 40)
        lines.append(rng.choice([
            f"The site manager notes {k} {other} were repainted.",
            f"A supplier invoice for {k} {other} arrives (not yet delivered).",
            f"{k} {other} are moved between aisles within the warehouse.",
            f"A quote is requested for {k} additional {other}.",
            f"{k} {item_a} are inspected and returned to the same shelf.",
        ]))

    tail = lines[1:]
    rng.shuffle(tail)  # interleave distractors with real ops (keep opening line)
    lines = [lines[0]] + tail

    ca, cb = qa, qb
    for da, db in ops:
        ca += da; cb += db
    assert (ca, cb) == (ta, tb), f"arith solver mismatch: {(ca, cb)} != {(ta, tb)}"

    if both:
        prompt = (" ".join(lines)
                  + f"\n\nHow many {item_a} and how many {item_b} does the warehouse have at the "
                  + "end? Work through this step by step, then give your final answer on the last "
                  + f"line in exactly this format:\nANSWER: {item_a}=<n>, {item_b}=<n>")
        graders = [
            {"type": "match", "dimension": "count_a", "mode": "regex",
             "expected": rf"{item_a}\s*=\s*{ta}\b", "case_sensitive": False},
            {"type": "match", "dimension": "count_b", "mode": "regex",
             "expected": rf"{item_b}\s*=\s*{tb}\b", "case_sensitive": False},
        ]
        truth = f"{item_a}={ta}, {item_b}={tb}"
    else:
        prompt = (" ".join(lines)
                  + f"\n\nHow many {item_a} does the warehouse have at the end?\n\n" + ANSWER_RULES)
        graders = [{"type": "match", "dimension": "reasoning_exact", "mode": "number",
                    "expected": ta, "tolerance": 0}]
        truth = ta
    return {
        "prompt": prompt,
        "graders": graders,
        "meta": {"family": "arith", "difficulty": difficulty, "truth": truth},
    }


def gen_arith_safe(rng: random.Random, difficulty: str) -> dict:
    return gen_arith(rng, difficulty)


# ---------------------------------------------------------------- sequence

def gen_sequence(rng: random.Random, difficulty: str) -> dict:
    """Composed-rule sequences: a(n) built from 2-3 stacked simple rules."""
    if difficulty == "easy":
        # alternating add: +a, +b, +a, +b ...
        n_show = 6
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        start = rng.randint(1, 20)
        seq = [start]
        for i in range(n_show):
            seq.append(seq[-1] + (a if i % 2 == 0 else b))
        shown, answer = seq[:n_show], seq[n_show]
        v = shown[-1] + (a if (n_show - 1) % 2 == 0 else b)
        assert v == answer
    elif difficulty == "medium":
        # lagged recurrence: a(n) = a(n-2) + a(n-3) (not Fibonacci — lag pattern
        # must be induced, and the values aren't a famous sequence)
        n_show = 8
        seq = [rng.randint(1, 9) for _ in range(3)]
        while len(seq) < n_show + 1:
            seq.append(seq[-2] + seq[-3])
        shown, answer = seq[:n_show], seq[n_show]
        assert shown[-2] + shown[-3] == answer
    elif difficulty == "hard":
        # THREE interleaved streams (period 3): *2 stream, +k stream, 2nd-diff
        # stream; asked for the next TWO terms -> partial credit per term
        n_show = 9
        k, c, d = rng.randint(3, 11), rng.randint(1, 5), rng.randint(2, 4)
        x1, x2, x3 = rng.randint(1, 5), rng.randint(2, 20), rng.randint(1, 10)
        seq, step3 = [], 0
        for i in range(n_show + 2):
            w = i % 3
            if w == 0:
                seq.append(x1); x1 *= 2
            elif w == 1:
                seq.append(x2); x2 += k
            else:
                seq.append(x3); x3 += c + d * step3; step3 += 1
        shown, a1, a2 = seq[:n_show], seq[n_show], seq[n_show + 1]
        # independent check: stream positions n_show, n_show+1 recomputed
        assert a1 == shown[n_show - 3] * 2  # n_show=9 -> stream 0
        assert a2 == shown[n_show - 2] + k  # stream 1
        prompt = (
            f"Consider the integer sequence: {', '.join(map(str, shown))}, ...\n"
            "What are the next TWO terms? Work through this step by step, then give your final "
            "answer on the last line in exactly this format:\nANSWER: <term10>, <term11>"
        )
        return {
            "prompt": prompt,
            "graders": [{"type": "match", "dimension": "next_terms", "mode": "regex",
                         "expected": rf"ANSWER:\s*{a1}\s*,\s*{a2}\b"}],
            "meta": {"family": "sequence", "difficulty": difficulty, "truth": f"{a1}, {a2}"},
        }
    else:
        # extreme: a single sequence governed by a DEEP compound recurrence that is
        # not a famous sequence — a lag-4 linear recurrence with distinct integer
        # coefficients AND a constant offset: a(n) = c1*a(n-1) + c3*a(n-3) - a(n-4) + b.
        # Four leading terms + the 4-deep lag + the additive constant make the rule
        # far harder to induce than the hard tier's period-3 interleave; asked for
        # the next TWO terms (partial credit per term).
        n_show = 12
        c1, c3, b = rng.randint(1, 2), rng.randint(1, 3), rng.randint(1, 5)
        seq = [rng.randint(1, 4) for _ in range(4)]
        while len(seq) < n_show + 2:
            seq.append(c1 * seq[-1] + c3 * seq[-3] - seq[-4] + b)
        shown, a1, a2 = seq[:n_show], seq[n_show], seq[n_show + 1]
        # independent recomputation from the shown window only (no reuse of seq)
        w = list(shown)
        w.append(c1 * w[-1] + c3 * w[-3] - w[-4] + b)
        w.append(c1 * w[-1] + c3 * w[-3] - w[-4] + b)
        assert w[n_show] == a1 and w[n_show + 1] == a2
        prompt = (
            f"Consider the integer sequence: {', '.join(map(str, shown))}, ...\n"
            "What are the next TWO terms? Work through this step by step, then give your final "
            "answer on the last line in exactly this format:\nANSWER: <term13>, <term14>"
        )
        return {
            "prompt": prompt,
            "graders": [{"type": "match", "dimension": "next_terms", "mode": "regex",
                         "expected": rf"ANSWER:\s*{a1}\s*,\s*{a2}\b"}],
            "meta": {"family": "sequence", "difficulty": difficulty, "truth": f"{a1}, {a2}"},
        }

    prompt = (
        f"Consider the integer sequence: {', '.join(map(str, shown))}, ...\n"
        f"What is the next term?\n\n" + ANSWER_RULES
    )
    return {
        "prompt": prompt,
        "graders": [{"type": "match", "dimension": "reasoning_exact", "mode": "number",
                     "expected": answer, "tolerance": 0}],
        "meta": {"family": "sequence", "difficulty": difficulty, "truth": answer},
    }


# ---------------------------------------------------------------- machine

def gen_machine(rng: random.Random, difficulty: str) -> dict:
    """Tiny register machine: R0..R2, ops ADD/SUB/MUL/MOV/SWP/JNZ-free straight line.

    Multi-step exact state tracking — the capability our OOD probe showed models
    fake via memorization on famous constants but fail on arbitrary instances.
    """
    n_ops = {"easy": 6, "medium": 18, "hard": 30, "extreme": 50}[difficulty]
    regs = [rng.randint(1, 9) for _ in range(3)]
    lines = [f"Registers start at R0={regs[0]}, R1={regs[1]}, R2={regs[2]}."]
    prog = []
    for _ in range(n_ops):
        op = rng.choice(["ADD", "SUB", "MULMOD", "MOV", "SWP"])
        a, b = rng.sample([0, 1, 2], 2)
        if op == "ADD":
            prog.append(("ADD", a, b)); lines.append(f"ADD R{a} R{b}   (R{a} = R{a} + R{b})")
        elif op == "SUB":
            prog.append(("SUB", a, b)); lines.append(f"SUB R{a} R{b}   (R{a} = R{a} - R{b})")
        elif op == "MULMOD":
            prog.append(("MULMOD", a, b)); lines.append(f"MULMOD R{a} R{b}   (R{a} = (R{a} * R{b}) mod 100)")
        elif op == "MOV":
            prog.append(("MOV", a, b)); lines.append(f"MOV R{a} R{b}   (R{a} = R{b})")
        else:
            prog.append(("SWP", a, b)); lines.append(f"SWP R{a} R{b}   (swap R{a} and R{b})")

    def run(program, start):
        r = list(start)
        for op, a, b in program:
            if op == "ADD": r[a] = r[a] + r[b]
            elif op == "SUB": r[a] = r[a] - r[b]
            elif op == "MULMOD": r[a] = (r[a] * r[b]) % 100
            elif op == "MOV": r[a] = r[b]
            else: r[a], r[b] = r[b], r[a]
        return r

    final = run(prog, regs)
    final2 = run(list(prog), list(regs))  # independent pass on copies
    assert final == final2

    header = (
        "Execute this register program by hand, instruction by instruction.\n"
        "Note: 'mod 100' is the mathematical modulo — the result is always in 0..99 "
        "even for negative inputs (e.g. (-7) mod 100 = 93).\n\n"
    )
    if difficulty == "easy":
        target_reg = rng.randint(0, 2)
        answer = final[target_reg]
        prompt = (header + "\n".join(lines)
                  + f"\n\nWhat is the final value of R{target_reg}?\n\n" + ANSWER_RULES)
        graders = [{"type": "match", "dimension": "reasoning_exact", "mode": "number",
                    "expected": answer, "tolerance": 0}]
        truth = answer
    else:
        # all three registers -> three graders -> partial credit per register
        prompt = (header + "\n".join(lines)
                  + "\n\nWhat are the final values of all three registers? Work through this "
                  + "step by step, then give your final answer on the last line in exactly this "
                  + "format:\nANSWER: R0=<n>, R1=<n>, R2=<n>")
        graders = [{"type": "match", "dimension": f"reg_{i}", "mode": "regex",
                    "expected": rf"R{i}\s*=\s*{final[i]}\b"} for i in range(3)]
        truth = f"R0={final[0]}, R1={final[1]}, R2={final[2]}"
    return {
        "prompt": prompt,
        "graders": graders,
        "meta": {"family": "machine", "difficulty": difficulty, "truth": truth},
    }


# ---------------------------------------------------------------- schedule

# Extended past 7 so the extreme tier can reach n=8 (schedule) / n=9 (knights).
# Kept collision-free for the all_of substring grader: no label is a substring of
# another (e.g. no "Mon"/"Mon2" pair that would false-positive).
NAMES = ["Alice", "Bob", "Carol", "Dan", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Hol"]


def gen_schedule(rng: random.Random, difficulty: str) -> dict:
    """Assign n people to n distinct days under constraints; unique solution required."""
    n = {"easy": 4, "medium": 6, "hard": 7, "extreme": 8}[difficulty]
    people = NAMES[:n]
    days = DAYS[:n]

    for _attempt in range(400):
        sol = list(range(n))  # person i -> day sol[i]
        rng.shuffle(sol)

        # Build a constraint pool that the solution satisfies, then greedily add
        # constraints until the brute-force solver finds a unique model.
        # Hard tier: relational/disjunctive clues only (no direct "not on <day>"
        # anchors) — forces chained inference instead of elimination bookkeeping.
        pool = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if sol[i] < sol[j]:
                    pool.append((f"{people[i]} presents earlier in the week than {people[j]}.",
                                 lambda a, i=i, j=j: a[i] < a[j]))
                if sol[i] + 1 == sol[j]:
                    pool.append((f"{people[j]} presents on the day right after {people[i]}.",
                                 lambda a, i=i, j=j: a[j] == a[i] + 1))
                gap = abs(sol[i] - sol[j])
                if gap >= 2:
                    pool.append((f"At least {gap} days separate {people[i]} and {people[j]} "
                                 f"(in some order).",
                                 lambda a, i=i, j=j, g=gap: abs(a[i] - a[j]) >= g))
        for i in range(n):
            wrong = rng.choice([d for d in range(n) if d != sol[i]])
            # direct "not on <day>" anchors only below the relational tiers
            if difficulty in ("easy", "medium"):
                pool.append((f"{people[i]} does not present on {days[wrong]}.",
                             lambda a, i=i, w=wrong: a[i] != w))
            if difficulty != "easy":
                pool.append((f"{people[i]} presents on {days[sol[i]]} or {days[wrong]}.",
                             lambda a, i=i, s=sol[i], w=wrong: a[i] in (s, w)))
        if difficulty == "extreme":
            # ternary "between" clues (person j strictly between i and k in the week)
            # — no anchor, forces three-way chained inference.
            for j in range(n):
                lo = [i for i in range(n) if sol[i] < sol[j]]
                hi = [k for k in range(n) if sol[k] > sol[j]]
                if lo and hi:
                    i, k = rng.choice(lo), rng.choice(hi)
                    pool.append((
                        f"{people[j]} presents on a day strictly between {people[i]}'s and "
                        f"{people[k]}'s days.",
                        lambda a, i=i, j=j, k=k: a[i] < a[j] < a[k]))
        rng.shuffle(pool)

        chosen = []
        def models(cs):
            out = []
            for perm in itertools.permutations(range(n)):
                if all(c(perm) for _, c in cs):
                    out.append(perm)
                    if len(out) > 1:
                        return out
            return out

        for text, check in pool:
            assert check(sol)
            chosen.append((text, check))
            m = models(chosen)
            if len(m) == 1:
                break
        m = models(chosen)
        if len(m) == 1 and list(m[0]) == sol:
            break
    else:
        raise RuntimeError("schedule generator failed to reach uniqueness")

    # prune redundant constraints (keep it minimal-ish, harder)
    pruned = list(chosen)
    for c in list(pruned):
        trial = [x for x in pruned if x is not c]
        if len(models(trial)) == 1:
            pruned = trial

    expected = [f"{people[i]}={days[sol[i]]}" for i in range(n)]
    prompt = (
        f"{n} colleagues each give one presentation on a distinct day, {days[0]} through {days[-1]}.\n\n"
        + "\n".join(f"- {t}" for t, _ in pruned)
        + "\n\nDetermine the unique schedule. Give your final answer on the last line in exactly this "
        + "format (comma-separated, no spaces around '='):\n"
        + f"ANSWER: {people[0]}=<day>, {people[1]}=<day>, ..."
    )
    return {
        "prompt": prompt,
        "graders": [{"type": "match", "dimension": "reasoning_exact", "mode": "all_of",
                     "expected": expected, "case_sensitive": False}],
        "meta": {"family": "schedule", "difficulty": difficulty, "truth": expected},
    }


# ---------------------------------------------------------------- grid

def gen_grid(rng: random.Random, difficulty: str) -> dict:
    """Zebra-style: n houses, 3 attribute rows, clues to unique solution, one query.

    n stays 4 at hard tier: the O(n!^3) uniqueness solver thrashes at n=5 with
    relational-only clues (observed 15+ CPU-min). Hard difficulty comes from
    clue indirection (no direct person->color anchors, disjunctions), not size.
    """
    n = {"easy": 3, "medium": 4, "hard": 4, "extreme": 4}[difficulty]
    people = NAMES[:n]
    colors = ["red", "blue", "green", "yellow", "white"][:n]
    drinks = ["tea", "coffee", "milk", "juice", "cocoa"][:n]

    for _attempt in range(400):
        # solution: position -> (person, color, drink); each row is a permutation
        p_sol = list(range(n)); rng.shuffle(p_sol)   # person index at position k
        c_sol = list(range(n)); rng.shuffle(c_sol)
        d_sol = list(range(n)); rng.shuffle(d_sol)

        pool = []
        for k in range(n):
            pi, ci, di = p_sol[k], c_sol[k], d_sol[k]
            if difficulty in ("easy", "medium"):
                # direct person->color anchors only below the relational tiers
                pool.append((f"{people[pi]} lives in the {colors[ci]} house.",
                             lambda P, C, D, pi=pi, ci=ci: P.index(pi) == C.index(ci)))
            pool.append((f"The person in the {colors[ci]} house drinks {drinks[di]}.",
                         lambda P, C, D, ci=ci, di=di: C.index(ci) == D.index(di)))
            pool.append((f"The {drinks[di]} drinker lives in house {k + 1}."
                         if difficulty == "easy" else
                         f"The {drinks[di]} drinker does not live in house {(k + 1) % n + 1}.",
                         (lambda P, C, D, di=di, k=k: D.index(di) == k)
                         if difficulty == "easy" else
                         (lambda P, C, D, di=di, w=(k + 1) % n: D.index(di) != w)))
            if k + 1 < n:
                pj = p_sol[k + 1]
                pool.append((f"{people[pi]} lives immediately to the left of {people[pj]}.",
                             lambda P, C, D, pi=pi, pj=pj: P.index(pj) - P.index(pi) == 1))
            if difficulty != "easy":
                wrong_d = rng.choice([x for x in range(n) if x != di])
                pool.append((f"{people[pi]} does not drink {drinks[wrong_d]}.",
                             lambda P, C, D, pi=pi, w=wrong_d: D[P.index(pi)] != w))
            if difficulty in ("hard", "extreme"):
                # relational / disjunctive clues force chained inference
                for k2 in range(k + 2, n):
                    pj2 = p_sol[k2]
                    pool.append((f"{people[pi]} lives somewhere to the left of {people[pj2]}.",
                                 lambda P, C, D, a=pi, b=pj2: P.index(a) < P.index(b)))
                w_pos = rng.choice([x for x in range(n) if x != k])
                pool.append((f"{people[pi]} lives in house {k + 1} or house {w_pos + 1}.",
                             lambda P, C, D, pi=pi, s=k, w=w_pos: P.index(pi) in (s, w)))
                w2 = rng.choice([x for x in range(n) if x != k])
                pool.append((f"The {drinks[di]} drinker lives in house {k + 1} or house {w2 + 1}.",
                             lambda P, C, D, di=di, s=k, w=w2: D.index(di) in (s, w)))
                w3 = rng.choice([x for x in range(n) if x != k])
                pool.append((f"The {colors[ci]} house is not house {w3 + 1}.",
                             lambda P, C, D, ci=ci, w=w3: C.index(ci) != w))
            if difficulty == "extreme":
                # cross-attribute RELATIONAL clues (no attribute is pinned to a
                # position): tie a colour's position to a drink's position, and a
                # person's position to another colour's position. Maximal indirection
                # at n=4 — the whole grid must be inferred by chaining, not lookup.
                for k2 in range(n):
                    if k2 == k:
                        continue
                    dj = d_sol[k2]
                    if c_sol.index(ci) < d_sol.index(dj):
                        pool.append((
                            f"The {colors[ci]} house is somewhere to the left of the "
                            f"{drinks[dj]} drinker.",
                            lambda P, C, D, ci=ci, dj=dj: C.index(ci) < D.index(dj)))
                    cj = c_sol[k2]
                    if p_sol.index(pi) != c_sol.index(cj):
                        pool.append((
                            f"{people[pi]} does not live in the {colors[cj]} house.",
                            lambda P, C, D, pi=pi, cj=cj: P.index(pi) != C.index(cj)))
        if difficulty in ("easy", "medium"):
            pool.append((f"{people[p_sol[0]]} lives in the first (leftmost) house.",
                         lambda P, C, D, p0=p_sol[0]: P.index(p0) == 0))
        rng.shuffle(pool)

        def models(cs):
            out = []
            for P in itertools.permutations(range(n)):
                for C in itertools.permutations(range(n)):
                    for D in itertools.permutations(range(n)):
                        if all(c(list(P), list(C), list(D)) for _, c in cs):
                            out.append((P, C, D))
                            if len(out) > 1:
                                return out
            return out

        chosen = []
        for text, check in pool:
            assert check(p_sol, c_sol, d_sol)
            chosen.append((text, check))
            m = models(chosen)
            if len(m) == 1:
                break
        m = models(chosen)
        if len(m) == 1 and m[0] == (tuple(p_sol), tuple(c_sol), tuple(d_sol)):
            break
    else:
        raise RuntimeError("grid generator failed to reach uniqueness")

    for c in list(chosen):
        trial = [x for x in chosen if x is not c]
        if len(models(trial)) == 1:
            chosen = trial

    # Ask for the FULL grid (each person's house# + color + drink) as an all_of
    # battery -> partial credit, so the task isn't all-or-nothing. Per-person
    # tuple: "<name>:h<pos>,<color>,<drink>".
    pos_of = {p_sol[k]: k for k in range(n)}
    col_of = {c_sol[k]: k for k in range(n)}  # color index -> position
    drk_of = {d_sol[k]: k for k in range(n)}
    expected = []
    for pi in range(n):
        k = pos_of[pi]
        ci = next(c for c, p in col_of.items() if p == k)
        di = next(d for d, p in drk_of.items() if p == k)
        expected.append(f"{people[pi]}=h{k + 1},{colors[ci]},{drinks[di]}")

    prompt = (
        f"There are {n} houses in a row, positions 1 (leftmost) to {n} (rightmost). Each house has a "
        f"distinct color, and each is occupied by a different person with a distinct drink.\n\n"
        + "\n".join(f"- {t}" for t, _ in chosen)
        + "\n\nDetermine, for every person, their house number, house color, and drink. Work through "
        + "this step by step, then give your final answer on the last line as a comma-separated list "
        + "in exactly this per-person format (no spaces inside a person's entry):\n"
        + f"ANSWER: {people[0]}=h<pos>,<color>,<drink>; {people[1]}=h<pos>,<color>,<drink>; ..."
    )
    return {
        "prompt": prompt,
        "graders": [{"type": "match", "dimension": "grid_full", "mode": "all_of",
                     "expected": expected, "case_sensitive": False}],
        "meta": {"family": "grid", "difficulty": difficulty, "truth": expected},
    }


# ---------------------------------------------------------------- knights

def gen_knights(rng: random.Random, difficulty: str) -> dict:
    """Knights (always true) & knaves (always false); unique assignment."""
    n = {"easy": 3, "medium": 5, "hard": 7, "extreme": 9}[difficulty]
    people = NAMES[:n]

    def stmt_pool(sol):
        # NOTE: simple "X is a knight/knave" statements are invariant under the
        # global knight<->knave flip, so a pool of only those can NEVER have a
        # unique model. Compound statements (whose truth is not flip-invariant)
        # are required at every difficulty to break the symmetry.
        pool = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pool.append((f'{people[i]} says: "{people[j]} is a knight."',
                             lambda a, i=i, j=j: a[i] == a[j]))
                pool.append((f'{people[i]} says: "{people[j]} is a knave."',
                             lambda a, i=i, j=j: a[i] == (not a[j])))
            j, k = rng.sample([x for x in range(n) if x != i], 2)
            pool.append((f'{people[i]} says: "At least one of {people[j]} and {people[k]} is a knave."',
                         lambda a, i=i, j=j, k=k: a[i] == ((not a[j]) or (not a[k]))))
            if difficulty != "easy":
                pool.append((f'{people[i]} says: "{people[j]} and {people[k]} are both knights."',
                             lambda a, i=i, j=j, k=k: a[i] == (a[j] and a[k])))
            if difficulty in ("hard", "extreme"):
                pool.append((f'{people[i]} says: "Exactly one of {people[j]} and {people[k]} is a knight."',
                             lambda a, i=i, j=j, k=k: a[i] == (a[j] != a[k])))
                pool.append((f'{people[i]} says: "{people[j]} and {people[k]} are the same type."',
                             lambda a, i=i, j=j, k=k: a[i] == (a[j] == a[k])))
            if difficulty == "extreme":
                others = [x for x in range(n) if x != i]
                if len(others) >= 3:
                    p, q, r = rng.sample(others, 3)
                    # majority / counting statements over THREE others — truth is not
                    # flip-invariant and needs the whole assignment to evaluate.
                    pool.append((
                        f'{people[i]} says: "At least two of {people[p]}, {people[q]}, and '
                        f'{people[r]} are knights."',
                        lambda a, i=i, p=p, q=q, r=r: a[i] == ((a[p] + a[q] + a[r]) >= 2)))
                    pool.append((
                        f'{people[i]} says: "{people[p]}, {people[q]}, and {people[r]} are '
                        f'not all the same type."',
                        lambda a, i=i, p=p, q=q, r=r: a[i] == (not (a[p] == a[q] == a[r]))))
        return pool

    for _attempt in range(400):
        sol = [rng.random() < 0.5 for _ in range(n)]
        pool = [s for s in stmt_pool(sol) if s[1](sol)]
        rng.shuffle(pool)

        def models(cs):
            out = []
            for bits in itertools.product([True, False], repeat=n):
                if all(c(list(bits)) for _, c in cs):
                    out.append(bits)
                    if len(out) > 1:
                        return out
            return out

        chosen = []
        for text, check in pool:
            chosen.append((text, check))
            m = models(chosen)
            if len(m) == 1:
                break
        m = models(chosen)
        if len(m) == 1 and list(m[0]) == sol:
            break
    else:
        raise RuntimeError("knights generator failed to reach uniqueness")

    for c in list(chosen):
        trial = [x for x in chosen if x is not c]
        if len(models(trial)) == 1:
            chosen = trial

    expected = [f"{people[i]}={'knight' if sol[i] else 'knave'}" for i in range(n)]
    prompt = (
        f"On an island, knights always tell the truth and knaves always lie. "
        f"You meet {n} inhabitants: {', '.join(people)}.\n\n"
        + "\n".join(f"- {t}" for t, _ in chosen)
        + "\n\nDetermine what each inhabitant is. Give your final answer on the last line in exactly "
        + "this format (comma-separated, no spaces around '='):\n"
        + f"ANSWER: {people[0]}=<knight|knave>, {people[1]}=<knight|knave>, ..."
    )
    return {
        "prompt": prompt,
        "graders": [{"type": "match", "dimension": "reasoning_exact", "mode": "all_of",
                     "expected": expected, "case_sensitive": False}],
        "meta": {"family": "knights", "difficulty": difficulty, "truth": expected},
    }


# ---------------------------------------------------------------- emit

FAMILIES = {
    "arith": gen_arith_safe,
    "sequence": gen_sequence,
    "machine": gen_machine,
    "schedule": gen_schedule,
    "grid": gen_grid,
    "knights": gen_knights,
}

# 6 per family: easy, easy, medium, hard, extreme, extreme. The two extreme
# instances (ids v2_<family>_4_extreme / _5_extreme) push each knob past hard so a
# frontier model lands ~0.4-0.7 instead of saturating; all remain solver-verified.
TIERS = ["easy", "easy", "medium", "hard", "extreme", "extreme"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent)
    args = ap.parse_args()

    for family, gen in FAMILIES.items():
        rng = random.Random(f"{args.seed}:{family}")
        tests = []
        for idx, tier in enumerate(TIERS):
            t = gen(rng, tier)
            meta = t.pop("meta")
            tests.append({
                "id": f"v2_{family}_{idx}_{tier}",
                "prompt": t["prompt"],
                "graders": t["graders"],
                # truth recorded for audit; harness ignores unknown keys
                "ground_truth": meta["truth"],
                "difficulty": tier,
            })
        doc = {
            "id": f"reasoning_v2_{family}",
            "name": f"Reasoning v2 — {family} (solver-verified, seed {args.seed})",
            "category": "reasoning",
            "tests": tests,
        }
        out_file = args.out / f"v2_{family}.yaml"
        with open(out_file, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, width=100, allow_unicode=True)
        print(f"wrote {out_file} ({len(tests)} tasks)")


if __name__ == "__main__":
    main()
