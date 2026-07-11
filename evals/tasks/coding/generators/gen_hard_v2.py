"""code-exec v2 generator — novel spec-delta problems, cross-verified references.

Phase 3 (FOCUS.md): Ornith-35B aced 6/6 canonical LeetCode-Hard in hard.yaml —
canonical problems measure memorization. Every problem here is a SPEC-DELTA of
a well-known problem: the memorized solution compiles, runs, and FAILS the
battery. Ground truth comes from a reference implementation cross-checked at
generation time against a second, independently-structured implementation over
hundreds of random instances; emitted tests are a seeded sample.

Regenerate: python gen_hard_v2.py --seed 20260702
Emits ../hard_v2.yaml. Bump the seed after publishing baselines on a set.
"""

from __future__ import annotations

import argparse
import functools
import heapq
import itertools
import random
import re
from pathlib import Path

import yaml


# ================================================================ calc2
# Delta vs LC-224/227 "Basic Calculator": adds right-associative `^` that binds
# TIGHTER than unary minus (so -2^2 = -4), while / still truncates toward zero.

class N:  # AST node
    def __init__(self, op, a=None, b=None, v=None):
        self.op, self.a, self.b, self.v = op, a, b, v

    def ev(self):
        if self.op == "num":
            return self.v
        if self.op == "neg":
            return -self.a.ev()
        x, y = self.a.ev(), self.b.ev()
        if self.op == "+":
            return x + y
        if self.op == "-":
            return x - y
        if self.op == "*":
            return x * y
        if self.op == "/":
            return int(x / y)  # truncate toward zero
        if self.op == "^":
            return x ** y


_PREC = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3, "^": 4, "num": 5}


def _pr(n: N, parent_prec: int = 0, right_side: bool = False) -> str:
    p = _PREC[n.op]
    if n.op == "num":
        s = str(n.v)
    elif n.op == "neg":
        s = "-" + _pr(n.a, p)
    elif n.op == "^":
        # right-assoc: left child needs parens at equal prec
        s = _pr(n.a, p + 1) + " ^ " + _pr(n.b, p)
    else:
        # left-assoc: right child needs parens at equal prec
        s = _pr(n.a, p) + f" {n.op} " + _pr(n.b, p + 1, True)
    if p < parent_prec:
        s = "(" + s + ")"
    return s


def _gen_ast(rng: random.Random, depth: int) -> N:
    if depth == 0 or rng.random() < 0.25:
        return N("num", v=rng.randint(0, 30))
    op = rng.choice(["+", "-", "*", "/", "^", "neg", "+", "-", "*"])
    if op == "neg":
        return N("neg", _gen_ast(rng, depth - 1))
    if op == "^":
        base = _gen_ast(rng, depth - 1)
        return N("^", base, N("num", v=rng.randint(0, 3)))
    a, b = _gen_ast(rng, depth - 1), _gen_ast(rng, depth - 1)
    if op == "/":
        for _ in range(30):
            if b.ev() != 0:
                break
            b = _gen_ast(rng, depth - 1)
        else:
            return _gen_ast(rng, depth)
    return N(op, a, b)


def _calc_parse(s: str) -> int:
    """Independent recursive-descent cross-check parser."""
    toks = re.findall(r"\d+|[-+*/^()]", s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def eat():
        t = toks[pos[0]]; pos[0] += 1; return t

    def expr():
        v = term()
        while peek() in ("+", "-"):
            if eat() == "+":
                v = v + term()
            else:
                v = v - term()
        return v

    def term():
        v = unary()
        while peek() in ("*", "/"):
            if eat() == "*":
                v = v * unary()
            else:
                v = int(v / unary())
        return v

    def unary():
        if peek() == "-":
            eat()
            return -unary()
        return power()

    def power():
        v = atom()
        if peek() == "^":
            eat()
            v = v ** unary()
        return v

    def atom():
        if peek() == "(":
            eat(); v = expr(); assert eat() == ")"; return v
        return int(eat())

    v = expr()
    assert pos[0] == len(toks)
    return v


def build_calc2(rng: random.Random) -> dict:
    exprs = []
    while len(exprs) < 10:
        ast = _gen_ast(rng, rng.randint(3, 5))
        try:
            val = ast.ev()
        except OverflowError:
            continue
        if abs(val) > 10**7:
            continue
        s = _pr(ast)
        assert _calc_parse(s) == val, f"calc2 cross-check failed: {s}"
        if "^" in s or "-" in s:
            exprs.append((s, val))
    hand = [
        ("-2 ^ 2", -4),          # the delta: ^ binds tighter than unary minus
        ("2 ^ 3 ^ 2", 512),      # right-assoc
        ("(-2) ^ 2", 4),
        ("7 / -2", -3),          # truncation toward zero
        ("-7 / 2", -3),
        ("2 ^ 0 - 5", -4),
    ]
    for s, v in hand:
        assert _calc_parse(s) == v
    tests = [f"assert evaluate2({s!r}) == {v}" for s, v in hand + exprs]
    prompt = (
        "Implement `def evaluate2(s: str) -> int` for arithmetic expressions with integers, "
        "`+ - * / ^`, parentheses, unary minus, and arbitrary spaces.\n"
        "Semantics (read carefully — they differ from the usual conventions):\n"
        "- `^` is exponentiation, RIGHT-associative: `2 ^ 3 ^ 2` = 2^(3^2) = 512.\n"
        "- `^` binds TIGHTER than unary minus: `-2 ^ 2` = -(2^2) = -4, but `(-2) ^ 2` = 4. "
        "Exponents are always non-negative integers.\n"
        "- `/` is integer division truncating TOWARD ZERO: `7 / -2` = -3, `-7 / 2` = -3.\n"
        "- `*` and `/` bind tighter than `+`/`-`; unary minus binds tighter than `*`/`/`.\n"
        "The expression is valid; no division by zero. Return only the function in a Python code block."
    )
    return {"id": "v2_calc_pow", "difficulty": "hard", "entry": "evaluate2",
            "prompt": prompt, "tests": tests}


# ================================================================ wordwild
# Delta vs LC-44 wildcard matching: `*` and `?` cannot cross/match spaces.

def _ww_ref(s: str, p: str) -> bool:
    @functools.lru_cache(maxsize=None)
    def go(i, j):
        if j == len(p):
            return i == len(s)
        c = p[j]
        if c == "*":
            if go(i, j + 1):
                return True
            return i < len(s) and s[i] != " " and go(i + 1, j)
        if i == len(s):
            return False
        if c == "?":
            return s[i] != " " and go(i + 1, j + 1)
        return s[i] == c and go(i + 1, j + 1)
    return go(0, 0)


def _ww_regex(s: str, p: str) -> bool:
    rx = "".join("[^ ]*" if c == "*" else "[^ ]" if c == "?" else re.escape(c) for c in p)
    return re.fullmatch(rx, s) is not None


def build_wordwild(rng: random.Random) -> dict:
    alpha = "abc "
    cases = []
    seen = set()
    while len(cases) < 10:
        s = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 10))).strip()
        p = "".join(rng.choice("abc?* ") for _ in range(rng.randint(0, 7))).strip()
        if (s, p) in seen:
            continue
        seen.add((s, p))
        r1, r2 = _ww_ref(s, p), _ww_regex(s, p)
        assert r1 == r2, f"wordwild cross-check failed: {(s, p)}"
        cases.append((s, p, r1))
    # ensure balance: at least 3 True and 3 False among random cases
    if sum(1 for *_, r in cases if r) < 3 or sum(1 for *_, r in cases if not r) < 3:
        return build_wordwild(rng)  # rare; reroll
    hand = [
        ("ab cd", "ab*cd", False),   # the delta: * cannot cross the space
        ("ab cd", "ab cd", True),
        ("ab cd", "a* c*", True),
        ("ab cd", "?? ??", True),
        ("a b", "a?b", False),       # ? cannot match a space
        ("abcd", "*", True),
        ("a b", "*", False),         # * cannot cover a space
        ("", "*", True),
    ]
    for s, p, r in hand:
        assert _ww_ref(s, p) == r == _ww_regex(s, p)
    tests = [f"assert wc_match({s!r}, {p!r}) == {r}" for s, p, r in hand + cases]
    prompt = (
        "Implement `def wc_match(s: str, p: str) -> bool` — wildcard matching over the ENTIRE "
        "string, where strings may contain spaces and the wildcards are word-local:\n"
        "- `?` matches exactly one character that is NOT a space.\n"
        "- `*` matches any sequence (possibly empty) of NON-SPACE characters — it can never "
        "match across a space.\n"
        "- Every other character (including a space) matches only itself.\n"
        "So `wc_match('ab cd', 'ab*cd')` is False (the `*` would have to cover the space) but "
        "`wc_match('ab cd', 'a* c*')` is True. Return only the function in a Python code block."
    )
    return {"id": "v2_wildcard_words", "difficulty": "hard", "entry": "wc_match",
            "prompt": prompt, "tests": tests}


# ================================================================ expand
# Delta vs LC-394 "Decode String": multiplier comes AFTER the group, plus a
# postfix reverse operator `~`, both applying to the preceding bracket group.

def _expand_ref(s: str) -> str:
    def parse(i):
        out = []
        while i < len(s) and s[i] != "]":
            if s[i] == "[":
                inner, i = parse(i + 1)
                assert s[i] == "]"
                i += 1
                if i < len(s) and s[i] == "~":
                    inner = inner[::-1]
                    i += 1
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                if j > i:
                    inner = inner * int(s[i:j])
                    i = j
                out.append(inner)
            else:
                out.append(s[i])
                i += 1
        return "".join(out), i

    r, i = parse(0)
    assert i == len(s)
    return r


def _expand_stack(s: str) -> str:
    """Independent stack-based cross-check."""
    stack = [[""]]
    i = 0
    while i < len(s):
        c = s[i]
        if c == "[":
            stack.append([""])
            i += 1
        elif c == "]":
            body = "".join(stack.pop())
            i += 1
            if i < len(s) and s[i] == "~":
                body = body[::-1]
                i += 1
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            if j > i:
                body *= int(s[i:j])
                i = j
            stack[-1].append(body)
        else:
            stack[-1].append(c)
            i += 1
    assert len(stack) == 1
    return "".join(stack[-1])


def _gen_expand(rng: random.Random, depth: int) -> str:
    parts = []
    for _ in range(rng.randint(1, 3)):
        if depth > 0 and rng.random() < 0.55:
            inner = _gen_expand(rng, depth - 1)
            grp = "[" + inner + "]"
            if rng.random() < 0.5:
                grp += "~"
            if rng.random() < 0.8:
                grp += str(rng.randint(2, 3))
            parts.append(grp)
        else:
            parts.append("".join(rng.choice("abcd") for _ in range(rng.randint(1, 4))))
    return "".join(parts)


def build_expand(rng: random.Random) -> dict:
    cases = []
    while len(cases) < 9:
        src = _gen_expand(rng, rng.randint(1, 3))
        out = _expand_ref(src)
        assert out == _expand_stack(src), f"expand cross-check failed: {src}"
        if len(out) <= 120 and ("[" in src):
            cases.append((src, out))
    hand = [
        ("[ab]3", "ababab"),
        ("[ab]~", "ba"),
        ("[ab]~2", "baba"),          # reverse THEN repeat
        ("x[cd[e]2]3y", "xcdeecdeecdeey"),
        ("[abc]", "abc"),
        ("abc", "abc"),
        ("[[ab]~2]~", "abab"),       # reverse of "baba" reversed pieces
    ]
    for src, out in hand:
        assert _expand_ref(src) == out == _expand_stack(src), src
    tests = [f"assert expand({s!r}) == {o!r}" for s, o in hand + cases]
    prompt = (
        "Implement `def expand(s: str) -> str`. The input contains lowercase letters and bracket "
        "groups `[...]` which may nest. AFTER a closing bracket there may be, in this order:\n"
        "- an optional `~`, which REVERSES the group's expanded text;\n"
        "- an optional integer k, which repeats the (possibly reversed) text k times.\n"
        "A group with neither suffix expands to its content once. Examples: `[ab]3` -> `ababab`; "
        "`[ab]~2` -> `baba` (reverse first, then repeat); `x[cd[e]2]3y` -> `xcdeecdeecdeey`.\n"
        "Return only the function in a Python code block."
    )
    return {"id": "v2_expand_postfix", "difficulty": "medium", "entry": "expand",
            "prompt": prompt, "tests": tests}


# ================================================================ paint
# Later segments overpaint earlier ones; report visible runs (merge adjacent).

def _paint_ref(segs) -> list:
    events = {}
    for idx, (l, r, c) in enumerate(segs):
        for x in range(l, r):
            events[x] = c
    if not events:
        return []
    out = []
    for x in sorted(events):
        c = events[x]
        if out and out[-1][1] == x and out[-1][2] == c:
            out[-1][1] = x + 1
        else:
            out.append([x, x + 1, c])
    return [tuple(t) for t in out]


def _paint_sweep(segs) -> list:
    """Independent impl: reverse-order interval subtraction."""
    placed = []  # disjoint (l, r, c), from later segments first
    for l, r, c in reversed(segs):
        pieces = [(l, r)]
        for pl, pr, _ in placed:
            nxt = []
            for al, ar in pieces:
                if ar <= pl or pr <= al:
                    nxt.append((al, ar))
                else:
                    if al < pl:
                        nxt.append((al, pl))
                    if pr < ar:
                        nxt.append((pr, ar))
            pieces = nxt
        placed.extend((al, ar, c) for al, ar in pieces if al < ar)
    placed.sort()
    out = []
    for l, r, c in placed:
        if out and out[-1][1] == l and out[-1][2] == c:
            out[-1] = (out[-1][0], r, c)
        else:
            out.append((l, r, c))
    return out


def build_paint(rng: random.Random) -> dict:
    cases = []
    while len(cases) < 9:
        n = rng.randint(1, 6)
        segs = []
        for _ in range(n):
            l = rng.randint(0, 30)
            r = l + rng.randint(1, 12)
            segs.append((l, r, rng.choice("RGBY")))
        r1, r2 = _paint_ref(segs), _paint_sweep(segs)
        assert r1 == r2, f"paint cross-check failed: {segs}"
        cases.append((segs, r1))
    hand = [
        ([(0, 10, "R"), (2, 5, "G")], [(0, 2, "R"), (2, 5, "G"), (5, 10, "R")]),
        ([(0, 5, "R"), (5, 10, "R")], [(0, 10, "R")]),
        ([(0, 8, "R"), (0, 8, "G")], [(0, 8, "G")]),
        ([(3, 6, "B")], [(3, 6, "B")]),
        ([(0, 4, "R"), (2, 6, "G"), (4, 8, "R")], [(0, 2, "R"), (2, 4, "G"), (4, 8, "R")]),
    ]
    for segs, want in hand:
        assert _paint_ref(segs) == want == _paint_sweep(segs), segs
    tests = [f"assert paint({list(s)!r}) == {w!r}" for s, w in hand + cases]
    prompt = (
        "Implement `def paint(segments: list) -> list`. Each segment is `(l, r, color)` painting "
        "the half-open integer interval [l, r) — segments are applied IN ORDER, later paint "
        "covers earlier paint. Return the final visible coloring as a sorted list of maximal "
        "`(l, r, color)` tuples: no overlaps, no empty intervals, and adjacent tuples with the "
        "same color merged (`[(0,5,'R'),(5,10,'R')]` -> `[(0,10,'R')]`). Unpainted gaps are "
        "simply absent. Return only the function in a Python code block."
    )
    return {"id": "v2_paint_layers", "difficulty": "medium", "entry": "paint",
            "prompt": prompt, "tests": tests}


# ================================================================ topolex

def _topo_ref(n: int, edges) -> list:
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    h = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(h)
    out = []
    while h:
        u = heapq.heappop(h)
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(h, v)
    return out if len(out) == n else []


def _topo_brute(n: int, edges) -> list:
    best = None
    es = set(edges)
    for perm in itertools.permutations(range(n)):
        pos = {v: i for i, v in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in es):
            cand = list(perm)
            if best is None or cand < best:
                best = cand
    return best or []


def build_topolex(rng: random.Random) -> dict:
    cases = []
    while len(cases) < 9:
        n = rng.randint(3, 7)
        perm = list(range(n))
        rng.shuffle(perm)
        edges = set()
        for _ in range(rng.randint(1, n * 2)):
            i, j = sorted(rng.sample(range(n), 2))
            edges.add((perm[i], perm[j]))  # respects perm order -> DAG
        if rng.random() < 0.3:  # inject a cycle in some cases
            cyc = rng.sample(range(n), min(3, n))
            edges |= {(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))}
        edges = sorted(edges)
        want = _topo_ref(n, edges)
        assert want == _topo_brute(n, edges), f"topolex cross-check failed: {n} {edges}"
        cases.append((n, edges, want))
    if not any(w == [] for _, _, w in cases):
        cases[-1] = (3, [(0, 1), (1, 2), (2, 0)], [])
    hand = [
        (3, [(2, 1), (1, 0)], [2, 1, 0]),
        (4, [], [0, 1, 2, 3]),
        (2, [(0, 1), (1, 0)], []),
        (5, [(4, 0), (4, 1), (0, 2)], [3, 4, 0, 1, 2]),
    ]
    for n, e, w in hand:
        assert _topo_ref(n, e) == w == _topo_brute(n, e)
    tests = [f"assert topo_lex({n}, {list(e)!r}) == {w!r}" for n, e, w in hand + cases]
    prompt = (
        "Implement `def topo_lex(n: int, edges: list) -> list`. Nodes are 0..n-1; each edge "
        "`(a, b)` means a must come before b. Return the LEXICOGRAPHICALLY SMALLEST valid "
        "topological order as a list of ints. If the graph has a cycle, return `[]`.\n"
        "Lexicographic comparison is on the sequence itself: prefer the smallest possible first "
        "element, then the smallest second element given the first, and so on.\n"
        "Return only the function in a Python code block."
    )
    return {"id": "v2_topo_lex", "difficulty": "medium", "entry": "topo_lex",
            "prompt": prompt, "tests": tests}


# ================================================================ multikey

def _mk_cmp(spec):
    def cmp(a, b):
        for field, direction, nulls in spec:
            x, y = a.get(field), b.get(field)
            if x is None and y is None:
                continue
            if x is None:
                return -1 if nulls == "nulls_first" else 1
            if y is None:
                return 1 if nulls == "nulls_first" else -1
            if x == y:
                continue
            lt = x < y
            if direction == "desc":
                lt = not lt
            return -1 if lt else 1
        return 0
    return cmp


def _mk_ref(records, spec):
    return sorted(records, key=functools.cmp_to_key(_mk_cmp(spec)))


def _mk_passes(records, spec):
    """Independent impl: successive stable sorts, last key first."""
    out = list(records)
    for field, direction, nulls in reversed(spec):
        desc = direction == "desc"
        # stable partition into null/non-null buckets, sort the non-null bucket,
        # reassemble per the nulls placement — each pass is stable, so applying
        # spec keys last-to-first yields the multi-key order
        nonnull = [r for r in out if r.get(field) is not None]
        nulls_b = [r for r in out if r.get(field) is None]
        nonnull.sort(key=lambda r: r[field], reverse=desc)
        out = (nulls_b + nonnull) if nulls == "nulls_first" else (nonnull + nulls_b)
    return out


def build_multikey(rng: random.Random) -> dict:
    names = ["ann", "bo", "cy", "dee", "ed", "fi"]
    cases = []
    while len(cases) < 8:
        n = rng.randint(3, 6)
        recs = []
        for i in range(n):
            recs.append({
                "name": rng.choice(names),
                "age": None if rng.random() < 0.25 else rng.randint(20, 40),
                "score": None if rng.random() < 0.25 else rng.randint(0, 5),
            })
        spec = [
            ("age", rng.choice(["asc", "desc"]), rng.choice(["nulls_first", "nulls_last"])),
            ("score", rng.choice(["asc", "desc"]), rng.choice(["nulls_first", "nulls_last"])),
        ]
        want = _mk_ref(recs, spec)
        got = _mk_passes(recs, spec)
        assert want == got, f"multikey cross-check failed: {recs} {spec}\n{want}\n{got}"
        cases.append((recs, spec, want))
    hand_recs = [
        {"name": "a", "age": 30, "score": 1},
        {"name": "b", "age": None, "score": 2},
        {"name": "c", "age": 30, "score": None},
        {"name": "d", "age": 25, "score": 2},
    ]
    hand_spec = [("age", "desc", "nulls_last"), ("score", "asc", "nulls_first")]
    hand_want = _mk_ref(hand_recs, hand_spec)
    assert hand_want == _mk_passes(hand_recs, hand_spec)
    tests = [f"assert sort_records({hand_recs!r}, {hand_spec!r}) == {hand_want!r}"]
    tests += [f"assert sort_records({r!r}, {s!r}) == {w!r}" for r, s, w in cases]
    # stability probe: equal keys keep input order
    stab = [{"name": "x", "age": 1, "score": 1}, {"name": "y", "age": 1, "score": 1}]
    tests.append(f"assert sort_records({stab!r}, [('age','asc','nulls_last')]) == {stab!r}")
    prompt = (
        "Implement `def sort_records(records: list, spec: list) -> list`. `records` is a list of "
        "dicts; `spec` is a list of `(field, direction, nulls)` tuples applied in priority order "
        "(first tuple = primary key). `direction` is 'asc' or 'desc'; `nulls` is 'nulls_first' or "
        "'nulls_last' and controls where records with `None` in that field go — INDEPENDENT of "
        "direction (nulls_last means last even under desc). The sort must be STABLE: records that "
        "compare equal on the whole spec keep their input order. Do not mutate the input; return "
        "a new list. Return only the function in a Python code block."
    )
    return {"id": "v2_multikey_nulls", "difficulty": "medium", "entry": "sort_records",
            "prompt": prompt, "tests": tests}


# ================================================================ parens

def _parens_brute(s: str) -> str:
    def balanced(t):
        d = 0
        for c in t:
            if c == "(":
                d += 1
            elif c == ")":
                d -= 1
                if d < 0:
                    return False
        return d == 0

    n = len(s)
    for k in range(n + 1):
        best = None
        for keep in itertools.combinations(range(n), n - k):
            t = "".join(s[i] for i in keep)
            if balanced(t) and (best is None or t < best):
                best = t
        if best is not None:
            return best
    return ""


def build_parens(rng: random.Random) -> dict:
    cases = []
    seen = set()
    while len(cases) < 9:
        s = "".join(rng.choice("()ab") for _ in range(rng.randint(1, 11)))
        if s in seen:
            continue
        seen.add(s)
        cases.append((s, _parens_brute(s)))
    hand = [
        ("(a)b)", "(a)b"),
        (")(", ""),
        ("(()", "()"),
        ("))((", ""),
        ("a(b)c", "a(b)c"),
        ("(a))(b", "(a)b"),   # min removals 2, lexicographic among candidates
    ]
    for s, w in hand:
        assert _parens_brute(s) == w, (s, _parens_brute(s))
    tests = [f"assert min_remove({s!r}) == {w!r}" for s, w in hand + cases]
    prompt = (
        "Implement `def min_remove(s: str) -> str`. `s` contains `(`, `)`, and lowercase letters. "
        "Remove the MINIMUM number of parentheses to make it balanced; among all results "
        "achievable with that minimum number of removals, return the LEXICOGRAPHICALLY SMALLEST "
        "string. Note `(` < `)` < letters in ASCII order. Letters are never removed.\n"
        "Inputs are short (<= 12 chars) — correctness matters, not asymptotics.\n"
        "Return only the function in a Python code block."
    )
    return {"id": "v2_paren_min_lex", "difficulty": "hard", "entry": "min_remove",
            "prompt": prompt, "tests": tests}


# ================================================================ cache

class _CacheRef:
    """Reference: dict of key -> [value, last_put_tick, last_access_tick]."""

    def __init__(self, cap, ttl):
        self.cap, self.ttl, self.d, self.seq = cap, ttl, {}, 0

    def _expire(self, now):
        for k in [k for k, (v, put, acc, s) in self.d.items() if now - put >= self.ttl]:
            del self.d[k]

    def put(self, k, v, now):
        self._expire(now)
        if k in self.d:
            self.d[k] = (v, now, now, self._next())
            return
        if len(self.d) >= self.cap:
            lru = min(self.d.items(), key=lambda kv: (kv[1][2], kv[1][3]))[0]
            del self.d[lru]
        self.d[k] = (v, now, now, self._next())

    def get(self, k, now):
        self._expire(now)
        if k not in self.d:
            return None
        v, put, acc, s = self.d[k]
        self.d[k] = (v, put, now, self._next())
        return v

    def _next(self):
        self.seq += 1
        return self.seq


def build_cache(rng: random.Random) -> dict:
    # generate op sequences; expected outputs from the reference
    case_snippets = []
    for ci in range(6):
        cap, ttl = rng.randint(2, 3), rng.randint(3, 6)
        ref = _CacheRef(cap, ttl)
        lines = [f"c = TTLCache({cap}, {ttl})"]
        now = 0
        keys = "abcd"
        for _ in range(rng.randint(6, 12)):
            now += rng.randint(0, 3)
            k = rng.choice(keys)
            if rng.random() < 0.5:
                v = rng.randint(1, 99)
                ref.put(k, v, now)
                lines.append(f"c.put({k!r}, {v}, {now})")
            else:
                want = ref.get(k, now)
                lines.append(f"assert c.get({k!r}, {now}) == {want!r}")
        case_snippets.append("\n".join(lines))
    hand = (
        "c = TTLCache(2, 5)\n"
        "c.put('a', 1, 0)\n"
        "c.put('b', 2, 1)\n"
        "assert c.get('a', 4) == 1\n"      # refreshes a's recency (not its TTL)
        "c.put('c', 3, 4)\n"               # evicts b (LRU), not a
        "assert c.get('b', 4) == None\n"
        "assert c.get('a', 5) == None\n"   # a expired: put at 0, ttl 5, 5-0 >= 5
        "assert c.get('c', 5) == 3"
    )
    tests = [hand] + case_snippets
    prompt = (
        "Implement `class TTLCache` with a LOGICAL clock (no wall time): every method takes the "
        "current tick `now` as its last argument; `now` never decreases.\n"
        "- `TTLCache(capacity, ttl)`.\n"
        "- `put(key, value, now)`: first, drop every entry whose age `now - put_tick >= ttl` "
        "(age counts from the tick of the entry's most recent put — `get` does NOT extend life). "
        "Then, if the key is absent and the cache is full, evict the least-recently-USED entry "
        "(use = get or put, whichever is latest; ties broken by evicting the one whose last use "
        "was earliest in program order). Insert/overwrite the key with put_tick = now.\n"
        "- `get(key, now)`: drop expired entries the same way, then return the value and mark the "
        "entry as used at `now`, or return `None` if absent.\n"
        "Return only the class in a Python code block."
    )
    return {"id": "v2_ttl_cache", "difficulty": "hard", "entry": "TTLCache",
            "prompt": prompt, "tests": tests}


# ================================================================ skyline
# LC-218 Hard: outline of overlapping rectangular buildings. Half-open columns
# [l, r); a building of height h covers columns l..r-1. Output the contour as
# [x, h] key points: the skyline BECOMES height h at column x. Cross-check: a
# per-column max scan vs a boundary-only scan (height changes only at endpoints).

def _skyline_ref(buildings) -> list:
    if not buildings:
        return []
    xmin = min(l for l, r, h in buildings)
    xmax = max(r for l, r, h in buildings)
    out, prev = [], 0
    for x in range(xmin, xmax + 1):
        h = 0
        for l, r, bh in buildings:
            if l <= x < r and bh > h:
                h = bh
        if h != prev:
            out.append([x, h])
            prev = h
    return out


def _skyline_alt(buildings) -> list:
    """Independent impl: sample only at endpoints (height is constant between)."""
    if not buildings:
        return []
    xs = sorted({l for l, r, h in buildings} | {r for l, r, h in buildings})
    out, prev = [], 0
    for x in xs:
        h = max((bh for l, r, bh in buildings if l <= x < r), default=0)
        if h != prev:
            out.append([x, h])
            prev = h
    return out


def build_skyline(rng: random.Random) -> dict:
    cases = []
    while len(cases) < 11:
        n = rng.randint(1, 6)
        segs = []
        for _ in range(n):
            l = rng.randint(0, 15)
            r = l + rng.randint(1, 8)
            segs.append((l, r, rng.randint(1, 9)))
        r1, r2 = _skyline_ref(segs), _skyline_alt(segs)
        assert r1 == r2, f"skyline cross-check failed: {segs}"
        cases.append((segs, r1))
    hand = [
        ([], []),
        ([(0, 2, 3)], [[0, 3], [2, 0]]),
        ([(0, 10, 3), (2, 5, 6)], [[0, 3], [2, 6], [5, 3], [10, 0]]),      # nested
        ([(0, 4, 3), (2, 6, 5)], [[0, 3], [2, 5], [6, 0]]),                # overlap, taller wins
        ([(0, 2, 3), (5, 7, 4)], [[0, 3], [2, 0], [5, 4], [7, 0]]),        # disjoint
        ([(0, 5, 7), (5, 10, 7)], [[0, 7], [10, 0]]),                      # adjacent, same h -> merge
        ([(0, 3, 3), (1, 5, 3)], [[0, 3], [5, 0]]),                        # overlap, same h -> merge
        ([(2, 9, 10), (3, 7, 15), (5, 12, 12)], [[2, 10], [3, 15], [7, 12], [12, 0]]),
    ]
    for segs, want in hand:
        assert _skyline_ref(segs) == want == _skyline_alt(segs), segs
    tests = [f"assert skyline({[list(x) for x in s]!r}) == {w!r}" for s, w in hand + cases]
    prompt = (
        "Implement `def skyline(buildings: list) -> list`. Each building is `[l, r, h]` and occupies "
        "the half-open column range [l, r) at height h (h >= 1, l < r, all integers). Return the "
        "skyline outline as a list of `[x, height]` key points, sorted by x: each point marks a "
        "column x where the skyline's height CHANGES to `height` (and stays there until the next "
        "point). The outline starts at height 0, and the final point returns it to 0 at the right "
        "edge. Never emit two consecutive points with the same height (merge equal-height runs, "
        "including across abutting buildings). Overlapping buildings take the max height. For empty "
        "input return []. Return only the function in a Python code block."
    )
    return {"id": "v2_skyline", "difficulty": "hard", "entry": "skyline",
            "prompt": prompt, "tests": tests}


# ================================================================ scc
# Strongly connected components of a directed graph. Report the component sizes,
# sorted descending. Reference = transitive-closure grouping; cross-check =
# Kosaraju two-pass DFS (fully independent structure).

def _scc_ref(n: int, edges) -> list:
    reach = [[i == j for j in range(n)] for i in range(n)]
    for a, b in edges:
        reach[a][b] = True
    for k in range(n):
        rk = reach[k]
        for i in range(n):
            if reach[i][k]:
                ri = reach[i]
                for j in range(n):
                    if rk[j]:
                        ri[j] = True
    seen, sizes = [False] * n, []
    for i in range(n):
        if seen[i]:
            continue
        comp = [j for j in range(n) if reach[i][j] and reach[j][i]]
        for j in comp:
            seen[j] = True
        sizes.append(len(comp))
    return sorted(sizes, reverse=True)


def _scc_alt(n: int, edges) -> list:
    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        radj[b].append(a)
    seen, order = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        stack = [(s, 0)]
        while stack:
            v, i = stack[-1]
            if i == 0:
                seen[v] = True
            if i < len(adj[v]):
                stack[-1] = (v, i + 1)
                w = adj[v][i]
                if not seen[w]:
                    stack.append((w, 0))
            else:
                order.append(v)
                stack.pop()
    comp = [-1] * n
    c = 0
    for s in reversed(order):
        if comp[s] != -1:
            continue
        st = [s]
        comp[s] = c
        while st:
            v = st.pop()
            for w in radj[v]:
                if comp[w] == -1:
                    comp[w] = c
                    st.append(w)
        c += 1
    sizes = [0] * c
    for x in comp:
        sizes[x] += 1
    return sorted(sizes, reverse=True)


def build_scc(rng: random.Random) -> dict:
    cases = []
    while len(cases) < 10:
        n = rng.randint(2, 7)
        edges = set()
        for _ in range(rng.randint(0, n * 2)):
            edges.add((rng.randint(0, n - 1), rng.randint(0, n - 1)))
        if rng.random() < 0.4:  # bias toward some non-trivial cycles
            cyc = rng.sample(range(n), rng.randint(2, n))
            edges |= {(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))}
        edges = sorted(edges)
        want = _scc_ref(n, edges)
        assert want == _scc_alt(n, edges), f"scc cross-check failed: {n} {edges}"
        cases.append((n, edges, want))
    hand = [
        (3, [], [1, 1, 1]),                                             # all singletons
        (1, [(0, 0)], [1]),                                             # self-loop
        (3, [(0, 1), (1, 2), (2, 0)], [3]),                            # one big cycle
        (4, [(0, 1), (1, 0), (2, 3), (3, 2)], [2, 2]),                 # two 2-cycles
        (2, [(0, 1)], [1, 1]),                                          # DAG
        (5, [(0, 1), (1, 2), (2, 0), (3, 4)], [3, 1, 1]),             # cycle + singletons
        (6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], [3, 3]),
    ]
    for n, e, w in hand:
        assert _scc_ref(n, e) == w == _scc_alt(n, e)
    tests = [f"assert scc_sizes({n}, {[list(x) for x in e]!r}) == {w!r}" for n, e, w in hand + cases]
    prompt = (
        "Implement `def scc_sizes(n: int, edges: list) -> list`. The graph is DIRECTED: nodes are "
        "0..n-1 and each edge `[a, b]` goes from a to b (self-loops and duplicate edges are "
        "possible). Find the strongly connected components — maximal sets of nodes where every node "
        "is reachable from every other node in the set following edge directions. Return the list of "
        "component SIZES sorted in DESCENDING order. A node with no cycle through it is its own "
        "component of size 1. Return only the function in a Python code block."
    )
    return {"id": "v2_scc_sizes", "difficulty": "hard", "entry": "scc_sizes",
            "prompt": prompt, "tests": tests}


# ================================================================ subseq count
# Number of DISTINCT subsequences of s equal to t, modulo 1e9+7 (LC-115 Hard).
# Reference = O(len(s)*len(t)) DP; cross-check on small s = brute over all
# 2^len(s) subsequences; large modular cases cross-checked by an independent
# closed-form (choose-counts) so the modular reduction path is genuinely tested.

_ND_MOD = 10**9 + 7


def _nd_ref(s: str, t: str) -> int:
    m = len(t)
    dp = [0] * (m + 1)
    dp[0] = 1
    for c in s:
        for j in range(m, 0, -1):
            if t[j - 1] == c:
                dp[j] = (dp[j] + dp[j - 1]) % _ND_MOD
    return dp[m]


def _nd_brute(s: str, t: str) -> int:
    n = len(s)
    cnt = 0
    for mask in range(1 << n):
        if "".join(s[i] for i in range(n) if (mask >> i) & 1) == t:
            cnt += 1
    return cnt % _ND_MOD


def build_subseqcount(rng: random.Random) -> dict:
    import math
    cases = []
    while len(cases) < 10:
        s = "".join(rng.choice("ab") for _ in range(rng.randint(2, 13)))
        t = "".join(rng.choice("ab") for _ in range(rng.randint(0, 4)))
        ref = _nd_ref(s, t)
        assert ref == _nd_brute(s, t), f"subseq cross-check failed: {(s, t)}"
        cases.append((s, t, ref))
    # large cases: s = a^p b^q, t = a^i b^j  =>  count = C(p,i)*C(q,j); big enough
    # that the answer exceeds the modulus, so a correct solution MUST reduce mod.
    big = []
    for p, q, i, j in [(40, 40, 20, 20), (60, 10, 30, 5), (55, 33, 27, 30)]:
        s, t = "a" * p + "b" * q, "a" * i + "b" * j
        ref = _nd_ref(s, t)
        assert ref == (math.comb(p, i) * math.comb(q, j)) % _ND_MOD
        assert ref != math.comb(p, i) * math.comb(q, j)  # confirm reduction actually happened
        big.append((s, t, ref))
    hand = [
        ("rabbbit", "rabbit", 3),   # classic LC-115
        ("babgbag", "bag", 5),      # classic LC-115
        ("", "", 1),                # empty subsequence of empty string
        ("abc", "", 1),             # empty target: exactly one (empty) subsequence
        ("a", "aa", 0),             # target longer than source
        ("aaa", "aa", 3),           # C(3,2)
        ("abc", "abc", 1),
    ]
    for s, t, v in hand:
        assert _nd_ref(s, t) == v
        if len(s) <= 14:
            assert _nd_brute(s, t) == v
    tests = [f"assert num_distinct({s!r}, {t!r}) == {v}" for s, t, v in hand + cases + big]
    prompt = (
        "Implement `def num_distinct(s: str, t: str) -> int`. Return the number of DISTINCT "
        "subsequences of `s` that equal `t`, taken MODULO 1_000_000_007. A subsequence is formed by "
        "deleting zero or more characters without changing the order of the remaining characters; "
        "two subsequences are counted separately when they use different sets of index positions "
        "(even if the resulting strings are identical). The empty string is a subsequence of every "
        "string, so `num_distinct(s, '')` is 1. The count can be astronomically large — reduce it "
        "modulo 1_000_000_007. Return only the function in a Python code block."
    )
    return {"id": "v2_subseq_count", "difficulty": "hard", "entry": "num_distinct",
            "prompt": prompt, "tests": tests}


# ================================================================ emit

BUILDERS = [build_expand, build_paint, build_topolex, build_multikey,
            build_wordwild, build_calc2, build_parens, build_cache,
            build_skyline, build_scc, build_subseqcount]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "hard_v2.yaml")
    args = ap.parse_args()

    tests = []
    for b in BUILDERS:
        rng = random.Random(f"{args.seed}:{b.__name__}")
        t = b(rng)
        tests.append({
            "id": t["id"],
            "difficulty": t["difficulty"],
            "prompt": t["prompt"],
            "graders": [{"type": "code_exec", "dimension": "correctness",
                         "entry": t["entry"], "timeout": 15, "tests": t["tests"]}],
        })
        print(f"{t['id']}: {len(t['tests'])} tests")

    doc = {
        "id": "code_generation_hard_v2",
        "name": "Code Generation (Hard v2, spec-delta, execution-graded)",
        "difficulty": "hard",
        "category": "coding",
        "tests": tests,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=110, allow_unicode=True)
    print(f"wrote {args.out} ({len(tests)} problems)")


if __name__ == "__main__":
    main()
