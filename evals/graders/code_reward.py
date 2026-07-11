"""Hardened execution reward for RL (Gate 1) — NOT the eval grader.

`code_exec.py` is the *eval* grader: partial credit, shared-namespace runner. That
shape is correct for measurement (partial credit makes the suite discriminate) but is
a reward-hacking gradient the moment it becomes an RL reward — and its runner execs the
model's solution into the *same* `globals()` as the hidden test battery, which enables
the entire documented attack catalog (overwrite/delete tests, monkeypatch scoring,
replace asserts with passing prints, exit(0) before grading, hardcode outputs; see
arxiv 2604.15149 "LLMs Gaming Verifiers", and experiments/agentic-coding-rl/RESEARCH.md).

This module is the RL reward. Design (mirrors Ornith's 3-layer anti-hack):
  * SPARSE BINARY — reward is 1.0 iff *all* hidden tests pass, else 0.0. No partial
    credit (the field converged here: DeepSWE, Kimi-Dev). `passed/total` is still
    reported in metadata for logging/curriculum, but never enters the reward.
  * LAYER 1 — trust boundary. The solution execs in its own namespace dict with its own
    `__builtins__`; grading uses stdlib references captured *before* the solution runs;
    each test executes in a fresh namespace seeded with a *clean* pre-snapshot of
    builtins (so rebinding `AssertionError`/`print` in the solution can't reach the
    grader). Results are emitted over a private pipe framed with a random nonce sent
    out-of-band (never via argv/env, both of which are scrubbed before the solution
    runs) — so monkeypatching stdout/`os.write` can't forge a pass.
  * LAYER 2 — zero-and-exclude monitor. A static red-flag scan of the solution source
    (exit calls, builtins/fd manipulation, references to grader internals) flags gaming:
    reward 0 AND `exclude=True` so the trajectory is dropped from the advantage estimate,
    not merely zeroed (Ornith layer-2; zeroing alone still leaves a learnable "don't do
    that exact thing" signal — exclusion removes it from the batch entirely).

Threat model: the adversary is a *policy under RL following the reward gradient*, not a
skilled human exploiting the box. We defend against reward-inflation (faking a pass),
not sandbox escape — the lab box is trusted single-user; the untrusted-agent Docker
sandbox is a separate concern (see code_exec.py note). rlimits + wall timeout bound
runaway rollouts.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field

# Reuse the eval grader's code extraction — identical "pull the solution out of the
# model response" logic; only the execution/scoring downstream differs.
from graders.code_exec import extract_code


@dataclass
class RewardResult:
    """RL reward for one (solution, tests) pair.

    `reward` is the only value that should flow into GRPO advantages. `exclude=True`
    means drop this trajectory from the batch entirely (gaming detected or, optionally,
    a timeout under Compact-Filtering). `passed`/`total` are for logging only.
    """

    reward: float  # 0.0 or 1.0 — sparse binary
    passed: int
    total: int
    gamed: bool = False
    exclude: bool = False
    reason: str = ""
    metadata: dict = field(default_factory=dict)


# --- Layer 2: static red-flag scan -------------------------------------------------
# Word-boundary patterns that a legitimate algorithmic solution has ~no reason to use,
# but a reward-hacking policy reaches for to tamper with grading or abort before it.
_GAMING_PATTERNS: list[tuple[str, str]] = [
    (r"\bos\s*\.\s*_exit\b", "os._exit (abort interpreter to skip/forge grading)"),
    (r"\bsys\s*\.\s*exit\b", "sys.exit (abort before grading)"),
    (r"\bexit\s*\(", "exit()/quit() call"),
    (r"\bquit\s*\(", "quit() call"),
    (r"\braise\s+SystemExit\b", "raise SystemExit"),
    (r"\bimport\s+builtins\b", "imports builtins (rebind AssertionError/print)"),
    (r"\b__builtins__\b", "touches __builtins__"),
    (r"\bAssertionError\s*=", "rebinds AssertionError"),
    (r"\bsys\s*\.\s*settrace\b", "sys.settrace (intercept execution)"),
    (r"\bsys\s*\.\s*setprofile\b", "sys.setprofile"),
    (r"\bos\s*\.\s*write\b", "os.write (forge grade channel)"),
    (r"\bos\s*\.\s*dup2?\b", "os.dup/dup2 (fd manipulation)"),
    (r"/proc/self/fd", "reads/writes raw fds via /proc"),
    (r"\b__GRADE__\b", "references grader sentinel"),
    (r"\b__results\b", "references grader internals (__results)"),
    (r"\b__TESTS\b", "references grader internals (__TESTS)"),
    (r"\bunittest\s*\.\s*mock\b", "unittest.mock (monkeypatch)"),
    (r"\bmonkeypatch\b", "monkeypatch"),
]
_GAMING_RE = [(re.compile(p), why) for p, why in _GAMING_PATTERNS]


def scan_for_gaming(code: str) -> list[str]:
    """Return human-readable reasons if the solution source trips a red flag, else []."""
    hits = []
    for rx, why in _GAMING_RE:
        if rx.search(code):
            hits.append(why)
    return hits


# --- The isolated driver (static; takes NO injected user content) ------------------
# Everything (solution, tests, setup, nonce) arrives over the tests pipe as one JSON
# blob, so there is no string-injection-into-a-template surface. The driver captures
# stdlib refs and a clean builtins snapshot BEFORE running any untrusted code, scrubs
# argv/env, runs the solution in its own namespace, then grades each test in a fresh
# namespace with the clean builtins, and writes a nonce-framed result to the result fd.
_DRIVER = textwrap.dedent(
    r"""
    import sys, os, json, builtins, traceback

    _tfd = int(sys.argv[1]); _rfd = int(sys.argv[2])

    # Read the whole blob from the tests pipe, then close it so the solution can't
    # reach back for the nonce.
    _chunks = []
    while True:
        _b = os.read(_tfd, 65536)
        if not _b:
            break
        _chunks.append(_b)
    os.close(_tfd)
    _blob = json.loads(b"".join(_chunks).decode())
    _nonce = _blob["nonce"]; _tests = _blob["tests"]
    _setup = _blob.get("setup", ""); _solution = _blob["solution"]

    # Capture references to the exact functions/objects the grader needs, BEFORE any
    # untrusted code can rebind them. Monkeypatching os.write/json.dumps/builtins after
    # this point cannot affect grading.
    _write = os.write
    _dumps = json.dumps
    _clean_builtins = dict(vars(builtins))

    # Belt-and-suspenders: remove the fd numbers / any secrets from places the solution
    # can read. The nonce never lived here (it came over the pipe).
    sys.argv = [""]
    try:
        os.environ.clear()
    except Exception:
        pass

    def _emit(payload):
        _write(_rfd, (_nonce + "\x1e" + _dumps(payload) + "\n").encode())
        os.close(_rfd)

    # Solution runs in its OWN namespace with its OWN builtins copy — it cannot see the
    # grader's locals (_write/_dumps/_nonce) and its mutations stay contained here.
    _sol_ns = {"__name__": "__solution__", "__builtins__": dict(vars(builtins))}
    try:
        if _setup:
            exec(compile(_setup, "<setup>", "exec"), _sol_ns)
        exec(compile(_solution, "<solution>", "exec"), _sol_ns)
    except BaseException:
        _err = (traceback.format_exc().strip().splitlines() or ["error"])[-1][:200]
        _emit({"nonce": _nonce, "solution_error": _err,
               "results": [[False, _err] for _ in _tests]})
        os._exit(0)

    _results = []
    for _t in _tests:
        # Fresh namespace per test: solution's defined names are available, but builtins
        # is the CLEAN snapshot — a failing assert raises the real AssertionError even if
        # the solution rebound it in its own namespace.
        _tg = dict(_sol_ns)
        _tg["__builtins__"] = _clean_builtins
        try:
            exec(compile(_t, "<test>", "exec"), _tg)
            _results.append([True, ""])
        except BaseException:
            # Catch BaseException so a solution calling sys.exit() *inside* a tested
            # function fails only that test instead of aborting the battery.
            _msg = (traceback.format_exc().strip().splitlines() or ["error"])[-1][:200]
            _results.append([False, _msg])

    _emit({"nonce": _nonce, "results": _results})
    os._exit(0)
    """
)


def _limit():
    # CPU seconds + ~1.5GB address space; block core dumps. Best-effort (same as code_exec).
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_AS, (1_500 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


def _run_isolated(solution: str, tests: list[str], setup: str, timeout: int, nonce: str):
    """Run the driver in a subprocess. Returns (results | None, reason).

    results is a list of [passed_bool, msg] the length of `tests`; None means the frame
    never arrived (crash/timeout/forge-attempt with no valid nonce line).
    """
    tfd_r, tfd_w = os.pipe()  # parent -> child: the blob
    rfd_r, rfd_w = os.pipe()  # child -> parent: the framed result
    blob = json.dumps({"nonce": nonce, "tests": tests, "setup": setup, "solution": solution})

    # Write the driver to a temp file (kept minimal; the driver reads everything else
    # over the pipe). We don't reuse cwd — child gets a throwaway one.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        drv = os.path.join(td, "driver.py")
        with open(drv, "w") as fh:
            fh.write(_DRIVER)
        try:
            proc = subprocess.Popen(
                [sys.executable, drv, str(tfd_r), str(rfd_w)],
                cwd=td, pass_fds=(tfd_r, rfd_w), preexec_fn=_limit,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except Exception as e:
            for fd in (tfd_r, tfd_w, rfd_r, rfd_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
            return None, f"spawn failed: {e}"

        # Parent no longer needs the child-side ends.
        os.close(tfd_r)
        os.close(rfd_w)
        os.write(tfd_w, blob.encode())
        os.close(tfd_w)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            os.close(rfd_r)
            return None, f"timeout after {timeout}s"

        # Driver has written its single framed line and exited; drain the pipe.
        data = b""
        while True:
            try:
                chunk = os.read(rfd_r, 65536)
            except OSError:
                break
            if not chunk:
                break
            data += chunk
        os.close(rfd_r)

    # Accept ONLY a line framed with our nonce. A solution flooding the pipe with forged
    # frames can't produce the nonce (scrubbed), so those lines are ignored.
    marker = (nonce + "\x1e").encode()
    idx = data.rfind(marker)
    if idx < 0:
        return None, "no valid result frame (crash or forge attempt)"
    tail = data[idx + len(marker):].split(b"\n", 1)[0]
    try:
        payload = json.loads(tail.decode())
    except Exception as e:
        return None, f"malformed result frame: {e}"
    if payload.get("nonce") != nonce:
        return None, "nonce mismatch"
    if "solution_error" in payload:
        return payload["results"], f"solution error: {payload['solution_error']}"
    return payload["results"], ""


def score(solution_text: str, tests: list[str], *, setup: str = "", entry: str | None = None,
          timeout: int = 10, exclude_on_timeout: bool = False) -> RewardResult:
    """Compute the sparse-binary RL reward for a model response against hidden tests.

    Args:
      solution_text: raw model output (code is extracted from it).
      tests: hidden test asserts (kept out of the model's cwd/prompt by the data pipeline).
      setup: optional code prepended into the solution namespace (e.g. "import math").
      entry: optional fn/class name to disambiguate which fenced block is the solution.
      timeout: wall-clock seconds for the whole battery.
      exclude_on_timeout: DeepSWE Compact-Filtering — drop timed-out trajectories from
        advantage. Default False (SWE-Master found this masking can hurt; A/B it).
    """
    if not tests:
        return RewardResult(0.0, 0, 0, reason="no tests defined")

    code = extract_code(solution_text, entry)
    if not code.strip():
        return RewardResult(0.0, 0, 0, reason="no code found in output")

    # Layer 2: zero-and-exclude on static red flags, BEFORE we even run it.
    hits = scan_for_gaming(code)
    if hits:
        return RewardResult(0.0, 0, len(tests), gamed=True, exclude=True,
                            reason="gaming detected: " + "; ".join(hits),
                            metadata={"flags": hits, "code": code})

    nonce = _make_nonce()
    results, reason = _run_isolated(code, tests, setup, timeout, nonce)
    n = len(tests)

    if results is None:
        excl = exclude_on_timeout and reason.startswith("timeout")
        return RewardResult(0.0, 0, n, exclude=excl, reason=reason, metadata={"code": code})

    passed = sum(1 for ok, _ in results if ok)
    reward = 1.0 if passed == n else 0.0  # sparse binary
    detail = [{"test": tests[i], "passed": ok, "error": msg}
              for i, (ok, msg) in enumerate(results)]
    note = reason or f"{passed}/{n} tests passed"
    return RewardResult(reward, passed, n, reason=note,
                        metadata={"results": detail, "code": code})


def _make_nonce() -> str:
    # secrets is fine here; the workflow-script Math.random ban does not apply (this is
    # ordinary Python, not a resumable workflow script).
    import secrets

    return secrets.token_hex(16)
