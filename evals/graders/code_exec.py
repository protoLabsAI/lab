"""Execution-based code grader — runs the model's code against hidden tests.

Why this exists: LLM-judge grading of code tops out. A 35B-class judge rewards
plausible-looking code and can't reliably catch subtle wrongness, so a strong
model ceilings the coding suite (0.95) and the eval stops discriminating. The
only objective signal for hard coding is *running it*. This grader extracts the
solution from the model output, runs it against a hidden test battery in a
sandboxed subprocess, and scores = fraction of tests that pass (partial credit
→ scores spread, which is what makes the suite discriminate again).

Task config (per grader):
    - type: code_exec
      dimension: correctness
      language: python            # python only for now
      entry: longest_run          # optional: fn/class name to disambiguate the code block
      timeout: 10                 # seconds, whole battery
      setup: "import math"        # optional, prepended to the test runner
      tests:                      # list of independent assert snippets (partial credit)
        - "assert solve([1,2,3]) == 6"
        - "assert solve([]) == 0"

Sandboxing: subprocess + wall-clock timeout + CPU/address-space rlimits + a
throwaway temp cwd. Not network-isolated (trusted single-user lab box; the
Docker sandbox is reserved for the untrusted *agentic* claw path). Model code
for algorithmic problems has no reason to phone home, but treat scores from
untrusted prompts accordingly.
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import textwrap

from graders.base import GradeResult, Grader

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str, entry: str | None = None) -> str:
    """Pull the solution code out of a model response.

    Prefer fenced ```python blocks. If `entry` is given, take the last block that
    defines it (models often show a sketch then the final version). Otherwise take
    the longest block. Fall back to the raw text (model answered with bare code).
    """
    blocks = [b.strip() for b in _FENCE.findall(text or "")]
    if not blocks:
        return (text or "").strip()
    if entry:
        named = [b for b in blocks if re.search(rf"\b(def|class)\s+{re.escape(entry)}\b", b)]
        if named:
            return named[-1]
    return max(blocks, key=len)


_RUNNER = textwrap.dedent(
    """
    import json, sys, traceback
    {setup}
    # --- model solution ---
    {solution}
    # --- hidden test battery ---
    __TESTS = json.loads({tests_json!r})
    __results = []
    for __t in __TESTS:
        try:
            exec(compile(__t, "<test>", "exec"), globals())
            __results.append([True, ""])
        except Exception:
            __results.append([False, traceback.format_exc().strip().splitlines()[-1][:200]])
    print("__GRADE__" + json.dumps(__results))
    """
)


def _limit():
    # CPU seconds + ~1.5GB address space; block core dumps. Best-effort.
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
        resource.setrlimit(resource.RLIMIT_AS, (1_500 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass


class CodeExecGrader(Grader):
    def __init__(self, dimension: str = "correctness", tests: list[str] | None = None,
                 entry: str | None = None, setup: str = "", timeout: int = 10,
                 language: str = "python"):
        self.dimension = dimension
        self.tests = tests or []
        self.entry = entry
        self.setup = setup or ""
        self.timeout = timeout
        self.language = language

    def grade(self, task_input: dict, task_output: dict, expected: dict | None = None) -> GradeResult:
        if self.language != "python":
            return GradeResult(self.dimension, 0.0, False, reasoning=f"unsupported language {self.language}")
        if not self.tests:
            return GradeResult(self.dimension, 0.0, False, reasoning="no tests defined")

        code = extract_code(task_output.get("output", ""), self.entry)
        if not code.strip():
            return GradeResult(self.dimension, 0.0, False, reasoning="no code found in output")

        script = _RUNNER.format(setup=self.setup, solution=code, tests_json=json.dumps(self.tests))
        n = len(self.tests)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "run.py")
            with open(path, "w") as fh:
                fh.write(script)
            try:
                proc = subprocess.run(
                    [sys.executable, path], cwd=td, capture_output=True, text=True,
                    timeout=self.timeout, preexec_fn=_limit,
                    env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except subprocess.TimeoutExpired:
                return GradeResult(self.dimension, 0.0, False, reasoning=f"timeout after {self.timeout}s (0/{n})")

        line = next((l for l in proc.stdout.splitlines() if l.startswith("__GRADE__")), None)
        if line is None:
            # solution itself failed to import/compile — whole battery is 0
            err = (proc.stderr.strip().splitlines() or ["unknown error"])[-1][:200]
            return GradeResult(self.dimension, 0.0, False, reasoning=f"solution error: {err} (0/{n})")

        results = json.loads(line[len("__GRADE__"):])
        passed = sum(1 for ok, _ in results if ok)
        score = passed / n
        fails = [f"  test{i}: {msg}" for i, (ok, msg) in enumerate(results) if not ok][:3]
        reasoning = f"{passed}/{n} tests passed" + ("\n" + "\n".join(fails) if fails else "")
        # per-test results paired with the assert text, so callers (e.g. a refinement
        # loop) can show the model exactly which cases failed and why.
        per_test = [{"test": self.tests[i], "passed": ok, "error": msg}
                    for i, (ok, msg) in enumerate(results)]
        return GradeResult.from_threshold(self.dimension, score, reasoning=reasoning,
                                          metadata={"passed": passed, "total": n,
                                                    "results": per_test, "code": code})
