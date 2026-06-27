"""Deterministic match grader — exact / contains / regex / number / all_of.

No LLM. Built for the quant-sensitivity suite: quant drift (FP8/INT4) shows up as a
flipped digit, a dropped token, or a broken format — things an LLM judge smooths over but
an exact check catches cleanly, with zero judge noise. Use this for any task that has a
single right answer or a precise required format.

YAML:
    graders:
      - type: match
        dimension: exact_recall
        mode: contains            # exact | contains | regex | number | all_of
        expected: "PURPLE-TIGER-42"
        case_sensitive: true      # optional (default true)
      - type: match
        dimension: arithmetic
        mode: number              # extracts the last number in the output
        expected: 130227          # compared within tolerance
        tolerance: 0              # abs tolerance (0 = exact)
      - type: match
        dimension: format
        mode: all_of              # expected is a list; partial credit = fraction present
        expected: ["\"status\":", "\"code\": 200", "```json"]
"""

from __future__ import annotations

import re

from graders.base import Grader, GradeResult

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


class MatchGrader(Grader):
    def __init__(self, dimension: str, mode: str = "contains", expected=None,
                 case_sensitive: bool = True, tolerance: float = 0.0, threshold: float = 0.999):
        self.dimension = dimension
        self.mode = mode
        self.expected = expected
        self.case_sensitive = case_sensitive
        self.tolerance = tolerance
        self.threshold = threshold

    def _norm(self, s: str) -> str:
        s = s.strip()
        return s if self.case_sensitive else s.lower()

    def grade(self, task_input: dict, task_output: dict, expected=None) -> GradeResult:
        exp = self.expected if self.expected is not None else expected
        out = task_output.get("output", "") if isinstance(task_output, dict) else str(task_output)
        score, reason = 0.0, ""

        if self.mode == "exact":
            score = 1.0 if self._norm(out) == self._norm(str(exp)) else 0.0
            reason = f"exact {'==' if score else '!='} {exp!r}"
        elif self.mode == "contains":
            hay = self._norm(out)
            score = 1.0 if self._norm(str(exp)) in hay else 0.0
            reason = f"{'found' if score else 'missing'} {exp!r}"
        elif self.mode == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            score = 1.0 if re.search(str(exp), out, flags) else 0.0
            reason = f"regex {'matched' if score else 'no match'}: {exp!r}"
        elif self.mode == "number":
            nums = [float(m.replace(",", "")) for m in _NUM.findall(out)]
            if not nums:
                reason = "no number found in output"
            else:
                got = nums[-1]  # last number = the answer in step-by-step outputs
                target = float(exp)
                ok = abs(got - target) <= self.tolerance
                score = 1.0 if ok else 0.0
                reason = f"got {got}, expected {target} (tol {self.tolerance}) -> {'ok' if ok else 'MISS'}"
        elif self.mode == "all_of":
            items = exp if isinstance(exp, list) else [exp]
            hay = self._norm(out)
            hits = [it for it in items if self._norm(str(it)) in hay]
            score = len(hits) / len(items) if items else 0.0
            reason = f"{len(hits)}/{len(items)} required fragments present"
        else:
            reason = f"unknown match mode {self.mode!r}"

        return GradeResult.from_threshold(self.dimension, score, self.threshold, reasoning=reason)
