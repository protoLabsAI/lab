"""Deterministic JSON validator grader — parse + assertion battery.

For structured-output tasks where correctness is a set of machine-checkable
invariants (schema shape, cross-field consistency, exact computed values), not
string similarity. Extracts the first JSON object/array from the model output
(fenced block preferred, else greedy brace match), then evaluates a list of
Python boolean expressions with the parsed value bound as `data`. Score =
fraction of assertions that hold (partial credit, code_exec-style).

YAML:
    graders:
      - type: json_validate
        dimension: structure
        assertions:
          - "isinstance(data, dict) and set(data) == {'moves', 'final'}"
          - "sum(data['final'].values()) == 480"
          - "all(v >= 0 for v in data['final'].values())"

Assertions run under eval() with builtins restricted to a safe arithmetic /
collection subset — they are authored by us in task YAML (trusted), the
restriction is defense-in-depth against accidents, not a sandbox.
"""

from __future__ import annotations

import json
import re

from graders.base import Grader, GradeResult

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "max": max,
    "min": min, "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip, "range": range,
    "enumerate": enumerate,
}


def extract_json(text: str):
    """First fenced JSON block, else first balanced {...} or [...] in the text."""
    candidates = [b.strip() for b in _FENCE.findall(text or "")]
    if not candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = (text or "").find(opener)
            while start != -1:
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == opener:
                        depth += 1
                    elif text[i] == closer:
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start:i + 1])
                            break
                if candidates:
                    break
                start = text.find(opener, start + 1)
            if candidates:
                break
    for c in candidates:
        try:
            return json.loads(c), None
        except json.JSONDecodeError as e:
            last_err = str(e)
    return None, (last_err if candidates else "no JSON found in output")


class JsonValidateGrader(Grader):
    def __init__(self, dimension: str = "structure", assertions: list[str] | None = None,
                 threshold: float = 0.999):
        self.dimension = dimension
        self.assertions = assertions or []
        self.threshold = threshold

    def grade(self, task_input: dict, task_output: dict, expected=None) -> GradeResult:
        if not self.assertions:
            return GradeResult(self.dimension, 0.0, False, reasoning="no assertions defined")
        out = task_output.get("output", "") if isinstance(task_output, dict) else str(task_output)
        data, err = extract_json(out)
        if data is None:
            return GradeResult.from_threshold(
                self.dimension, 0.0, self.threshold,
                reasoning=f"unparseable: {err} (0/{len(self.assertions)})")

        # Bind `data` in the GLOBALS namespace, not locals: a generator/list
        # comprehension inside eval() runs in its own scope that can see the
        # eval globals but NOT the eval locals, so `data` referenced in a
        # comprehension body would raise NameError if passed as a local.
        passed, fails = 0, []
        for i, a in enumerate(self.assertions):
            try:
                ok = bool(eval(a, {"__builtins__": _SAFE_BUILTINS, "data": data}, {}))
            except Exception as e:
                ok = False
                fails.append(f"  a{i}: raised {type(e).__name__}: {e}")
            else:
                if not ok:
                    fails.append(f"  a{i}: false: {a[:90]}")
            passed += ok
        score = passed / len(self.assertions)
        reasoning = (f"{passed}/{len(self.assertions)} assertions hold"
                     + ("\n" + "\n".join(fails[:3]) if fails else ""))
        return GradeResult.from_threshold(self.dimension, score, self.threshold,
                                          reasoning=reasoning)
