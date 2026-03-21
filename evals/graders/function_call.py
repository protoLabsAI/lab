"""Function calling grader — structured tool call accuracy.

Evaluates whether the model produces correct function/tool calls:
- Name accuracy: correct function selected
- Argument accuracy: correct parameter values
- Structural validity: well-formed JSON, required params present
- Parallel call accuracy: all expected calls made in multi-tool scenarios
"""

from __future__ import annotations

import json
from typing import Any

from graders.base import Grader, GradeResult


class FunctionCallGrader(Grader):
    """Grade function/tool call accuracy against expected calls."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def grade(self, task_input: dict, task_output: dict, expected: dict | None = None) -> GradeResult:
        if expected is None:
            return GradeResult.from_threshold("function_call", 0.0, self.threshold,
                                              reasoning="No expected output provided")

        expected_calls = expected.get("tool_calls", [])
        actual_calls = task_output.get("tool_calls", [])

        if not expected_calls:
            # No calls expected — check model didn't call anything
            if not actual_calls:
                return GradeResult.from_threshold("function_call", 1.0, self.threshold,
                                                  reasoning="Correctly made no tool calls")
            return GradeResult.from_threshold("function_call", 0.0, self.threshold,
                                              reasoning=f"Expected no calls, got {len(actual_calls)}")

        scores = []
        details = []

        for i, exp in enumerate(expected_calls):
            exp_name = exp.get("name", "")
            exp_args = exp.get("arguments", {})

            # Find best matching actual call
            best_score = 0.0
            best_detail = f"Expected call '{exp_name}' not found"

            for act in actual_calls:
                act_name = act.get("name", "") if isinstance(act, dict) else ""
                act_args = act.get("arguments", {}) if isinstance(act, dict) else {}

                # Parse stringified args
                if isinstance(act_args, str):
                    try:
                        act_args = json.loads(act_args)
                    except (json.JSONDecodeError, TypeError):
                        act_args = {}

                call_score = 0.0
                call_detail = []

                # Name match (50% weight)
                if act_name == exp_name:
                    call_score += 0.5
                    call_detail.append("name: match")
                else:
                    call_detail.append(f"name: expected '{exp_name}', got '{act_name}'")
                    continue  # wrong function, skip arg checking

                # Argument match (50% weight)
                if exp_args:
                    arg_matches = 0
                    arg_total = len(exp_args)
                    for key, exp_val in exp_args.items():
                        act_val = act_args.get(key)
                        if act_val is not None and _values_match(act_val, exp_val):
                            arg_matches += 1
                        else:
                            call_detail.append(f"arg '{key}': expected {exp_val!r}, got {act_val!r}")
                    arg_score = arg_matches / arg_total if arg_total > 0 else 1.0
                    call_score += 0.5 * arg_score
                    call_detail.append(f"args: {arg_matches}/{arg_total}")
                else:
                    call_score += 0.5
                    call_detail.append("args: none expected")

                if call_score > best_score:
                    best_score = call_score
                    best_detail = "; ".join(call_detail)

            scores.append(best_score)
            details.append(f"call {i+1}: {best_detail} ({best_score:.2f})")

        # Penalize extra unexpected calls
        extra_calls = max(0, len(actual_calls) - len(expected_calls))
        extra_penalty = min(0.2, extra_calls * 0.05)

        final_score = max(0.0, (sum(scores) / len(scores)) - extra_penalty) if scores else 0.0

        return GradeResult.from_threshold(
            dimension="function_call",
            score=final_score,
            threshold=self.threshold,
            reasoning="; ".join(details) + (f" (-{extra_penalty:.2f} for {extra_calls} extra calls)" if extra_calls else ""),
        )


def _values_match(actual: Any, expected: Any) -> bool:
    """Fuzzy value comparison for function call arguments."""
    # Exact match
    if actual == expected:
        return True
    # String comparison (case-insensitive)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.lower().strip() == expected.lower().strip()
    # Numeric comparison (with tolerance)
    try:
        return abs(float(actual) - float(expected)) < 0.01
    except (ValueError, TypeError):
        pass
    # String representation match
    return str(actual).strip() == str(expected).strip()
