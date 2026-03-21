"""Outcome-based graders — deterministic checks on final state.

Grade what the agent *produced*, not the path it took.
These are fast, cheap, and reproducible.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from graders.base import Grader, GradeResult


class OutcomeGrader(Grader):
    """Deterministic outcome checking with partial credit.

    Supports multiple assertions per dimension, each weighted equally.
    Partial credit = fraction of assertions that pass.
    """

    def __init__(self, dimension: str, assertions: list[Callable], threshold: float = 0.75):
        self.dimension = dimension
        self.assertions = assertions
        self.threshold = threshold

    def grade(self, task_input: dict, task_output: dict, expected: dict | None = None) -> GradeResult:
        results = []
        for assertion_fn in self.assertions:
            try:
                passed = assertion_fn(task_output, expected)
                results.append(bool(passed))
            except Exception as e:
                results.append(False)

        score = sum(results) / len(results) if results else 0.0
        return GradeResult.from_threshold(
            dimension=self.dimension,
            score=score,
            threshold=self.threshold,
            reasoning=f"{sum(results)}/{len(results)} assertions passed",
        )


# ── Common assertion factories ──────────────────────────────────────────────


def contains_string(key: str, expected_substr: str) -> Callable:
    """Assert that output[key] contains a substring."""
    def check(output: dict, _expected: dict | None) -> bool:
        return expected_substr.lower() in str(output.get(key, "")).lower()
    return check


def matches_regex(key: str, pattern: str) -> Callable:
    """Assert that output[key] matches a regex pattern."""
    def check(output: dict, _expected: dict | None) -> bool:
        return bool(re.search(pattern, str(output.get(key, ""))))
    return check


def equals(key: str) -> Callable:
    """Assert output[key] equals expected[key]."""
    def check(output: dict, expected: dict | None) -> bool:
        if expected is None:
            return False
        return output.get(key) == expected.get(key)
    return check


def file_exists(path: str) -> Callable:
    """Assert a file was created at path."""
    import os
    def check(output: dict, _expected: dict | None) -> bool:
        return os.path.exists(path)
    return check


def json_valid(key: str) -> Callable:
    """Assert output[key] is valid JSON."""
    def check(output: dict, _expected: dict | None) -> bool:
        try:
            json.loads(str(output.get(key, "")))
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    return check


def tool_was_called(tool_name: str) -> Callable:
    """Assert that a specific tool was invoked (check trace metadata)."""
    def check(output: dict, _expected: dict | None) -> bool:
        tools_used = output.get("_tools_called", [])
        return tool_name in tools_used
    return check
