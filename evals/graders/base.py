"""Base grader interface.

All graders produce GradeResult objects with a score, dimension name,
and optional reasoning. Grade outcomes, not paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GradeResult:
    """Result of grading a single dimension of a task."""

    dimension: str  # e.g. "correctness", "tool_usage", "safety"
    score: float  # 0.0 - 1.0
    passed: bool  # score >= threshold
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_threshold(cls, dimension: str, score: float, threshold: float = 0.75, **kwargs):
        return cls(dimension=dimension, score=score, passed=score >= threshold, **kwargs)


@dataclass
class TaskResult:
    """Aggregated result across all grading dimensions for a task."""

    task_id: str
    trial: int
    grades: list[GradeResult]
    trace_id: str | None = None
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if not self.grades:
            return 0.0
        return sum(g.score for g in self.grades) / len(self.grades)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.grades)

    @property
    def pass_k(self) -> dict[str, float]:
        """Individual dimension pass rates."""
        return {g.dimension: g.score for g in self.grades}


class Grader(ABC):
    """Base class for all graders."""

    @abstractmethod
    def grade(self, task_input: dict, task_output: dict, expected: dict | None = None) -> GradeResult:
        """Grade a single dimension of a task execution."""
        ...
