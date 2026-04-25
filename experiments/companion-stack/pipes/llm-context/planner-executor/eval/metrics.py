"""Scoring utilities for the planner/executor eval."""
from __future__ import annotations
from dataclasses import dataclass, field
from models import LoopResult


@dataclass
class TaskScore:
    task_id: str
    category: str
    success: bool
    completed_steps: int
    total_steps: int
    replans: int
    retries: int
    plan_latency: float
    execution_latency: float
    review_latency: float
    total_latency: float


@dataclass
class RunSummary:
    baseline: str
    scores: list[TaskScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def success_rate(self) -> float:
        return sum(s.success for s in self.scores) / self.n if self.n else 0.0

    @property
    def avg_total_latency(self) -> float:
        return sum(s.total_latency for s in self.scores) / self.n if self.n else 0.0

    @property
    def avg_plan_latency(self) -> float:
        return sum(s.plan_latency for s in self.scores) / self.n if self.n else 0.0

    @property
    def avg_exec_latency(self) -> float:
        return sum(s.execution_latency for s in self.scores) / self.n if self.n else 0.0

    @property
    def replan_rate(self) -> float:
        return sum(s.replans for s in self.scores) / self.n if self.n else 0.0

    def by_category(self) -> dict[str, list[TaskScore]]:
        cats: dict[str, list[TaskScore]] = {}
        for s in self.scores:
            cats.setdefault(s.category, []).append(s)
        return cats

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"BASELINE: {self.baseline}")
        print(f"{'='*60}")
        print(f"  Tasks:          {self.n}")
        print(f"  Success rate:   {self.success_rate:.0%}")
        print(f"  Avg latency:    {self.avg_total_latency:.2f}s total "
              f"(plan {self.avg_plan_latency:.2f}s + exec {self.avg_exec_latency:.2f}s)")
        print(f"  Replan rate:    {self.replan_rate:.2f} per task")
        print()
        for cat, scores in self.by_category().items():
            sr = sum(s.success for s in scores) / len(scores)
            avg_lat = sum(s.total_latency for s in scores) / len(scores)
            print(f"  [{cat}] {sum(s.success for s in scores)}/{len(scores)} "
                  f"({sr:.0%}) avg {avg_lat:.2f}s")
        print()
        for s in sorted(self.scores, key=lambda x: x.task_id):
            icon = "✅" if s.success else "❌"
            print(f"  {icon} {s.task_id} {s.total_latency:.2f}s "
                  f"steps={s.completed_steps}/{s.total_steps} "
                  f"replans={s.replans}")


def score_result(result: LoopResult, category: str) -> TaskScore:
    return TaskScore(
        task_id=result.task_id,
        category=category,
        success=result.success,
        completed_steps=result.completed_steps,
        total_steps=result.total_steps,
        replans=result.replans,
        retries=result.retries,
        plan_latency=result.plan_latency,
        execution_latency=result.execution_latency,
        review_latency=result.review_latency,
        total_latency=result.total_latency,
    )
