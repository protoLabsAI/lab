"""Langfuse scoring integration — submit and retrieve eval scores.

Bridges our grading framework with Langfuse's scoring system.
Scores attached to traces enable the eval flywheel:
  production traces → failing traces → datasets → experiments → compare
"""

from __future__ import annotations

import os
from typing import Any

from langfuse import Langfuse

from graders.base import GradeResult, TaskResult


class LangfuseScorer:
    """Submit grading results to Langfuse as scores."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ):
        self.client = Langfuse(
            public_key=public_key or os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.environ.get("LANGFUSE_SECRET_KEY"),
            host=host or os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_URL", "http://localhost:3001"),
        )

    def submit_grades(self, result: TaskResult) -> list[str]:
        """Submit all grades from a TaskResult to Langfuse. Returns score IDs."""
        score_ids = []
        for grade in result.grades:
            score = self.client.score(
                trace_id=result.trace_id,
                name=f"eval/{grade.dimension}",
                value=grade.score,
                comment=grade.reasoning,
                data_type="NUMERIC",
            )
            score_ids.append(score.id)

        # Submit aggregate pass/fail
        self.client.score(
            trace_id=result.trace_id,
            name="eval/passed",
            value=1.0 if result.passed else 0.0,
            comment=f"Aggregate: {result.score:.2f} ({sum(g.passed for g in result.grades)}/{len(result.grades)} dimensions passed)",
            data_type="NUMERIC",
        )
        self.client.flush()
        return score_ids

    def create_dataset(self, name: str, description: str = "") -> None:
        """Create a Langfuse dataset for eval tasks."""
        self.client.create_dataset(name=name, description=description)

    def add_dataset_item(
        self,
        dataset_name: str,
        input: dict,
        expected_output: dict | None = None,
        metadata: dict | None = None,
        source_trace_id: str | None = None,
    ) -> None:
        """Add an eval task as a dataset item."""
        self.client.create_dataset_item(
            dataset_name=dataset_name,
            input=input,
            expected_output=expected_output,
            metadata=metadata or {},
            source_trace_id=source_trace_id,
        )

    def run_experiment(
        self,
        dataset_name: str,
        experiment_name: str,
        task_fn,
        evaluators: list | None = None,
    ):
        """Run a dataset experiment through Langfuse.

        task_fn signature: task_fn(*, item, **kwargs) -> dict
        """
        dataset = self.client.get_dataset(dataset_name)
        return dataset.run_experiment(
            name=experiment_name,
            task=task_fn,
            evaluators=evaluators or [],
        )

    def get_traces_for_scoring(
        self,
        name: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list:
        """Fetch recent traces for manual or batch scoring."""
        return self.client.fetch_traces(
            name=name,
            tags=tags,
            limit=limit,
        ).data
