from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: int
    description: str
    tool: str | None = None          # None = LLM-only reasoning step
    args: dict = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    success_criteria: str


class ExecutionResult(BaseModel):
    step_id: int
    status: Literal["ok", "error", "skipped"]
    output: str
    tool_calls: list[dict] = Field(default_factory=list)


class ReviewDecision(BaseModel):
    verdict: Literal["pass", "retry", "replan", "abort"]
    failed_step: int | None = None
    reason: str | None = None
    revised_steps: list[PlanStep] | None = None  # populated on "replan"


class LoopResult(BaseModel):
    task_id: str
    goal: str
    success: bool
    total_steps: int
    completed_steps: int
    replans: int
    retries: int
    final_answer: str
    plan: Plan
    execution_results: list[ExecutionResult]
    review: ReviewDecision
    # latency breakdown (seconds)
    plan_latency: float
    execution_latency: float
    review_latency: float
    total_latency: float
