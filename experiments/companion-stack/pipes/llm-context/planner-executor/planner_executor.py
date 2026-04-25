"""
Planner/Executor loop.

Planner  — Qwen3.6-27B-FP8 (local,       :8000, thinking ON)
Executor — Qwen3.6-35B-A3B-FP8 (local-voice, :8002, thinking OFF)

Flow:
  1. Planner decomposes the goal into a structured Plan.
  2. Executor runs each step, calling tools as needed.
  3. Planner reviews the results and decides: pass | retry | replan | abort.
  4. On retry/replan, loop back up to 2 times, then abort.
"""

from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from models import (
    ExecutionResult,
    LoopResult,
    Plan,
    PlanStep,
    ReviewDecision,
)

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

PLANNER_URL   = "http://localhost:8000/v1"
EXECUTOR_URL  = "http://localhost:8002/v1"
PLANNER_MODEL = "local"
EXECUTOR_MODEL = "local-voice"
MAX_REPLANS = 2

planner  = OpenAI(base_url=PLANNER_URL,  api_key="none")
executor = OpenAI(base_url=EXECUTOR_URL, api_key="none")


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """\
You are a planning agent. Given a goal and optionally the results of a \
previous execution attempt, produce a structured step-by-step plan as JSON.

Return ONLY valid JSON matching this schema (no markdown fences, no prose):
{
  "goal": "<restate goal concisely>",
  "steps": [
    {
      "step_id": 1,
      "description": "<what to do>",
      "tool": "<tool_name or null>",
      "args": {},
      "depends_on": []
    }
  ],
  "success_criteria": "<how to know the goal is achieved>"
}

Rules:
- Keep plans tight — 2 to 6 steps for most goals.
- Use tool=null for pure reasoning/synthesis steps.
- Set depends_on to step_ids that must complete before this step runs.
- Do not explain. Return JSON only.
"""

REVIEWER_SYSTEM = """\
You are a review agent. Given a goal, a plan, and the execution results, \
decide whether the task succeeded.

Return ONLY valid JSON (no markdown, no prose):
{
  "verdict": "pass" | "retry" | "replan" | "abort",
  "failed_step": <step_id or null>,
  "reason": "<brief reason, required for retry/replan/abort>",
  "revised_steps": [<PlanStep objects> or null]
}

Verdict rules:
- "pass"   — goal achieved, results look correct.
- "retry"  — one step failed but the plan is still valid; just re-run that step.
- "replan" — the plan itself was wrong; provide revised_steps to fix it.
- "abort"  — goal is unachievable with available tools; stop and explain in reason.

Return JSON only.
"""

EXECUTOR_SYSTEM = """\
You are an execution agent. You are given a single step to execute. \
Call the appropriate tool if one is specified, or reason and produce \
a result if no tool is needed.

Reply with a concise result. Do not plan, do not review — just execute.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(content: str, model_cls: type) -> Any:
    """Strip markdown fences and parse JSON into a Pydantic model."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return model_cls.model_validate_json(text)


def _planner_call(messages: list[dict], label: str = "plan") -> tuple[str, float]:
    t0 = time.perf_counter()
    # Review passes don't need CoT — disable thinking for fast JSON verdict.
    # Plan/replan passes keep thinking enabled for quality decomposition.
    thinking_on = label != "review"
    resp = planner.chat.completions.create(
        model=PLANNER_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking_on}},
    )
    latency = time.perf_counter() - t0
    content = resp.choices[0].message.content or ""
    print(f"  [planner/{label}] {'🧠' if thinking_on else '⚡'} {latency:.2f}s — {len(content)} chars")
    return content, latency


def _executor_call(
    step: PlanStep,
    context: str,
    tools: list[dict] | None = None,
) -> tuple[ExecutionResult, float]:
    messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Context so far:\n{context}\n\n"
                f"Step {step.step_id}: {step.description}\n"
                + (f"Tool to call: {step.tool}\nArgs: {json.dumps(step.args)}" if step.tool else "No tool — reason and produce a result.")
            ),
        },
    ]
    kwargs: dict = dict(
        model=EXECUTOR_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    if tools and step.tool:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    t0 = time.perf_counter()
    resp = executor.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0

    msg = resp.choices[0].message
    tool_calls = []
    output = msg.content or ""

    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append({
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments or "{}"),
            })
        output = f"Called {len(tool_calls)} tool(s): " + ", ".join(t["name"] for t in tool_calls)

    status: str = "ok"
    if not output and not tool_calls:
        status = "error"
        output = "(no output from executor)"

    result = ExecutionResult(
        step_id=step.step_id,
        status=status,    # type: ignore[arg-type]
        output=output,
        tool_calls=tool_calls,
    )
    print(f"  [executor/step {step.step_id}] {latency:.2f}s — {status}: {output[:80]}")
    return result, latency


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    task_id: str,
    goal: str,
    tools: list[dict] | None = None,
) -> LoopResult:
    """
    Run the planner/executor loop for a single task.

    Args:
        task_id: Unique identifier for this task (used in results).
        goal:    Natural-language goal for the agent.
        tools:   Optional list of OpenAI-format tool definitions.

    Returns:
        LoopResult with full trace and metrics.
    """
    print(f"\n{'='*60}")
    print(f"TASK {task_id}: {goal}")
    print(f"{'='*60}")

    plan_latency_total = 0.0
    exec_latency_total = 0.0
    review_latency_total = 0.0
    replans = 0
    retries = 0
    all_results: list[ExecutionResult] = []
    current_plan: Plan | None = None
    final_answer = ""
    review = ReviewDecision(verdict="abort", reason="loop did not complete")

    # -- Phase 1: initial plan --
    plan_messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"Goal: {goal}"},
    ]
    raw, lat = _planner_call(plan_messages, label="initial")
    plan_latency_total += lat

    try:
        current_plan = _parse_json_response(raw, Plan)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"  [planner] JSON parse failed: {e}")
        return LoopResult(
            task_id=task_id, goal=goal, success=False,
            total_steps=0, completed_steps=0, replans=0, retries=0,
            final_answer="Planner failed to produce a valid plan.",
            plan=Plan(goal=goal, steps=[], success_criteria=""),
            execution_results=[], review=ReviewDecision(verdict="abort", reason=str(e)),
            plan_latency=plan_latency_total, execution_latency=0.0,
            review_latency=0.0, total_latency=plan_latency_total,
        )

    for attempt in range(MAX_REPLANS + 1):
        print(f"\n  [loop] attempt {attempt + 1}/{MAX_REPLANS + 1} — {len(current_plan.steps)} steps")

        # -- Phase 2: execute each step in order --
        step_results: list[ExecutionResult] = []
        context_lines: list[str] = []

        for step in current_plan.steps:
            # wait for dependencies
            dep_ok = all(
                any(r.step_id == dep and r.status == "ok" for r in step_results)
                for dep in step.depends_on
            )
            if step.depends_on and not dep_ok:
                step_results.append(ExecutionResult(
                    step_id=step.step_id, status="skipped",
                    output="Skipped: dependency failed.", tool_calls=[],
                ))
                continue

            result, lat = _executor_call(step, "\n".join(context_lines), tools)
            exec_latency_total += lat
            step_results.append(result)
            context_lines.append(f"Step {step.step_id} result: {result.output}")

        all_results.extend(step_results)

        # -- Phase 3: planner reviews --
        results_summary = "\n".join(
            f"  Step {r.step_id}: [{r.status.upper()}] {r.output}"
            for r in step_results
        )
        review_messages = [
            {"role": "system", "content": REVIEWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Plan:\n{current_plan.model_dump_json(indent=2)}\n\n"
                    f"Execution results:\n{results_summary}"
                ),
            },
        ]
        raw_review, lat = _planner_call(review_messages, label="review")
        review_latency_total += lat

        try:
            review = _parse_json_response(raw_review, ReviewDecision)
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"  [reviewer] JSON parse failed: {e} — treating as abort")
            review = ReviewDecision(verdict="abort", reason=f"reviewer parse error: {e}")

        print(f"  [reviewer] verdict={review.verdict} reason={review.reason or '—'}")

        if review.verdict == "pass":
            # Synthesise final answer from last step output
            final_answer = step_results[-1].output if step_results else "(no output)"
            break

        if review.verdict == "abort":
            final_answer = review.reason or "Task aborted."
            break

        if attempt >= MAX_REPLANS:
            final_answer = f"Max replans reached. Last result: {step_results[-1].output if step_results else '(none)'}"
            break

        if review.verdict == "retry" and review.failed_step is not None:
            retries += 1
            # keep same plan but re-run just the failed step
            failed = next((s for s in current_plan.steps if s.step_id == review.failed_step), None)
            if failed:
                current_plan = Plan(
                    goal=current_plan.goal,
                    steps=[failed],
                    success_criteria=current_plan.success_criteria,
                )

        elif review.verdict == "replan":
            replans += 1
            if review.revised_steps:
                current_plan = Plan(
                    goal=current_plan.goal,
                    steps=review.revised_steps,
                    success_criteria=current_plan.success_criteria,
                )
            else:
                # ask planner to replan from scratch with failure context
                plan_messages = [
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Goal: {goal}\n\n"
                            f"Previous plan failed:\n{results_summary}\n\n"
                            f"Reviewer said: {review.reason}\n\n"
                            "Produce a revised plan."
                        ),
                    },
                ]
                raw, lat = _planner_call(plan_messages, label="replan")
                plan_latency_total += lat
                try:
                    current_plan = _parse_json_response(raw, Plan)
                except (ValidationError, json.JSONDecodeError) as e:
                    print(f"  [planner] replan parse failed: {e}")
                    break

    total_latency = plan_latency_total + exec_latency_total + review_latency_total
    completed = sum(1 for r in all_results if r.status == "ok")
    success = review.verdict == "pass"

    print(f"\n  [result] success={success} steps={completed}/{len(all_results)} "
          f"replans={replans} retries={retries} total={total_latency:.2f}s")

    return LoopResult(
        task_id=task_id,
        goal=goal,
        success=success,
        total_steps=len(current_plan.steps) if current_plan else 0,
        completed_steps=completed,
        replans=replans,
        retries=retries,
        final_answer=final_answer,
        plan=current_plan or Plan(goal=goal, steps=[], success_criteria=""),
        execution_results=all_results,
        review=review,
        plan_latency=plan_latency_total,
        execution_latency=exec_latency_total,
        review_latency=review_latency_total,
        total_latency=total_latency,
    )
