"""Run custom eval tasks against agents via the gateway.

Custom tasks are defined in tasks/ and evaluated using our grading framework.
Results are submitted to Langfuse for tracking and comparison.

Usage:
    python -m runners.run_custom --suite tool_use --model local
    python -m runners.run_custom --task tasks/tool_use/calculator.yaml --model local
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import yaml
from openai import OpenAI

from graders.base import GradeResult, TaskResult
from graders.outcome import OutcomeGrader
from graders.llm_judge import LLMJudge
from graders.langfuse_scorer import LangfuseScorer


def load_task(task_path: Path) -> dict:
    """Load a task definition from YAML."""
    with open(task_path) as f:
        return yaml.safe_load(f)


def run_agent(client: OpenAI, model: str, task: dict, max_turns: int = 20) -> dict:
    """Execute a task by sending prompts to an agent via the gateway.

    Returns the full conversation trace and final output.
    """
    messages = [{"role": "system", "content": task.get("system_prompt", "You are a helpful assistant.")}]
    messages.append({"role": "user", "content": task["prompt"]})

    tools = task.get("tools", None)
    tool_calls_made = []
    turns = 0
    start = time.time()

    while turns < max_turns:
        kwargs = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 8000}
        if tools:
            kwargs["tools"] = tools
        extra_body = dict(task.get("extra_body") or {})
        ctk = dict(extra_body.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = ctk
        kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        turns += 1

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                tool_calls_made.append(tc.function.name)
                # Simulate tool response (custom tasks provide mock responses)
                tool_response = task.get("mock_tool_responses", {}).get(tc.function.name, '{"status": "ok"}')
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_response if isinstance(tool_response, str) else json.dumps(tool_response),
                })
        else:
            # Final response — combine content + reasoning for thinking models
            final_content = choice.message.content or ""
            if not final_content:
                # Thinking models put output in reasoning_content when they use all tokens thinking
                rc = getattr(choice.message, "reasoning_content", None) or ""
                # Also check provider_specific_fields
                psf = getattr(choice.message, "provider_specific_fields", {}) or {}
                rc = rc or psf.get("reasoning_content", "") or psf.get("reasoning", "")
                if rc:
                    final_content = rc
            messages.append({"role": "assistant", "content": final_content})
            break

    elapsed = time.time() - start

    return {
        "output": final_content,
        "messages": messages,
        "turns": turns,
        "duration_s": round(elapsed, 2),
        "_tools_called": tool_calls_made,
    }


def grade_task(task: dict, output: dict, gateway_url: str = "http://localhost:4000/v1", api_key: str = "not-needed", judge_url: str | None = None) -> list[GradeResult]:
    """Apply graders defined in the task config."""
    grades = []
    grader_configs = task.get("graders", [])

    for gc in grader_configs:
        if gc["type"] == "llm_judge":
            grader = LLMJudge(
                dimension=gc["dimension"],
                rubric=gc.get("rubric"),
                model=gc.get("model", "claude-sonnet-4-6"),
                base_url=judge_url or gateway_url,
                api_key=api_key,
            )
            # Pass clean output to judge — just the text, not the full trace
            clean_output = {"output": output.get("output", "")}
            grades.append(grader.grade(
                task_input={"prompt": task["prompt"]},
                task_output=clean_output,
                expected=task.get("expected"),
            ))

    return grades


@click.command()
@click.option("--task", "task_path", type=click.Path(exists=True), help="Path to a single task YAML")
@click.option("--suite", help="Task suite directory name (e.g., tool_use, browser)")
@click.option("--model", default="local", help="Gateway model name")
@click.option("--trials", default=3, help="Trials per task")
@click.option("--gateway-url", default="http://localhost:4000/v1")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--submit-langfuse", is_flag=True, help="Submit scores to Langfuse")
@click.option("--thinking", is_flag=True, help="Enable thinking/reasoning mode (Gemma 4, etc.)")
def main(task_path, suite, model, trials, gateway_url, api_key, submit_langfuse, thinking):
    """Run custom eval tasks and grade results."""
    client = OpenAI(base_url=gateway_url, api_key=api_key)
    scorer = LangfuseScorer() if submit_langfuse else None

    # Collect task files
    tasks_dir = Path(__file__).parent.parent / "tasks"
    if task_path:
        task_files = [Path(task_path)]
    elif suite:
        suite_dir = tasks_dir / suite
        task_files = sorted(suite_dir.glob("*.yaml"))
    else:
        click.echo("Specify --task or --suite")
        sys.exit(1)

    click.echo(f"Model: {model} | Trials: {trials} | Tasks: {len(task_files)}")
    click.echo("=" * 60)

    all_results = []

    for tf in task_files:
        file_data = load_task(tf)
        suite_name = file_data.get("name", tf.stem)

        # Support both formats:
        # 1. Single task: top-level "prompt" field
        # 2. Multi-test: "tests" array with per-test "prompt" fields
        if "tests" in file_data:
            tasks = file_data["tests"]
            # Inherit system_prompt from file level
            system_prompt = file_data.get("system_prompt")
            for t in tasks:
                if system_prompt and "system_prompt" not in t:
                    t["system_prompt"] = system_prompt
                # Inherit graders from file level if not on test
                if "graders" not in t and "graders" in file_data:
                    t["graders"] = file_data["graders"]
        else:
            tasks = [file_data]

        click.echo(f"\n{suite_name} ({len(tasks)} tests)")

        for task in tasks:
            if thinking:
                eb = dict(task.get("extra_body") or {})
                ctk = dict(eb.get("chat_template_kwargs") or {})
                ctk["enable_thinking"] = True
                eb["chat_template_kwargs"] = ctk
                task["extra_body"] = eb
            task_id = task.get("id", file_data.get("id", tf.stem))
            click.echo(f"\n  {task_id}:")

            trial_results = []
            for trial in range(1, trials + 1):
                output = run_agent(client, model, task)
                judge_gateway = os.environ.get("JUDGE_GATEWAY_URL", "http://100.101.189.45:4000/v1")
                grades = grade_task(task, output, gateway_url=gateway_url, api_key=api_key, judge_url=judge_gateway)

                result = TaskResult(
                    task_id=task_id,
                    trial=trial,
                    grades=grades,
                    model=model,
                )
                trial_results.append(result)

                status = "PASS" if result.passed else "FAIL"
                click.echo(f"    Trial {trial}: {status} (score={result.score:.2f}, turns={output['turns']}, {output['duration_s']}s)")

                if scorer:
                    scorer.submit_grades(result)

            all_passed = all(r.passed for r in trial_results)
            avg_score = sum(r.score for r in trial_results) / len(trial_results)
            click.echo(f"    Pass^{trials}: {'PASS' if all_passed else 'FAIL'} (avg={avg_score:.2f})")
            all_results.append(trial_results)

    # Summary
    total_tasks = len(all_results)
    passed_tasks = sum(1 for trials_list in all_results if all(r.passed for r in trials_list))
    click.echo(f"\n{'='*60}")
    click.echo(f"Results: {passed_tasks}/{total_tasks} tasks passed (pass^{trials})")


if __name__ == "__main__":
    main()
