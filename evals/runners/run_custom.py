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
from graders.base import GradeResult, TaskResult
from graders.creative import (
    CHARACTER_VOICE_RUBRIC,
    ENGAGEMENT_RUBRIC,
    GM_QUALITY_RUBRIC,
    NARRATIVE_RUBRIC,
    RESEARCH_SYNTHESIS_RUBRIC,
    WORLD_BUILDING_RUBRIC,
)
from graders.langfuse_scorer import LangfuseScorer
from graders.llm_judge import LLMJudge
from graders.tool_channel import ToolChannelGrader
from openai import OpenAI

# Dimension-specific rubrics — override the generic DEFAULT_RUBRIC when available
DIMENSION_RUBRICS: dict[str, str] = {
    "narrative_quality": NARRATIVE_RUBRIC,
    "character_voice": CHARACTER_VOICE_RUBRIC,
    "world_building": WORLD_BUILDING_RUBRIC,
    "engagement": ENGAGEMENT_RUBRIC,
    "gm_quality": GM_QUALITY_RUBRIC,
    "research_synthesis": RESEARCH_SYNTHESIS_RUBRIC,
}


def load_task(task_path: Path) -> dict:
    """Load a task definition from YAML."""
    with open(task_path) as f:
        return yaml.safe_load(f)


def run_agent(client: OpenAI, model: str, task: dict, max_turns: int = 20, max_tokens: int = 32768) -> dict:
    """Execute a task by sending prompts to an agent via the gateway.

    Returns the full conversation trace and final output.
    """
    messages = [{"role": "system", "content": task.get("system_prompt", "You are a helpful assistant.")}]
    messages.append({"role": "user", "content": task["prompt"]})

    tools = task.get("tools", None)
    tool_calls_made = []
    tool_calls_detail = []
    turns = 0
    start = time.time()

    while turns < max_turns:
        extra_body = dict(task.get("extra_body") or {})
        ctk = dict(extra_body.get("chat_template_kwargs") or {})
        ctk.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = ctk

        # Qwen-recommended sampling params (vLLM-specific go in extra_body)
        # Thinking uses precise coding preset (temp=0.6, no presence_penalty)
        is_thinking = ctk.get("enable_thinking", False)
        extra_body.setdefault("top_k", 20)
        extra_body.setdefault("min_p", 0.0)
        extra_body.setdefault("presence_penalty", 0.0 if is_thinking else 1.5)
        extra_body.setdefault("repetition_penalty", 1.0)

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.6 if is_thinking else 0.7,
            "top_p": 0.95 if is_thinking else 0.8,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        turns += 1

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                tool_calls_made.append(tc.function.name)
                _args = tc.function.arguments
                try:
                    _args = json.loads(_args) if isinstance(_args, str) else _args
                except Exception:
                    pass
                tool_calls_detail.append({"name": tc.function.name, "arguments": _args})
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
        "_tool_calls_detail": tool_calls_detail,
        "_text_leaked_tool": _detect_tool_leak(final_content),
    }


# Heuristic: did the model emit a tool call as TEXT in content instead of using
# the structured tool_calls field? (the failure mode channel_correctness checks)
_TOOL_LEAK_PATTERNS = [
    r"<tool_call", r"</tool_call", r"<tool_code", r"```tool", r"<function",
    r"\[TOOL_CALL\]", r"print\(default_api\.", r'"name"\s*:\s*".+?"\s*,\s*"arguments"',
]


def _detect_tool_leak(text: str) -> bool:
    import re as _re
    t = text or ""
    return any(_re.search(p, t, _re.I) for p in _TOOL_LEAK_PATTERNS)


def grade_task(task: dict, output: dict, gateway_url: str = "http://ava:4000/v1", api_key: str = "not-needed", judge_url: str | None = None) -> list[GradeResult]:
    """Apply graders defined in the task config."""
    grades = []
    grader_configs = task.get("graders", [])

    # Default judge to the smart lane (:8000 / local, Qwen 27B+MTP). The fast lane (:8002)
    # is now DiffusionGemma, which can't guided-decode (HTTP 500) — judges must not route there.
    default_judge_url = judge_url or os.environ.get(
        "JUDGE_GATEWAY_URL", "http://localhost:8000/v1"
    )
    default_judge_model = os.environ.get("JUDGE_MODEL", "local")

    for gc in grader_configs:
        if gc["type"] == "match":
            # Deterministic exact/regex/number/format check — no LLM. For single-right-answer
            # or precise-format tasks (e.g. quant-sensitivity), where judge noise would mask
            # the very drift we're trying to detect.
            from graders.match import MatchGrader
            grader = MatchGrader(
                dimension=gc.get("dimension", "match"),
                mode=gc.get("mode", "contains"),
                expected=gc.get("expected"),
                case_sensitive=gc.get("case_sensitive", True),
                tolerance=gc.get("tolerance", 0.0),
            )
            grades.append(grader.grade(
                task_input={"prompt": task["prompt"]},
                task_output={"output": output.get("output", "")},
                expected=task.get("expected"),
            ))
        elif gc["type"] == "tool_channel":
            # Deterministic check of the tool-call channel — an LLM judge cannot
            # distinguish a structured tool_call from text in serialized output.
            grader = ToolChannelGrader(dimension=gc.get("dimension", "channel_correctness"))
            grades.append(grader.grade(
                task_input={"prompt": task["prompt"]},
                task_output=output,  # full trace: _tool_calls_detail, _text_leaked_tool
                expected=task.get("expected"),
            ))
        elif gc["type"] == "llm_judge":
            rubric = gc.get("rubric") or DIMENSION_RUBRICS.get(gc["dimension"])
            grader = LLMJudge(
                dimension=gc["dimension"],
                rubric=rubric,
                model=gc.get("model", default_judge_model),
                base_url=default_judge_url,
                api_key=api_key,
            )
            # Pass clean output to judge — text plus the tools actually called
            # via the structured tool_calls channel, so tool-aware dimensions
            # (channel_correctness, tool_selection) can see what the model did.
            clean_output = {
                "output": output.get("output", ""),
                "tools_called": output.get("_tools_called", []),
            }
            grades.append(grader.grade(
                task_input={"prompt": task["prompt"]},
                task_output=clean_output,
                expected=task.get("expected"),
            ))

    return grades


@click.command()
@click.option("--task", "task_path", type=click.Path(exists=True), help="Path to a single task YAML")
@click.option("--suite", help="Task suite directory name (e.g., tool_use, browser)")
@click.option("--model", default="protolabs/smart", help="Gateway model name")
@click.option("--trials", default=3, help="Trials per task")
@click.option("--gateway-url", default="http://ava:4000/v1")
@click.option("--api-key", envvar=["GATEWAY_API_KEY", "LITELLM_API_KEY"], default="not-needed")
@click.option("--submit-langfuse", is_flag=True, help="Submit scores to Langfuse")
@click.option("--thinking", is_flag=True, help="Enable thinking/reasoning mode (Gemma 4, etc.)")
@click.option("--output-dir", type=click.Path(), default=None, help="Directory to write result JSON")
@click.option("--max-tokens", default=32768, type=int, help="Max output tokens per turn. Lower (4096) for tiny models with 8K context.")
def main(task_path, suite, model, trials, gateway_url, api_key, submit_langfuse, thinking, output_dir, max_tokens):
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
    suite_results: dict[str, dict] = {}

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

        suite_task_results: list[dict] = []

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
            trial_records: list[dict] = []
            for trial in range(1, trials + 1):
                output = run_agent(client, model, task, max_tokens=max_tokens)
                judge_gateway = os.environ.get("JUDGE_GATEWAY_URL", "http://ava:4000/v1")
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

                trial_records.append({
                    "trial": trial,
                    "passed": result.passed,
                    "score": result.score,
                    "grades": [
                        {"dimension": g.dimension, "score": g.score, "passed": g.passed}
                        for g in grades
                    ],
                    "duration_s": output["duration_s"],
                    "turns": output["turns"],
                })

            all_passed = all(r.passed for r in trial_results)
            avg_score = sum(r.score for r in trial_results) / len(trial_results)
            click.echo(f"    Pass^{trials}: {'PASS' if all_passed else 'FAIL'} (avg={avg_score:.2f})")
            all_results.append(trial_results)

            suite_task_results.append({
                "task_id": task_id,
                "trials": trial_records,
                "all_passed": all_passed,
                "avg_score": avg_score,
            })

        suite_results[suite_name] = {
            "tasks": suite_task_results,
        }

    # Summary
    total_tasks = len(all_results)
    passed_tasks = sum(1 for trials_list in all_results if all(r.passed for r in trials_list))
    click.echo(f"\n{'='*60}")
    click.echo(f"Results: {passed_tasks}/{total_tasks} tasks passed (pass^{trials})")

    # Per-trial mean-score band (run-to-run noise) — policy: report 3-trial mean±spread.
    if trials > 1 and total_tasks > 0:
        import statistics
        per_trial = [
            sum(tl[i].score for tl in all_results) / total_tasks
            for i in range(trials)
        ]
        mean = sum(per_trial) / trials
        half_range = (max(per_trial) - min(per_trial)) / 2
        click.echo(
            f"Mean score: {mean:.3f} ± {half_range:.3f}  "
            f"(range {min(per_trial):.3f}-{max(per_trial):.3f}, std {statistics.pstdev(per_trial):.3f}, n={trials})"
        )

    # Write structured results (merge with existing file if present —
    # the profile runner calls us once per suite into the same output dir)
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_file = out_path / "custom_results.json"

        # Load existing results to merge suites across invocations
        existing: dict = {}
        if results_file.exists():
            with open(results_file) as f:
                existing = json.load(f)

        merged_suites = existing.get("suites", {})
        merged_suites.update(suite_results)

        # Recompute summary across all merged suites
        all_task_results = [t for s in merged_suites.values() for t in s.get("tasks", [])]
        merged_total = len(all_task_results)
        merged_passed = sum(1 for t in all_task_results if t.get("all_passed"))

        results_data = {
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trials": trials,
            "thinking": thinking,
            "suites": merged_suites,
            "summary": {
                "total_tasks": merged_total,
                "passed_tasks": merged_passed,
                "pass_rate": merged_passed / merged_total if merged_total > 0 else 0.0,
            },
        }
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        click.echo(f"Results written to {results_file}")


if __name__ == "__main__":
    main()
