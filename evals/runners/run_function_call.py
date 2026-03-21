"""Run function calling accuracy evals.

Tests structured tool call correctness: name, arguments, structure.

Usage:
    python -m runners.run_function_call --model local --suite basic
    python -m runners.run_function_call --model local --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from openai import OpenAI

from graders.function_call import FunctionCallGrader
from graders.base import TaskResult
from graders.langfuse_scorer import LangfuseScorer

SUITE_DIR = Path(__file__).parent.parent / "function_call" / "test_cases"


def run_function_call_test(client: OpenAI, model: str, test: dict) -> dict:
    """Send a prompt with tools and capture the model's tool calls."""
    messages = [{"role": "user", "content": test["prompt"]}]
    tools = test.get("tools", [])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.0,
            max_tokens=1000,
        )
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = tc.function.arguments
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": args,
                })

        return {
            "tool_calls": tool_calls,
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
        }
    except Exception as e:
        return {"tool_calls": [], "content": "", "error": str(e)}


@click.command()
@click.option("--model", default="local", help="Gateway model name")
@click.option("--suite", default=None, help="Test suite name (e.g., basic)")
@click.option("--all-suites", is_flag=True, help="Run all test suites")
@click.option("--gateway-url", default="http://localhost:4000/v1")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--submit-langfuse", is_flag=True, help="Submit scores to Langfuse")
def main(model, suite, all_suites, gateway_url, api_key, submit_langfuse):
    """Run function calling accuracy evals."""
    client = OpenAI(base_url=gateway_url, api_key=api_key)
    grader = FunctionCallGrader()
    scorer = LangfuseScorer() if submit_langfuse else None

    # Collect test files
    if suite:
        files = sorted(SUITE_DIR.glob(f"{suite}.yaml"))
    elif all_suites:
        files = sorted(SUITE_DIR.glob("*.yaml"))
    else:
        click.echo("Specify --suite <name> or --all-suites")
        sys.exit(1)

    if not files:
        click.echo(f"No test files found in {SUITE_DIR}")
        sys.exit(1)

    click.echo(f"Model: {model} | Suites: {len(files)}")
    click.echo("=" * 60)

    total_tests = 0
    total_passed = 0

    for test_file in files:
        with open(test_file) as f:
            suite_data = yaml.safe_load(f)

        suite_name = suite_data.get("name", test_file.stem)
        tests = suite_data.get("tests", [])
        click.echo(f"\n{suite_name} ({len(tests)} tests)")

        for test in tests:
            output = run_function_call_test(client, model, test)
            result = grader.grade(
                task_input={"prompt": test["prompt"]},
                task_output=output,
                expected=test.get("expected", {}),
            )

            total_tests += 1
            if result.passed:
                total_passed += 1

            status = "PASS" if result.passed else "FAIL"
            click.echo(f"  {test.get('id', '?')}: {status} ({result.score:.2f}) — {result.reasoning[:80]}")

    click.echo(f"\n{'='*60}")
    click.echo(f"Results: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.0f}%)")


if __name__ == "__main__":
    main()
