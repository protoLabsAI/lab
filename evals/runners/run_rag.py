"""Run RAG quality evaluations.

Evaluates retrieval-augmented generation on four dimensions:
groundedness, context relevance, answer relevance, faithfulness.

Usage:
    python -m runners.run_rag --model local --corpus technical_docs
    python -m runners.run_rag --model local --test-file rag/test_cases.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from openai import OpenAI

from graders.rag import all_rag_judges
from graders.base import TaskResult
from graders.langfuse_scorer import LangfuseScorer

RAG_DIR = Path(__file__).parent.parent / "rag"


def generate_answer(client: OpenAI, model: str, question: str, context: str) -> str:
    """Generate an answer given a question and retrieved context."""
    messages = [
        {"role": "system", "content": f"Answer the question using only the provided context.\n\nContext:\n{context}"},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1000,
    )
    return response.choices[0].message.content or ""


@click.command()
@click.option("--model", default="local", help="Gateway model name to generate answers")
@click.option("--judge-model", default="claude-sonnet-4-6", help="Model for LLM-as-judge scoring")
@click.option("--test-file", type=click.Path(exists=True), help="YAML file with test cases")
@click.option("--gateway-url", default="http://localhost:4000/v1")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--submit-langfuse", is_flag=True, help="Submit scores to Langfuse")
def main(model, judge_model, test_file, gateway_url, api_key, submit_langfuse):
    """Run RAG quality evaluations."""
    client = OpenAI(base_url=gateway_url, api_key=api_key)
    judges = all_rag_judges(model=judge_model, base_url=gateway_url, api_key=api_key)
    scorer = LangfuseScorer() if submit_langfuse else None

    if not test_file:
        # Look for default test cases
        default_file = RAG_DIR / "test_cases.yaml"
        if default_file.exists():
            test_file = str(default_file)
        else:
            click.echo("Specify --test-file or create rag/test_cases.yaml")
            sys.exit(1)

    with open(test_file) as f:
        data = yaml.safe_load(f)

    tests = data.get("tests", [])
    click.echo(f"Model: {model} | Judge: {judge_model} | Tests: {len(tests)}")
    click.echo("=" * 60)

    for test in tests:
        question = test["question"]
        context = test["context"]
        expected_answer = test.get("expected_answer")

        click.echo(f"\nQ: {question[:80]}...")

        # Generate answer
        answer = generate_answer(client, model, question, context)
        click.echo(f"A: {answer[:80]}...")

        # Grade all dimensions
        for judge in judges:
            result = judge.grade(
                task_input={"prompt": question},
                task_output={"output": answer},
                expected={"context": context, "expected_answer": expected_answer},
            )
            status = "PASS" if result.passed else "FAIL"
            click.echo(f"  {result.dimension}: {result.score:.2f} ({status}) — {result.reasoning[:60]}")

    click.echo(f"\n{'='*60}")
    click.echo("Done. View detailed scores in Langfuse." if submit_langfuse else "Done.")


if __name__ == "__main__":
    main()
