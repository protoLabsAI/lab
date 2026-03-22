"""Run WildBench evaluation — 1,024 real user tasks from WildChat.

Two-phase pipeline:
  1. Inference: generate model responses for all tasks
  2. Judging: score responses using a judge model (GPT-4o recommended)

WB-Score correlates 0.89 with Chatbot Arena Elo ratings.

Usage:
    # Phase 1: Generate responses
    python -m runners.run_wildbench infer --model local --gateway-url http://localhost:8000/v1

    # Phase 2: Judge responses (uses cloud model via gateway)
    python -m runners.run_wildbench judge --results results/wildbench/local.json --judge gpt-5.4

    # Both phases
    python -m runners.run_wildbench run --model local --judge gpt-5.4

    # Show results
    python -m runners.run_wildbench show --results results/wildbench/local.scored.json
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from openai import OpenAI

RESULTS_DIR = Path(__file__).parent.parent / "results" / "wildbench"

# WB-Score judge template (adapted from WildBench evaluation/eval_template.score.v2.md)
JUDGE_TEMPLATE = """\
You are a helpful assistant tasked with evaluating an AI assistant's response to a user query.

## Conversation History
{conversation}

## Model Response
{response}

## Task-Specific Checklist
{checklist}

## Evaluation Instructions

Evaluate the model's response based on the conversation context and checklist above.
Consider: helpfulness, accuracy, depth, creativity, relevance, and following instructions.

Provide your evaluation as JSON with exactly these fields:
- "strengths": brief description of what the response does well
- "weaknesses": brief description of what could be improved
- "score": integer from 1-10 where:
  1-3: Poor (incorrect, unhelpful, or harmful)
  4-5: Below average (partially addresses the query but significant issues)
  6-7: Good (addresses the query well with minor issues)
  8-9: Very good (comprehensive, accurate, well-structured)
  10: Excellent (exceptional quality, could not be meaningfully improved)

Respond with ONLY valid JSON, no other text."""


def load_wildbench():
    """Load the WildBench dataset from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildBench", "v2", split="test")
    return ds


def format_conversation(messages: list[dict]) -> str:
    """Format conversation history for the judge prompt."""
    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"[{role}]: {msg['content']}")
    return "\n\n".join(parts)


@click.group()
def main():
    """WildBench evaluation — 1,024 real user tasks."""
    pass


@main.command()
@click.option("--model", default="local", help="Model name for the API")
@click.option("--gateway-url", default="http://localhost:4000/v1", help="API base URL")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--max-tokens", default=4096, help="Max tokens per response")
@click.option("--limit", default=None, type=int, help="Limit number of tasks (for testing)")
@click.option("--output", default=None, help="Output file path (default: auto)")
def infer(model, gateway_url, api_key, max_tokens, limit, output):
    """Phase 1: Generate model responses for WildBench tasks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=gateway_url, api_key=api_key)
    ds = load_wildbench()

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out_path = Path(output) if output else RESULTS_DIR / f"{model}.json"
    results = []
    total = len(ds)

    click.echo(f"Model:  {model}")
    click.echo(f"Tasks:  {total}")
    click.echo(f"Output: {out_path}")
    click.echo()

    for i, item in enumerate(ds):
        session_id = item["session_id"]
        messages = item["conversation_input"]
        primary_tag = item.get("primary_tag", "unknown")

        start = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            output_text = resp.choices[0].message.content or ""
            tokens = resp.usage.completion_tokens if resp.usage else 0
            error = None
        except Exception as e:
            output_text = ""
            tokens = 0
            error = str(e)

        elapsed = time.monotonic() - start

        results.append({
            "session_id": session_id,
            "primary_tag": primary_tag,
            "conversation_input": messages,
            "checklist": item.get("checklist", []),
            "output": output_text,
            "tokens": tokens,
            "time_s": round(elapsed, 2),
            "error": error,
            "model": model,
        })

        status = "OK" if not error else f"ERR: {error[:50]}"
        click.echo(f"  [{i + 1}/{total}] {session_id[:12]}... {primary_tag:30s} {tokens:>5d} tok  {elapsed:.1f}s  {status}")

        # Save incrementally every 50 tasks
        if (i + 1) % 50 == 0 or i == total - 1:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    click.echo(f"\nDone. {len(results)} responses saved to {out_path}")


@main.command()
@click.option("--results", required=True, type=click.Path(exists=True), help="Inference results JSON")
@click.option("--judge", default="gpt-5.4", help="Judge model name")
@click.option("--gateway-url", default="http://localhost:4000/v1", help="Gateway URL for judge")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--limit", default=None, type=int, help="Limit tasks to judge")
def judge(results, judge, gateway_url, api_key, limit):
    """Phase 2: Score responses using a judge model."""
    client = OpenAI(base_url=gateway_url, api_key=api_key)

    with open(results) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]

    model_name = data[0]["model"] if data else "unknown"
    out_path = Path(results).with_suffix(".scored.json")
    total = len(data)

    click.echo(f"Judge:   {judge}")
    click.echo(f"Model:   {model_name}")
    click.echo(f"Tasks:   {total}")
    click.echo(f"Output:  {out_path}")
    click.echo()

    for i, item in enumerate(data):
        if item.get("error"):
            item["judge_score"] = 0
            item["judge_reasoning"] = f"Inference error: {item['error']}"
            continue

        conversation = format_conversation(item["conversation_input"])
        checklist = "\n".join(f"- {c}" for c in item.get("checklist", []))

        prompt = JUDGE_TEMPLATE.format(
            conversation=conversation,
            response=item["output"],
            checklist=checklist or "No specific checklist for this task.",
        )

        try:
            resp = client.chat.completions.create(
                model=judge,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
            )
            judge_output = resp.choices[0].message.content or ""

            # Parse JSON from judge
            # Handle markdown code fences
            cleaned = judge_output.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            score = int(parsed.get("score", 0))
            reasoning = f"Strengths: {parsed.get('strengths', '')} | Weaknesses: {parsed.get('weaknesses', '')}"
        except json.JSONDecodeError:
            score = 0
            reasoning = f"Judge output not valid JSON: {judge_output[:100]}"
        except Exception as e:
            score = 0
            reasoning = f"Judge error: {str(e)[:100]}"

        item["judge_score"] = score
        item["judge_reasoning"] = reasoning
        item["judge_model"] = judge

        click.echo(f"  [{i + 1}/{total}] {item['session_id'][:12]}... score={score:>2d}  {item['primary_tag'][:25]}")

        # Save incrementally
        if (i + 1) % 50 == 0 or i == total - 1:
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)

    # Compute summary
    scored = [d for d in data if d.get("judge_score", 0) > 0]
    if scored:
        avg_raw = sum(d["judge_score"] for d in scored) / len(scored)
        wb_score = (avg_raw - 5) * 2  # WildBench rescaling

        # Per-category breakdown
        categories = {}
        for d in scored:
            tag = d.get("primary_tag", "unknown")
            categories.setdefault(tag, []).append(d["judge_score"])

        click.echo(f"\n{'=' * 60}")
        click.echo(f"WB-Score Results: {model_name} (judged by {judge})")
        click.echo(f"{'=' * 60}")
        click.echo(f"  Tasks scored: {len(scored)}/{total}")
        click.echo(f"  Avg raw score: {avg_raw:.2f}/10")
        click.echo(f"  WB-Score: {wb_score:+.2f}")
        click.echo()
        click.echo(f"  {'Category':<35s} {'Avg':>5s} {'Count':>5s}")
        click.echo(f"  {'-' * 47}")
        for tag in sorted(categories.keys()):
            scores = categories[tag]
            avg = sum(scores) / len(scores)
            click.echo(f"  {tag:<35s} {avg:>5.1f} {len(scores):>5d}")

    click.echo(f"\nScored results saved to {out_path}")


@main.command()
@click.option("--model", required=True, help="Model name for the API")
@click.option("--judge", default="gpt-5.4", help="Judge model")
@click.option("--gateway-url", default="http://localhost:4000/v1")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--max-tokens", default=4096)
@click.option("--limit", default=None, type=int, help="Limit tasks")
def run(model, judge, gateway_url, api_key, max_tokens, limit):
    """Run both inference and judging in one step."""
    from click.testing import CliRunner

    runner = CliRunner()

    # Phase 1
    out_path = str(RESULTS_DIR / f"{model}.json")
    args = ["--model", model, "--gateway-url", gateway_url, "--api-key", api_key,
            "--max-tokens", str(max_tokens), "--output", out_path]
    if limit:
        args += ["--limit", str(limit)]
    result = runner.invoke(infer, args, catch_exceptions=False)
    click.echo(result.output)

    # Phase 2
    args = ["--results", out_path, "--judge", judge, "--gateway-url", gateway_url,
            "--api-key", api_key]
    if limit:
        args += ["--limit", str(limit)]
    result = runner.invoke(judge, args, catch_exceptions=False)
    click.echo(result.output)


@main.command()
@click.option("--results", required=True, type=click.Path(exists=True), help="Scored results JSON")
def show(results):
    """Show WB-Score results from a scored JSON file."""
    with open(results) as f:
        data = json.load(f)

    scored = [d for d in data if d.get("judge_score", 0) > 0]
    if not scored:
        click.echo("No scored results found.")
        return

    model = scored[0].get("model", "unknown")
    judge_model = scored[0].get("judge_model", "unknown")
    avg_raw = sum(d["judge_score"] for d in scored) / len(scored)
    wb_score = (avg_raw - 5) * 2

    categories = {}
    for d in scored:
        tag = d.get("primary_tag", "unknown")
        categories.setdefault(tag, []).append(d["judge_score"])

    click.echo(f"\nWB-Score: {model} (judged by {judge_model})")
    click.echo(f"{'=' * 55}")
    click.echo(f"  Tasks: {len(scored)}/{len(data)}")
    click.echo(f"  Avg raw: {avg_raw:.2f}/10")
    click.echo(f"  WB-Score: {wb_score:+.2f}")
    click.echo()

    # Macro categories (WildBench grouping)
    macro = {
        "Creative": ["Creative Writing", "Editing", "Brainstorming", "Role playing"],
        "Planning & Reasoning": ["Reasoning", "Planning"],
        "Math & Data": ["Math", "Data Analysis"],
        "Info & Advice": ["Information seeking", "Advice seeking"],
        "Coding": ["Coding & Debugging"],
    }

    click.echo(f"  {'Macro Category':<25s} {'Avg':>5s} {'Count':>5s}")
    click.echo(f"  {'-' * 37}")
    for macro_name, tags in macro.items():
        macro_scores = []
        for tag in tags:
            macro_scores.extend(categories.get(tag, []))
        if macro_scores:
            avg = sum(macro_scores) / len(macro_scores)
            click.echo(f"  {macro_name:<25s} {avg:>5.1f} {len(macro_scores):>5d}")

    click.echo()
    click.echo(f"  {'Primary Tag':<35s} {'Avg':>5s} {'Count':>5s}")
    click.echo(f"  {'-' * 47}")
    for tag in sorted(categories.keys()):
        scores = categories[tag]
        avg = sum(scores) / len(scores)
        click.echo(f"  {tag:<35s} {avg:>5.1f} {len(scores):>5d}")


if __name__ == "__main__":
    main()
