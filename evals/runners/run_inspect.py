"""Run Inspect AI benchmarks against the gateway.

Thin wrapper around `inspect eval` that configures our gateway as the
model provider and submits scores to Langfuse.

Inspect AI bundles dozens of established benchmarks:
  GAIA, SWE-bench, BrowseComp, Cybench, HumanEval, MMLU-Pro, GSM8K, etc.

Usage:
    python -m runners.run_inspect --benchmark gaia --model local
    python -m runners.run_inspect --benchmark humaneval --model claude-sonnet-4-6
    python -m runners.run_inspect --list
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

RESULTS_DIR = Path(__file__).parent.parent / "results" / "inspect"


def get_available_benchmarks() -> list[str]:
    """Well-known Inspect AI benchmarks."""
    return [
        # Coding
        "humaneval", "swe_bench", "swe_bench_verified", "mbpp",
        # Reasoning
        "mmlu_pro", "gsm8k", "math", "bbh", "arc",
        # Agents
        "gaia", "browsecomp", "cybench",
        # Knowledge
        "gpqa", "triviaqa",
    ]


@click.command()
@click.option("--benchmark", default=None, help="Inspect benchmark name (e.g., gaia, humaneval)")
@click.option("--model", default="local", help="Gateway model name")
@click.option("--gateway-url", default="http://ava:4000/v1")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--limit", default=None, type=int, help="Limit number of samples")
@click.option("--list", "list_benchmarks", is_flag=True, help="List available benchmarks")
@click.option("--submit-langfuse", is_flag=True, help="Submit scores to Langfuse")
def main(benchmark, model, gateway_url, api_key, limit, list_benchmarks, submit_langfuse):
    """Run Inspect AI benchmarks against a gateway model."""
    if list_benchmarks:
        click.echo("Available benchmarks:")
        for b in get_available_benchmarks():
            click.echo(f"  {b}")
        return

    if not benchmark:
        click.echo("Specify --benchmark <name> or --list")
        sys.exit(1)

    # Check inspect is installed
    try:
        subprocess.run(["inspect", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        click.echo("Error: inspect-ai not installed. Run: pip install inspect-ai")
        sys.exit(1)

    # Build output directory
    run_dir = RESULTS_DIR / f"{benchmark}_{model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Benchmark: {benchmark}")
    click.echo(f"Model: {model} via {gateway_url}")
    click.echo(f"Results: {run_dir}")

    # Build inspect eval command
    # Inspect uses openai/ prefix for OpenAI-compatible endpoints
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = gateway_url

    cmd = [
        "inspect", "eval",
        f"inspect_evals/{benchmark}",
        "--model", f"openai/{model}",
        "--log-dir", str(run_dir),
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    click.echo(f"\nRunning: {' '.join(cmd)}")
    click.echo("=" * 60)

    result = subprocess.run(cmd, env=env, cwd=str(run_dir))

    if result.returncode != 0:
        click.echo(f"\nInspect eval failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Parse results and optionally submit to Langfuse
    log_files = sorted(run_dir.glob("*.json"))
    if log_files and submit_langfuse:
        try:
            from graders.langfuse_scorer import LangfuseScorer
            from graders.base import GradeResult, TaskResult

            scorer = LangfuseScorer()
            for log_file in log_files:
                with open(log_file) as f:
                    log_data = json.load(f)
                # Extract scores from Inspect log format
                if "results" in log_data:
                    metrics = log_data["results"].get("scores", [{}])
                    for metric in metrics:
                        for metric_name, metric_val in metric.get("metrics", {}).items():
                            click.echo(f"  {metric_name}: {metric_val.get('value', 'N/A')}")
        except ImportError:
            click.echo("Warning: langfuse not available for score submission")

    click.echo(f"\nResults saved to: {run_dir}")
    click.echo(f"View logs: inspect view --log-dir {run_dir}")


if __name__ == "__main__":
    main()
