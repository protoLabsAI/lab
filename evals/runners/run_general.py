"""Run general LLM quality benchmarks via Inspect AI.

Convenience wrapper that runs a standard battery of benchmarks:
HumanEval, MMLU-Pro, GSM8K, BBH, ARC.

Usage:
    python -m runners.run_general --model local
    python -m runners.run_general --model local --bench humaneval
"""

from __future__ import annotations

import sys

import click

STANDARD_BATTERY = ["humaneval", "mmlu_pro", "gsm8k", "bbh", "arc"]


@click.command()
@click.option("--model", default="local", help="Gateway model name")
@click.option("--bench", default=None, help="Specific benchmark (default: run standard battery)")
@click.option("--gateway-url", default="http://ava:4000/v1")
@click.option("--api-key", envvar=["GATEWAY_API_KEY", "LITELLM_API_KEY"], default="not-needed")
@click.option("--limit", default=None, type=int, help="Limit samples per benchmark")
@click.option("--submit-langfuse", is_flag=True)
def main(model, bench, gateway_url, api_key, limit, submit_langfuse):
    """Run standard LLM quality benchmarks."""
    try:
        from runners.run_inspect import main as inspect_main
    except ImportError:
        click.echo("Error: inspect-ai required. Run: pip install inspect-ai")
        sys.exit(1)

    benchmarks = [bench] if bench else STANDARD_BATTERY
    click.echo(f"Model: {model}")
    click.echo(f"Benchmarks: {', '.join(benchmarks)}")
    click.echo("=" * 60)

    ctx = click.Context(inspect_main)
    for benchmark in benchmarks:
        click.echo(f"\n--- {benchmark} ---")
        try:
            ctx.invoke(
                inspect_main,
                benchmark=benchmark,
                model=model,
                gateway_url=gateway_url,
                api_key=api_key,
                limit=limit,
                list_benchmarks=False,
                submit_langfuse=submit_langfuse,
            )
        except SystemExit:
            click.echo(f"Warning: {benchmark} failed, continuing...")
            continue


if __name__ == "__main__":
    main()
