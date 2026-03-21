"""Compare eval results across models.

Usage:
    python -m runners.compare results/local_* results/claude-sonnet-4-6_*
    python -m runners.compare --from-langfuse --experiment "baseline-v1" "opus-27b-v1"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from graders.langfuse_scorer import LangfuseScorer


def load_trace_results(result_dir: Path) -> dict:
    """Load results from claw-eval trace files."""
    results = {}
    for trace_file in result_dir.glob("*.jsonl"):
        task_id = trace_file.stem
        entries = []
        with open(trace_file) as f:
            for line in f:
                entries.append(json.loads(line))
        # Extract final scores from TraceEnd events
        for entry in entries:
            if entry.get("event") == "TraceEnd":
                results[task_id] = {
                    "score": entry.get("score", 0.0),
                    "passed": entry.get("passed", False),
                    "duration_s": entry.get("wall_time_s", 0),
                    "tokens": entry.get("total_tokens", 0),
                }
    return results


@click.command()
@click.argument("result_dirs", nargs=-1, type=click.Path(exists=True))
@click.option("--from-langfuse", is_flag=True, help="Compare Langfuse experiments instead")
@click.option("--experiments", multiple=True, help="Langfuse experiment names to compare")
def main(result_dirs, from_langfuse, experiments):
    """Compare eval results across models or experiment runs."""
    console = Console()

    if from_langfuse and experiments:
        console.print("[bold]Langfuse experiment comparison[/bold]")
        console.print("(Use Langfuse UI at http://localhost:3001 for interactive comparison)")
        console.print(f"Experiments: {', '.join(experiments)}")
        return

    if not result_dirs:
        console.print("Provide result directories or use --from-langfuse --experiments")
        sys.exit(1)

    # Build comparison table
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="bold")
    for d in result_dirs:
        table.add_column(Path(d).name, justify="center")

    all_results = {}
    for d in result_dirs:
        all_results[Path(d).name] = load_trace_results(Path(d))

    # Aggregate metrics
    models = list(all_results.keys())
    metrics = {}
    for model, results in all_results.items():
        total = len(results)
        passed = sum(1 for r in results.values() if r.get("passed"))
        avg_score = sum(r.get("score", 0) for r in results.values()) / total if total else 0
        total_tokens = sum(r.get("tokens", 0) for r in results.values())
        avg_duration = sum(r.get("duration_s", 0) for r in results.values()) / total if total else 0
        metrics[model] = {
            "Tasks": total,
            "Passed": f"{passed}/{total}",
            "Pass Rate": f"{passed/total*100:.1f}%" if total else "N/A",
            "Avg Score": f"{avg_score:.3f}",
            "Total Tokens": f"{total_tokens:,}",
            "Avg Duration": f"{avg_duration:.1f}s",
        }

    for metric in ["Tasks", "Passed", "Pass Rate", "Avg Score", "Total Tokens", "Avg Duration"]:
        row = [metric] + [metrics[m].get(metric, "—") for m in models]
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
