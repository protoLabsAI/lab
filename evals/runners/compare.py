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


def load_trace_results(result_dir: Path) -> dict:
    """Load results from claw-eval trace files.

    Reads grading_result events (post-hoc grading) when available,
    falls back to trace_end for timing/token data.
    """
    results = {}

    # Trace files can be nested in subdirectories
    trace_files = list(result_dir.rglob("*.jsonl"))
    if not trace_files:
        return results

    for trace_file in trace_files:
        trace_end = None
        grading_result = None

        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = entry.get("type", "")
                if event_type == "trace_end":
                    trace_end = entry
                elif event_type == "grading_result":
                    grading_result = entry

        if trace_end is None:
            continue

        # Derive task_id from trace_start or grading_result or filename
        task_id = None
        if grading_result:
            task_id = grading_result.get("task_id")
        if not task_id:
            # Fall back to filename: T02_email_triage_<uuid>.jsonl
            stem = trace_file.stem
            parts = stem.rsplit("_", 1)
            task_id = parts[0] if len(parts) > 1 else stem

        # Prefer grading_result scores over trace_end
        if grading_result:
            scores = grading_result.get("scores", {})
            result = {
                "task_score": grading_result.get("task_score", 0.0),
                "passed": grading_result.get("passed", False),
                "completion": scores.get("completion", 0.0),
                "robustness": scores.get("robustness", 0.0),
                "communication": scores.get("communication", 0.0),
                "safety": scores.get("safety", 1.0),
                "graded": True,
            }
        else:
            scores = trace_end.get("scores", {})
            result = {
                "task_score": trace_end.get("task_score", 0.0),
                "passed": trace_end.get("passed", False),
                "completion": scores.get("completion", 0.0),
                "robustness": scores.get("robustness", 0.0),
                "communication": scores.get("communication", 0.0),
                "safety": scores.get("safety", 1.0),
                "graded": False,
            }

        # Timing and token data always from trace_end
        gr_failures = grading_result.get("failure_modes", []) if grading_result else []
        result.update({
            "wall_time_s": trace_end.get("wall_time_s", 0),
            "total_tokens": trace_end.get("total_tokens", 0),
            "total_turns": trace_end.get("total_turns", 0),
            "failure_modes": gr_failures or trace_end.get("failure_modes", []),
        })

        # Group by task_id (multiple trials)
        if task_id not in results:
            results[task_id] = []
        results[task_id].append(result)

    return results


def aggregate_metrics(results: dict) -> dict:
    """Compute aggregate metrics from grouped trial results."""
    all_trials = []
    for trials in results.values():
        all_trials.extend(trials)

    total_trials = len(all_trials)
    total_tasks = len(results)
    graded = sum(1 for t in all_trials if t.get("graded"))

    if total_trials == 0:
        return {
            "tasks": 0, "trials": 0, "graded": 0, "passed": 0,
            "pass_rate": 0, "pass3": 0, "pass3_eligible": 0,
            "avg_task_score": 0, "avg_completion": 0,
            "avg_robustness": 0, "avg_communication": 0,
            "avg_tokens": 0, "avg_duration": 0,
        }

    passed = sum(1 for t in all_trials if t["passed"])
    avg_score = sum(t["task_score"] for t in all_trials) / total_trials
    avg_completion = sum(t["completion"] for t in all_trials) / total_trials
    avg_robustness = sum(t["robustness"] for t in all_trials) / total_trials
    avg_communication = sum(t["communication"] for t in all_trials) / total_trials
    avg_tokens = sum(t["total_tokens"] for t in all_trials) / total_trials
    avg_duration = sum(t["wall_time_s"] for t in all_trials) / total_trials

    # pass^3: task passes only if ALL trials pass
    pass3_tasks = sum(
        1 for trials in results.values()
        if len(trials) >= 3 and all(t["passed"] for t in trials[:3])
    )
    tasks_with_3_trials = sum(
        1 for trials in results.values() if len(trials) >= 3
    )

    return {
        "tasks": total_tasks,
        "trials": total_trials,
        "graded": graded,
        "passed": passed,
        "pass_rate": passed / total_trials if total_trials else 0,
        "pass3": pass3_tasks,
        "pass3_eligible": tasks_with_3_trials,
        "avg_task_score": avg_score,
        "avg_completion": avg_completion,
        "avg_robustness": avg_robustness,
        "avg_communication": avg_communication,
        "avg_tokens": avg_tokens,
        "avg_duration": avg_duration,
    }


@click.command()
@click.argument("result_dirs", nargs=-1, type=click.Path(exists=True))
@click.option("--from-langfuse", is_flag=True, help="Compare Langfuse experiments")
@click.option("--experiments", multiple=True, help="Langfuse experiment names")
@click.option("--per-task", is_flag=True, help="Show per-task breakdown")
def main(result_dirs, from_langfuse, experiments, per_task):
    """Compare eval results across models or experiment runs."""
    console = Console()

    if from_langfuse and experiments:
        console.print("[bold]Langfuse experiment comparison[/bold]")
        console.print("(Use Langfuse UI for interactive comparison)")
        console.print(f"Experiments: {', '.join(experiments)}")
        return

    if not result_dirs:
        console.print("Provide result directories or use --from-langfuse")
        sys.exit(1)

    all_results = {}
    all_metrics = {}
    for d in result_dirs:
        name = Path(d).name
        all_results[name] = load_trace_results(Path(d))
        all_metrics[name] = aggregate_metrics(all_results[name])

    models = list(all_results.keys())

    # Summary table
    table = Table(title="Model Comparison")
    table.add_column("Metric", style="bold")
    for m in models:
        table.add_column(m, justify="center")

    rows = [
        ("Tasks", lambda m: str(all_metrics[m]["tasks"])),
        ("Trials", lambda m: str(all_metrics[m]["trials"])),
        ("Graded", lambda m: str(all_metrics[m]["graded"])),
        ("Passed", lambda m: f"{all_metrics[m]['passed']}/{all_metrics[m]['trials']}"),
        ("Pass Rate", lambda m: f"{all_metrics[m]['pass_rate']:.1%}"),
        ("Pass^3", lambda m: (
            f"{all_metrics[m]['pass3']}/{all_metrics[m]['pass3_eligible']}"
            if all_metrics[m]["pass3_eligible"] else "—"
        )),
        ("Avg Task Score", lambda m: f"{all_metrics[m]['avg_task_score']:.3f}"),
        ("Avg Completion", lambda m: f"{all_metrics[m]['avg_completion']:.3f}"),
        ("Avg Robustness", lambda m: f"{all_metrics[m]['avg_robustness']:.3f}"),
        ("Avg Communication", lambda m: f"{all_metrics[m]['avg_communication']:.3f}"),
        ("Avg Tokens", lambda m: f"{all_metrics[m]['avg_tokens']:,.0f}"),
        ("Avg Duration", lambda m: f"{all_metrics[m]['avg_duration']:.1f}s"),
    ]

    for label, fn in rows:
        table.add_row(label, *[fn(m) for m in models])

    console.print(table)

    # Per-task breakdown
    if per_task:
        # Collect all task IDs
        all_task_ids = sorted(set(
            tid for r in all_results.values() for tid in r
        ))

        task_table = Table(title="Per-Task Results")
        task_table.add_column("Task", style="bold")
        for m in models:
            task_table.add_column(f"{m} score", justify="center")
            task_table.add_column(f"{m} pass", justify="center")

        for tid in all_task_ids:
            row = [tid]
            for m in models:
                trials = all_results[m].get(tid, [])
                if trials:
                    avg = sum(t["task_score"] for t in trials) / len(trials)
                    passed = sum(1 for t in trials if t["passed"])
                    row.append(f"{avg:.3f}")
                    style = "green" if passed == len(trials) else (
                        "yellow" if passed > 0 else "red"
                    )
                    row.append(f"[{style}]{passed}/{len(trials)}[/{style}]")
                else:
                    row.extend(["—", "—"])
            task_table.add_row(*row)

        console.print(task_table)


if __name__ == "__main__":
    main()
