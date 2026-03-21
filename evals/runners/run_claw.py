"""Run claw-eval benchmark suite against the gateway.

Wraps claw-eval CLI with our gateway config, captures results,
and submits scores to Langfuse for cross-model comparison.

Usage:
    python -m runners.run_claw --model local --tasks T01,T02
    python -m runners.run_claw --model claude-sonnet-4-6 --all
    python -m runners.run_claw --compare local,claude-sonnet-4-6
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

CLAW_DIR = Path(__file__).parent.parent / "claw" / "claw-eval"
TASKS_DIR = CLAW_DIR / "tasks"
CONFIG_TEMPLATE = Path(__file__).parent.parent / "claw" / "config.yaml"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def resolve_task(task_id: str) -> str:
    """Resolve a short task ID (e.g. T02) to the full task.yaml path."""
    # Try exact match first
    exact = TASKS_DIR / task_id / "task.yaml"
    if exact.exists():
        return str(exact)
    # Try prefix match (T02 -> T02_email_triage)
    # Use underscore boundary to prevent T10 matching T100, T101, etc.
    matches = sorted(TASKS_DIR.glob(f"{task_id}_*"))
    if not matches:
        # Fallback: try without underscore for exact numeric matches like T100
        matches = sorted(TASKS_DIR.glob(f"{task_id}*"))
    if len(matches) == 1:
        task_yaml = matches[0] / "task.yaml"
        if task_yaml.exists():
            return str(task_yaml)
    elif len(matches) > 1:
        # Prefer English (non-zh) variant
        for m in matches:
            if "zh" not in m.name:
                task_yaml = m / "task.yaml"
                if task_yaml.exists():
                    return str(task_yaml)
        return str(matches[0] / "task.yaml")
    raise FileNotFoundError(f"No task found for '{task_id}'. Available: claw-eval list")


def build_claw_config(model: str, gateway_url: str, api_key: str) -> Path:
    """Generate a claw-eval config targeting a specific model via gateway."""
    with open(CONFIG_TEMPLATE) as f:
        config = yaml.safe_load(f)

    config["model"]["model_id"] = model
    config["model"]["base_url"] = gateway_url
    config["model"]["api_key"] = api_key

    run_dir = RESULTS_DIR / f"{model}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config["defaults"]["trace_dir"] = str(run_dir)

    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path


@click.command()
@click.option("--model", default="local", help="Gateway model name to evaluate")
@click.option("--tasks", default=None, help="Comma-separated task IDs (e.g., T01,T02)")
@click.option("--all-tasks", is_flag=True, help="Run all available tasks")
@click.option("--trials", default=3, help="Number of trials per task (default: 3 for pass^3)")
@click.option("--gateway-url", default="http://localhost:4000/v1", help="Gateway base URL")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--workers", default=4, help="Parallel workers for batch mode")
@click.option("--port-offset", default=0, help="Port offset for mock services (use different values for parallel runs)")
def main(model, tasks, all_tasks, trials, gateway_url, api_key, workers, port_offset):
    """Run claw-eval tasks against a gateway model."""
    if not CLAW_DIR.exists():
        click.echo("Error: claw-eval submodule not initialized. Run: git submodule update --init")
        sys.exit(1)

    config_path = build_claw_config(model, gateway_url, api_key)
    click.echo(f"Config: {config_path}")
    click.echo(f"Model: {model} via {gateway_url}")
    click.echo(f"Trials: {trials}")

    # Build claw-eval command
    cmd = ["claw-eval"]

    if tasks:
        # Run specific tasks
        for task_id in tasks.split(","):
            task_path = resolve_task(task_id.strip())
            task_cmd = cmd + ["run", "--task", task_path, "--trials", str(trials), "--config", str(config_path), "--port-offset", str(port_offset)]
            click.echo(f"\n{'='*60}")
            click.echo(f"Running {task_id.strip()} ({Path(task_path).parent.name})...")
            subprocess.run(task_cmd, cwd=str(CLAW_DIR))
    elif all_tasks:
        # Batch mode
        batch_cmd = cmd + [
            "batch",
            "--trials", str(trials),
            "--config", str(config_path),
            "--parallel", str(workers),
            "--port-base-offset", str(port_offset),
        ]
        click.echo(f"\nRunning batch eval with {workers} workers...")
        subprocess.run(batch_cmd, cwd=str(CLAW_DIR))
    else:
        click.echo("Specify --tasks T01,T02 or --all-tasks")
        sys.exit(1)

    click.echo(f"\nResults saved to: {config_path.parent}")


if __name__ == "__main__":
    main()
