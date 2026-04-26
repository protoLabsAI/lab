"""Run a named evaluation profile (quick or full).

Orchestrates claw-eval, custom suites, and function-call runners from a
single YAML profile config.

Usage:
    python -m runners.run_profile --name quick --model local
    python -m runners.run_profile --name full --model local
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

PROFILES_DIR = Path(__file__).parent.parent / "profiles"
EVALS_DIR = Path(__file__).parent.parent


def load_profile(name: str) -> dict:
    """Load a profile YAML by name."""
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in PROFILES_DIR.glob("*.yaml")]
        raise click.ClickException(
            f"Profile '{name}' not found. Available: {', '.join(available)}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def run_command(cmd: list[str], label: str) -> tuple[bool, float]:
    """Run a subprocess, stream output, return (success, duration_s)."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {label}")
    click.echo(f"{'=' * 60}\n")
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=str(EVALS_DIR))
    duration = time.monotonic() - start
    success = result.returncode == 0
    status = click.style("PASS", fg="green") if success else click.style("FAIL", fg="red")
    click.echo(f"\n  [{status}] {label} ({duration:.0f}s)")
    return success, duration


@click.command()
@click.option("--name", required=True, help="Profile name (quick, full)")
@click.option("--model", default="local", help="Gateway model name")
@click.option("--gateway-url", default="http://ava:4000/v1")
@click.option("--api-key", envvar="GATEWAY_API_KEY", default="not-needed")
@click.option("--trials", default=None, type=int, help="Override profile trial count")
@click.option("--claw-only", is_flag=True, help="Run only claw-eval tasks")
@click.option("--custom-only", is_flag=True, help="Run only custom suites")
@click.option("--skip-claw", is_flag=True, help="Skip claw-eval tasks")
@click.option("--skip-fc", is_flag=True, help="Skip function-call tests")
def main(name, model, gateway_url, api_key, trials, claw_only, custom_only, skip_claw, skip_fc):
    """Run a named evaluation profile."""
    profile = load_profile(name)
    trial_count = trials or profile.get("trials", 1)
    profile_name = profile.get("name", name)

    # Auto-use OPENROUTER_API_KEY when targeting OpenRouter
    if "openrouter" in gateway_url:
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            api_key = or_key

    click.echo(f"\nProfile: {profile_name}")
    click.echo(f"Model:   {model}")
    click.echo(f"Trials:  {trial_count}")

    results = []
    total_start = time.monotonic()

    # Determine what to run based on flags
    run_claw = not custom_only and not skip_claw
    run_custom = not claw_only
    run_fc = not claw_only and not skip_fc

    # Build the run.sh base command
    # We call run.sh so Infisical secrets get injected
    run_sh = str(EVALS_DIR / "run.sh")

    # --- Claw-eval ---
    if run_claw and "claw" in profile:
        claw = profile["claw"]
        tasks = claw.get("tasks", "T02,T04,T06,T08")
        port_offset = claw.get("port_offset", 200)
        cmd = [
            "bash", run_sh, "claw",
            "--model", model,
            "--tasks", tasks,
            "--trials", str(trial_count),
            "--port-offset", str(port_offset),
            "--gateway-url", gateway_url,
            "--api-key", api_key,
        ]
        success, dur = run_command(cmd, f"Claw-Eval ({tasks})")
        results.append(("claw-eval", success, dur))

    # --- Custom suites ---
    if run_custom and "custom" in profile:
        for suite_cfg in profile["custom"]:
            suite_name = suite_cfg["suite"]
            cmd = [
                "bash", run_sh, "custom",
                "--suite", suite_name,
                "--model", model,
                "--trials", str(trial_count),
                "--gateway-url", gateway_url,
                "--api-key", api_key,
            ]
            success, dur = run_command(cmd, f"Custom: {suite_name}")
            results.append((f"custom/{suite_name}", success, dur))

    # --- Function calling ---
    if run_fc and "function_call" in profile:
        fc = profile["function_call"]
        suites = fc.get("suites", [])
        if suites:
            cmd = [
                "bash", run_sh, "function-call",
                "--model", model,
                "--all-suites",
                "--gateway-url", gateway_url,
                "--api-key", api_key,
            ]
            success, dur = run_command(cmd, "Function Calling (all suites)")
            results.append(("function-call", success, dur))

    # --- Summary ---
    total_dur = time.monotonic() - total_start
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  PROFILE SUMMARY: {profile_name}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Model:    {model}")
    click.echo(f"  Trials:   {trial_count}")
    click.echo(f"  Duration: {total_dur:.0f}s ({total_dur/60:.1f}m)")
    click.echo()

    for name_str, success, dur in results:
        status = click.style("PASS", fg="green") if success else click.style("FAIL", fg="red")
        click.echo(f"  [{status}] {name_str:30s} ({dur:.0f}s)")

    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    click.echo(f"\n  Total: {passed}/{total} suite runs succeeded")
    click.echo()


if __name__ == "__main__":
    main()
