"""Run a named evaluation profile (smoke, quick, or full).

Orchestrates claw-eval, custom suites, and function-call runners from a
single YAML profile config. Writes aggregated results to a timestamped
directory under results/.

Usage:
    python -m runners.run_profile --name smoke --model local
    python -m runners.run_profile --name quick --model local
    python -m runners.run_profile --name full --model local
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml

PROFILES_DIR = Path(__file__).parent.parent / "profiles"
EVALS_DIR = Path(__file__).parent.parent
RESULTS_DIR = EVALS_DIR / "results"


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


def collect_claw_results(results_dir: Path, model: str) -> dict | None:
    """Read claw-eval batch_summary.json if it exists."""
    # Claw-eval writes to results/<model>_<timestamp>/
    # Find the most recent matching directory
    candidates = sorted(results_dir.glob(f"{model}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in candidates:
        summary = d / "batch_summary.json"
        if summary.exists():
            with open(summary) as f:
                return json.load(f)
    return None


def aggregate_results(
    output_dir: Path,
    profile_name: str,
    model: str,
    trial_count: int,
    suite_results: list[tuple[str, bool, float]],
    timestamp: str,
) -> dict:
    """Read child runner JSON outputs and build a unified profile_results.json."""
    aggregated: dict = {
        "profile": profile_name,
        "model": model,
        "trials": trial_count,
        "timestamp": timestamp,
        "lab_git_hash": _get_git_hash(),
        "suites": {},
        "summary": {},
    }

    # Read custom suite results
    custom_json = output_dir / "custom_results.json"
    if custom_json.exists():
        with open(custom_json) as f:
            custom_data = json.load(f)
        for suite_name, suite_data in custom_data.get("suites", {}).items():
            aggregated["suites"][f"custom/{suite_name}"] = suite_data

    # Read function-call results
    fc_json = output_dir / "function_call_results.json"
    if fc_json.exists():
        with open(fc_json) as f:
            fc_data = json.load(f)
        aggregated["suites"]["function_call"] = fc_data.get("summary", {})
        aggregated["suites"]["function_call"]["tests"] = fc_data.get("tests", [])

    # Read claw-eval results (claw writes its own format)
    claw_summary = collect_claw_results(RESULTS_DIR, model)
    if claw_summary:
        aggregated["suites"]["claw-eval"] = claw_summary

    # Build top-level summary from subprocess pass/fail
    total_suites = len(suite_results)
    passed_suites = sum(1 for _, s, _ in suite_results if s)
    total_duration = sum(d for _, _, d in suite_results)

    # Compute aggregate custom scores if available
    custom_tasks_total = 0
    custom_tasks_passed = 0
    if custom_json.exists():
        with open(custom_json) as f:
            cd = json.load(f)
        custom_tasks_total = cd.get("summary", {}).get("total_tasks", 0)
        custom_tasks_passed = cd.get("summary", {}).get("passed_tasks", 0)

    aggregated["summary"] = {
        "suites_run": total_suites,
        "suites_passed": passed_suites,
        "total_duration_s": round(total_duration, 1),
        "custom_tasks_total": custom_tasks_total,
        "custom_tasks_passed": custom_tasks_passed,
        "custom_pass_rate": (
            round(custom_tasks_passed / custom_tasks_total, 3)
            if custom_tasks_total > 0
            else None
        ),
    }

    return aggregated


def _get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(EVALS_DIR),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


@click.command()
@click.option("--name", required=True, help="Profile name (smoke, quick, full)")
@click.option("--model", default="protolabs/smart", help="Gateway model name")
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Create output directory for this run
    output_dir = RESULTS_DIR / f"{model}_{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-use OPENROUTER_API_KEY when targeting OpenRouter
    if "openrouter" in gateway_url:
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if or_key:
            api_key = or_key

    click.echo(f"\nProfile: {profile_name}")
    click.echo(f"Model:   {model}")
    click.echo(f"Trials:  {trial_count}")
    click.echo(f"Output:  {output_dir}")

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

    # --- Custom suites (all suites write to a single output file) ---
    if run_custom and "custom" in profile:
        for suite_cfg in profile["custom"]:
            suite_name = suite_cfg["suite"]
            cmd = [
                "bash", run_sh, "custom",
                "--suite", suite_name,
                "--model", model,
                "--trials", str(trial_count),
                "--output-dir", str(output_dir),
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
                "--trials", str(trial_count),
                "--output-dir", str(output_dir),
                "--gateway-url", gateway_url,
                "--api-key", api_key,
            ]
            success, dur = run_command(cmd, "Function Calling (all suites)")
            results.append(("function-call", success, dur))

    # --- Aggregate & write results ---
    total_dur = time.monotonic() - total_start

    profile_results = aggregate_results(
        output_dir, profile_name, model, trial_count, results,
        datetime.now(timezone.utc).isoformat(),
    )
    profile_results["summary"]["total_duration_s"] = round(total_dur, 1)

    results_file = output_dir / "profile_results.json"
    with open(results_file, "w") as f:
        json.dump(profile_results, f, indent=2)

    # --- Print summary ---
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  PROFILE SUMMARY: {profile_name}")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Model:    {model}")
    click.echo(f"  Trials:   {trial_count}")
    click.echo(f"  Duration: {total_dur:.0f}s ({total_dur/60:.1f}m)")
    click.echo(f"  Output:   {output_dir}")
    click.echo()

    for name_str, success, dur in results:
        status = click.style("PASS", fg="green") if success else click.style("FAIL", fg="red")
        click.echo(f"  [{status}] {name_str:30s} ({dur:.0f}s)")

    # Print custom suite scores if available
    custom_summary = profile_results.get("summary", {})
    ct = custom_summary.get("custom_tasks_total", 0)
    cp = custom_summary.get("custom_tasks_passed", 0)
    if ct > 0:
        rate = custom_summary.get("custom_pass_rate", 0) or 0
        click.echo(f"\n  Custom:  {cp}/{ct} tasks passed ({rate:.0%})")

    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    click.echo(f"\n  Total: {passed}/{total} suite runs succeeded")
    click.echo(f"  Results: {results_file}")
    click.echo()


if __name__ == "__main__":
    main()
