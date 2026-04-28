"""Push eval results to HuggingFace dataset for leaderboard tracking.

Reads profile_results.json (and child result files) from a completed eval
run and pushes a flat row to protoLabsAI/eval-results on HuggingFace.

Usage:
    python -m runners.push_results --results-dir results/local_quick_20260428_120000
    python -m runners.push_results --results-dir results/local_quick_20260428_120000 --dry-run
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
from huggingface_hub import HfApi

REPO_ID = "protoLabsAI/eval-results"
EVALS_DIR = Path(__file__).parent.parent


def build_result_row(results_dir: Path, model_meta: dict | None = None) -> dict:
    """Build a flat result row from a completed eval run directory."""
    profile_json = results_dir / "profile_results.json"
    if not profile_json.exists():
        raise FileNotFoundError(f"No profile_results.json in {results_dir}")

    with open(profile_json) as f:
        data = json.load(f)

    meta = model_meta or {}
    summary = data.get("summary", {})

    row: dict = {
        # Identity
        "run_id": results_dir.name,
        "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        # Model
        "model_id": meta.get("model_id", ""),
        "model_alias": data.get("model", ""),
        "quantization": meta.get("quantization", ""),
        "parameters_b": meta.get("parameters_b"),
        "architecture": meta.get("architecture", ""),
        "active_parameters_b": meta.get("active_parameters_b"),
        "tp_size": meta.get("tp_size", 1),
        "gpu_config": meta.get("gpu_config", ""),
        "max_model_len": meta.get("max_model_len"),
        "extra_flags": meta.get("extra_flags", ""),
        # Eval config
        "eval_profile": data.get("profile", ""),
        "trials": data.get("trials", 1),
        "judge_model": "claude-sonnet-4-6",
        "lab_git_hash": data.get("lab_git_hash", ""),
        # Aggregate scores
        "suites_run": summary.get("suites_run", 0),
        "suites_passed": summary.get("suites_passed", 0),
        "custom_tasks_total": summary.get("custom_tasks_total", 0),
        "custom_tasks_passed": summary.get("custom_tasks_passed", 0),
        "custom_pass_rate": summary.get("custom_pass_rate"),
        "total_duration_s": summary.get("total_duration_s"),
    }

    # Per-suite scores from custom results
    custom_json = results_dir / "custom_results.json"
    if custom_json.exists():
        with open(custom_json) as f:
            custom_data = json.load(f)
        for suite_name, suite_data in custom_data.get("suites", {}).items():
            tasks = suite_data.get("tasks", [])
            if tasks:
                total = len(tasks)
                passed = sum(1 for t in tasks if t.get("all_passed"))
                col = f"score_{suite_name.replace('/', '_')}"
                row[col] = round(passed / total, 3) if total > 0 else None

    # Function-call results
    fc_json = results_dir / "function_call_results.json"
    if fc_json.exists():
        with open(fc_json) as f:
            fc_data = json.load(f)
        fc_summary = fc_data.get("summary", {})
        row["score_function_call"] = fc_summary.get("pass_rate")

    # Speed metrics (read from speed-test output if present)
    speed_json = results_dir / "speed_metrics.json"
    if speed_json.exists():
        with open(speed_json) as f:
            speed = json.load(f)
        row["decode_tok_s"] = speed.get("decode_tok_s")
        row["ttft_ms"] = speed.get("ttft_ms")
        row["tpot_ms"] = speed.get("tpot_ms")

    return row


def push_to_hf(row: dict, profile: str = "results") -> str:
    """Push a result row to the HuggingFace dataset repo."""
    api = HfApi()

    # Determine shard path by month
    ts = row.get("timestamp", "")
    month = ts[:7] if len(ts) >= 7 else datetime.now(timezone.utc).strftime("%Y-%m")
    shard_path = f"data/{profile}/{month}.parquet"

    # Download existing shard or start fresh
    try:
        local = api.hf_hub_download(REPO_ID, shard_path, repo_type="dataset")
        df = pd.read_parquet(local)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except Exception:
        df = pd.DataFrame([row])

    # Write to temp file and upload
    tmp = Path("/tmp") / f"eval_{profile}_{month.replace('-', '_')}.parquet"
    df.to_parquet(tmp, index=False)

    commit_msg = (
        f"Add {row.get('model_alias', 'unknown')} "
        f"{row.get('eval_profile', '')} results "
        f"({ts[:10]})"
    )
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=shard_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=commit_msg,
    )

    return f"https://huggingface.co/datasets/{REPO_ID}"


# Well-known model aliases → metadata
KNOWN_MODELS: dict[str, dict] = {
    "local": {
        "model_id": "Qwen/Qwen3.5-27B",
        "quantization": "FP8",
        "parameters_b": 27.0,
        "architecture": "dense",
        "active_parameters_b": 27.0,
        "tp_size": 1,
        "gpu_config": "1x RTX PRO 6000 Blackwell",
    },
    "local-voice": {
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "quantization": "FP8",
        "parameters_b": 35.0,
        "architecture": "moe",
        "active_parameters_b": 3.0,
        "tp_size": 1,
        "gpu_config": "1x RTX PRO 6000 Blackwell",
    },
}


@click.command()
@click.option(
    "--results-dir", required=True, type=click.Path(exists=True),
    help="Path to completed eval results directory",
)
@click.option("--model-id", default=None, help="Override model HF repo ID")
@click.option("--quantization", default=None, help="Override quantization (FP8, INT4, BF16)")
@click.option("--dry-run", is_flag=True, help="Print row without pushing to HF")
def main(results_dir, model_id, quantization, dry_run):
    """Push eval results to HuggingFace dataset."""
    results_path = Path(results_dir)

    # Load profile results to get model alias
    profile_json = results_path / "profile_results.json"
    if not profile_json.exists():
        raise click.ClickException(f"No profile_results.json in {results_path}")

    with open(profile_json) as f:
        profile_data = json.load(f)

    alias = profile_data.get("model", "")
    meta = dict(KNOWN_MODELS.get(alias, {}))

    # CLI overrides
    if model_id:
        meta["model_id"] = model_id
    if quantization:
        meta["quantization"] = quantization

    row = build_result_row(results_path, model_meta=meta)

    if dry_run:
        click.echo("Dry run — would push this row:\n")
        click.echo(json.dumps(row, indent=2, default=str))
        return

    click.echo(f"Pushing results for {alias} ({row.get('eval_profile', '')})...")
    url = push_to_hf(row, profile=row.get("eval_profile", "results"))
    click.echo(f"Done: {url}")


if __name__ == "__main__":
    main()
