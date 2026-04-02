"""Bayesian Combination Search — Find the best mix of subagent prompts.

Inspired by DSPy MIPROv2: independently-optimized subagent prompts may
interact. This uses Optuna's TPE sampler to search over combinations of
original vs optimized prompts for each subagent (explorer, analyst, writer)
and find the global optimum.

Usage:
    cd ~/dev/lab
    export GATEWAY_API_KEY=<from-infisical>

    # Dry run — list available prompt variants
    .venv/bin/python experiments/agent-lightning/search_combos.py --dry-run

    # Run 8 trials of Bayesian search
    .venv/bin/python experiments/agent-lightning/search_combos.py --trials 8

    # Fewer trials for quick test
    .venv/bin/python experiments/agent-lightning/search_combos.py --trials 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import optuna

# Allow importing llm_judge from the same directory
sys.path.insert(0, str(Path(__file__).parent))
import llm_judge

# --- Config ---
RESEARCHER_URL = "http://localhost:7872/api/chat"
TASKS_PATH = Path("/home/ava/dev/protoResearcher/evals/tasks.json")
SUBAGENT_CONFIG = Path("/home/ava/dev/protoResearcher/graph/subagents/config.py")
PROMPTS_DIR = Path(__file__).parent / "results" / "multi-prompt"
RESULTS_DIR = Path(__file__).parent / "results" / "combo_search"

SUBAGENTS = ["explorer", "analyst", "writer"]

SUBAGENT_TASKS = {
    "explorer": ["discord_scan", "hf_model_search", "github_trending"],
    "analyst": ["paper_analysis", "multi_step_scan_analyze"],
    "writer": ["digest_generation", "complex_multi_tool"],
}


def read_subagent_prompt(target: str) -> str:
    """Read the current subagent prompt from config.py."""
    content = SUBAGENT_CONFIG.read_text()
    config_name = f"{target.upper()}_CONFIG"
    idx = content.find(config_name)
    if idx == -1:
        raise ValueError(f"Config {config_name} not found")
    prompt_start = content.find('system_prompt="""', idx)
    if prompt_start == -1:
        raise ValueError(f"system_prompt not found in {config_name}")
    prompt_start += len('system_prompt="""')
    prompt_end = content.find('"""', prompt_start)
    return content[prompt_start:prompt_end]


def write_subagent_prompt(target: str, new_prompt: str):
    """Write an updated subagent prompt into config.py."""
    content = SUBAGENT_CONFIG.read_text()
    config_name = f"{target.upper()}_CONFIG"
    idx = content.find(config_name)
    prompt_start = content.find('system_prompt="""', idx)
    prompt_start += len('system_prompt="""')
    prompt_end = content.find('"""', prompt_start)
    new_content = content[:prompt_start] + new_prompt + content[prompt_end:]
    SUBAGENT_CONFIG.write_text(new_content)


def load_variants() -> dict[str, dict[str, str]]:
    """Load original and optimized prompt variants for each subagent.

    Returns: {subagent: {"original": prompt, "optimized": prompt}} where
    optimized falls back to config.py current prompt if _best.md missing.
    """
    variants = {}
    for sa in SUBAGENTS:
        original_path = PROMPTS_DIR / f"{sa}_original.md"
        optimized_path = PROMPTS_DIR / f"{sa}_best.md"

        # Original: from saved file, else from config.py
        if original_path.exists():
            original = original_path.read_text()
        else:
            original = read_subagent_prompt(sa)

        # Optimized: from saved file, else same as original (no variant)
        if optimized_path.exists():
            optimized = optimized_path.read_text()
        else:
            optimized = None

        v = {"original": original}
        if optimized and optimized.strip() != original.strip():
            v["optimized"] = optimized
        variants[sa] = v
    return variants


def load_tasks(filter_ids: list[str]) -> list[dict]:
    tasks = json.loads(TASKS_PATH.read_text())
    return [t for t in tasks if t["id"] in filter_ids]


async def run_task(prompt: str, session_id: str, timeout: float = 180) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(
                RESEARCHER_URL,
                json={"message": prompt, "session_id": session_id},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            return f"[error: {e}]"


async def eval_combination(
    combo: dict[str, str],
    variants: dict[str, dict[str, str]],
    all_tasks: dict[str, list[dict]],
    judge_client,
    trial_label: str,
) -> tuple[float, list[dict]]:
    """Write a combination of prompts, run all tasks, return composite score."""
    # Write chosen prompts to config.py
    for sa in SUBAGENTS:
        choice = combo[sa]
        write_subagent_prompt(sa, variants[sa][choice])
    await asyncio.sleep(2)  # Let bind mount propagate

    results = []
    for sa in SUBAGENTS:
        tasks = all_tasks.get(sa, [])
        for task in tasks:
            sid = f"combo-{trial_label}-{task['id']}-{int(time.time())}"
            response = await run_task(task["prompt"], sid)
            score = await llm_judge.judge_response(task, response, client=judge_client)
            results.append({
                "subagent": sa,
                "task_id": task["id"],
                "score": score.composite,
                "judge_detail": score.to_dict(),
                "response_preview": response[:300],
            })
            tag = f"{combo[sa][:3]}"
            print(f"    [{sa}:{tag}] {task['id']:30s} score={score.composite:.3f}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    return avg, results


def objective_factory(variants, all_tasks, judge_client, loop):
    """Return an Optuna objective function closed over shared state."""

    def objective(trial: optuna.Trial) -> float:
        combo = {}
        for sa in SUBAGENTS:
            choices = list(variants[sa].keys())
            if len(choices) == 1:
                combo[sa] = choices[0]
            else:
                combo[sa] = trial.suggest_categorical(sa, choices)

        label = f"t{trial.number}"
        desc = " | ".join(f"{sa}={combo[sa]}" for sa in SUBAGENTS)
        print(f"\n--- Trial {trial.number}: {desc} ---")

        score, results = loop.run_until_complete(
            eval_combination(combo, variants, all_tasks, judge_client, label)
        )

        print(f"  => composite={score:.3f}")
        trial.set_user_attr("combo", combo)
        trial.set_user_attr("results", results)
        return score

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian combination search over subagent prompt variants"
    )
    parser.add_argument("--trials", type=int, default=8, help="Number of Optuna trials")
    parser.add_argument("--dry-run", action="store_true", help="List variants, don't run")
    args = parser.parse_args()

    variants = load_variants()

    # --- Dry run: just show what we have ---
    if args.dry_run:
        print("Available prompt variants:\n")
        for sa in SUBAGENTS:
            choices = list(variants[sa].keys())
            print(f"  {sa}: {choices}")
            for c in choices:
                length = len(variants[sa][c])
                print(f"    {c}: {length} chars")
        total = 1
        for sa in SUBAGENTS:
            total *= len(variants[sa])
        print(f"\nTotal combinations: {total}")
        print(f"Requested trials: {args.trials}")
        return

    # --- Load tasks ---
    all_tasks: dict[str, list[dict]] = {}
    total_tasks = 0
    for sa in SUBAGENTS:
        tasks = load_tasks(SUBAGENT_TASKS[sa])
        all_tasks[sa] = tasks
        total_tasks += len(tasks)

    total_combos = 1
    for sa in SUBAGENTS:
        total_combos *= len(variants[sa])
    print(f"Combo search: {total_combos} possible combinations, {args.trials} trials")
    print(f"Tasks: {total_tasks} total across {len(SUBAGENTS)} subagents")
    print(f"Judge: {llm_judge.JUDGE_MODEL}\n")

    judge_client = llm_judge._make_client()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run Optuna search (TPE sampler is default) ---
    loop = asyncio.new_event_loop()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="subagent_combo_search",
    )
    study.optimize(
        objective_factory(variants, all_tasks, judge_client, loop),
        n_trials=args.trials,
    )
    loop.close()

    # --- Report results ---
    best = study.best_trial
    best_combo = best.user_attrs["combo"]

    print("\n" + "=" * 60)
    print(f"BEST COMBINATION (trial {best.number}, score {best.value:.3f}):")
    for sa in SUBAGENTS:
        print(f"  {sa}: {best_combo[sa]}")
    print("=" * 60)

    # Save full results
    trial_log = []
    for t in study.trials:
        trial_log.append({
            "number": t.number,
            "score": t.value,
            "combo": t.user_attrs.get("combo", {}),
            "results": t.user_attrs.get("results", []),
        })

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"search_{ts}.json"
    log_path.write_text(json.dumps({
        "best_trial": best.number,
        "best_score": best.value,
        "best_combo": best_combo,
        "n_trials": len(study.trials),
        "trials": trial_log,
    }, indent=2, default=str))
    print(f"\nResults saved to {log_path}")

    # Write best combination to config.py
    for sa in SUBAGENTS:
        choice = best_combo[sa]
        write_subagent_prompt(sa, variants[sa][choice])
    print("Best combination written to config.py")


if __name__ == "__main__":
    main()
