"""Multi-Prompt APO — Optimize subagent system prompts.

Extends the APO loop to optimize protoResearcher's subagent prompts
(Explorer, Analyst, Writer) in addition to SOUL.md. Each subagent prompt
is optimized against tasks that exercise its specific capabilities.

Usage:
    cd ~/dev/lab
    export GATEWAY_API_KEY=<from-infisical>

    # Optimize a specific subagent prompt
    .venv/bin/python experiments/agent-lightning/run_multi_apo.py --target explorer
    .venv/bin/python experiments/agent-lightning/run_multi_apo.py --target analyst
    .venv/bin/python experiments/agent-lightning/run_multi_apo.py --target writer

    # Dry run to evaluate current prompts
    .venv/bin/python experiments/agent-lightning/run_multi_apo.py --target explorer --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import llm_judge

# --- Config ---
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://ava:4000/v1")
DEFAULT_CRITIC_MODEL = "claude-sonnet-4-6"
RESEARCHER_URL = "http://localhost:7872/api/chat"
TASKS_PATH = Path("/home/ava/dev/protoResearcher/evals/tasks.json")
SUBAGENT_CONFIG = Path("/home/ava/dev/protoResearcher/graph/subagents/config.py")
RESULTS_DIR = Path(__file__).parent / "results" / "multi-prompt"

# Task → subagent mapping: which tasks exercise which subagent
SUBAGENT_TASKS = {
    "explorer": [
        "discord_scan",       # Uses discord_feed (Explorer tool)
        "hf_model_search",    # Uses huggingface (Explorer tool)
        "github_trending",    # Uses github_trending (Explorer tool)
    ],
    "analyst": [
        "paper_analysis",           # Uses paper_reader (Analyst tool)
        "multi_step_scan_analyze",  # Multi-step, Analyst does the analysis part
    ],
    "writer": [
        "digest_generation",     # Uses research_memory (Writer tool)
        "complex_multi_tool",    # Complex synthesis → Writer produces the final brief
    ],
}

GRADIENT_PROMPT = """\
You are an expert prompt engineer optimizing a subagent system prompt.

## Subagent Role
This is the **{subagent_name}** subagent of a multi-agent research assistant.
Its role: {subagent_description}

## Current System Prompt
{current_prompt}

## Task Results
The agent was tested on tasks that exercise this subagent's capabilities:

{task_results}

## Instructions
Analyze what went wrong or could be improved. Write a concise critique (3-5 bullets) \
focusing on specific, actionable changes to improve task performance. \
Focus on: workflow clarity, tool usage patterns, output structure, error handling, \
and whether the rules effectively guide behavior.

Critique:"""

EDIT_PROMPT = """\
You are an expert prompt engineer. Apply the critique to improve a subagent system prompt.

## Current System Prompt
{current_prompt}

## Critique
{critique}

## Instructions
Rewrite the COMPLETE system prompt incorporating the critique. \
Keep the core role and tool allowlist intact — modify workflow, rules, \
and behavioral guidance based on the critique. \
Output ONLY the new system prompt, nothing else."""


def load_tasks(filter_ids: list[str]) -> list[dict]:
    tasks = json.loads(TASKS_PATH.read_text())
    return [t for t in tasks if t["id"] in filter_ids]


def read_subagent_prompt(target: str) -> str:
    """Read the current subagent prompt from config.py."""
    content = SUBAGENT_CONFIG.read_text()
    # Parse the system_prompt string from the config
    config_name = f"{target.upper()}_CONFIG"
    idx = content.find(config_name)
    if idx == -1:
        raise ValueError(f"Config {config_name} not found")

    # Find system_prompt= within this config block
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


async def evaluate_prompt(
    target: str, prompt_content: str, tasks: list[dict], label: str,
    judge_client,
) -> tuple[float, list[dict]]:
    """Write subagent prompt, run tasks, judge, return (avg_score, results)."""
    write_subagent_prompt(target, prompt_content)
    await asyncio.sleep(2)  # Let bind mount propagate

    results = []
    for task in tasks:
        sid = f"mapo-{target}-{label}-{task['id']}-{int(time.time())}"
        response = await run_task(task["prompt"], sid)
        score = await llm_judge.judge_response(task, response, client=judge_client)

        results.append({
            "task_id": task["id"],
            "prompt": task["prompt"][:80],
            "score": score.composite,
            "judge_detail": score.to_dict(),
            "response_preview": response[:300],
        })
        print(f"    {task['id']:30s} composite={score.composite:.3f}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    return avg, results


async def generate_critique(
    client: AsyncOpenAI, model: str,
    target: str, description: str,
    current_prompt: str, task_results: list[dict],
) -> str:
    results_text = "\n".join(
        f"- Task: {r['prompt']}\n  Score: {r['score']:.3f}\n  Response: {r['response_preview']}"
        for r in task_results
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": GRADIENT_PROMPT.format(
                subagent_name=target,
                subagent_description=description,
                current_prompt=current_prompt,
                task_results=results_text,
            ),
        }],
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


async def apply_edit(
    client: AsyncOpenAI, model: str,
    current_prompt: str, critique: str,
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": EDIT_PROMPT.format(
                current_prompt=current_prompt, critique=critique,
            ),
        }],
        max_tokens=4096,
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


SUBAGENT_DESCRIPTIONS = {
    "explorer": "Scans research sources (Discord, HuggingFace, GitHub, web) to discover papers, models, and trends.",
    "analyst": "Reads papers deeply, extracts findings, rates significance, stores to knowledge base.",
    "writer": "Synthesizes research findings into clear, actionable digests and publishes to Discord.",
}


async def run_multi_apo(args):
    target = args.target
    task_ids = SUBAGENT_TASKS[target]
    tasks = load_tasks(task_ids)

    if not tasks:
        print(f"No tasks found for target '{target}' with IDs: {task_ids}")
        return

    # Split train/val
    if len(tasks) >= 3:
        train_tasks = tasks[:2]
        val_tasks = tasks[2:]
    else:
        train_tasks = tasks
        val_tasks = tasks

    print(f"Target: {target} subagent")
    print(f"Tasks: {len(tasks)} total, {len(train_tasks)} train, {len(val_tasks)} val")
    print(f"Critic: {args.critic_model}, Judge: {llm_judge.JUDGE_MODEL}")
    print()

    gw_key = os.environ.get("GATEWAY_API_KEY", "not-needed")
    critic_client = AsyncOpenAI(base_url=GATEWAY_URL, api_key=gw_key)
    judge_client = llm_judge._make_client()

    seed_prompt = read_subagent_prompt(target)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save original
    (RESULTS_DIR / f"{target}_original.md").write_text(seed_prompt)

    # Evaluate seed
    print("=== Evaluating seed prompt ===")
    seed_score, seed_results = await evaluate_prompt(
        target, seed_prompt, val_tasks, "seed", judge_client,
    )
    print(f"  Seed score: {seed_score:.3f}\n")

    if args.dry_run:
        return

    beam = [{"prompt": seed_prompt, "score": seed_score, "version": "seed"}]
    best = {"prompt": seed_prompt, "score": seed_score, "version": "seed"}
    log = [{"round": 0, "version": "seed", "score": seed_score}]

    for rnd in range(1, args.rounds + 1):
        print(f"=== Round {rnd}/{args.rounds} ===")
        new_candidates = []

        for parent in beam:
            print(f"  Training rollout for '{parent['version']}'...")
            _, train_results = await evaluate_prompt(
                target, parent["prompt"], train_tasks, f"r{rnd}-train", judge_client,
            )

            for b in range(args.branch_factor):
                print(f"  Generating critique {b+1}/{args.branch_factor}...")
                critique = await generate_critique(
                    critic_client, args.critic_model,
                    target, SUBAGENT_DESCRIPTIONS[target],
                    parent["prompt"], train_results,
                )

                print(f"  Applying edit...")
                candidate = await apply_edit(
                    critic_client, args.critic_model,
                    parent["prompt"], critique,
                )

                if not candidate or len(candidate) < 50:
                    print(f"  [skip] Too short ({len(candidate or '')} chars)")
                    continue

                version = f"r{rnd}-p{beam.index(parent)}-b{b}"
                new_candidates.append({
                    "prompt": candidate,
                    "score": None,
                    "version": version,
                    "critique": critique[:200],
                })

        all_candidates = beam + new_candidates
        print(f"\n  Evaluating {len(all_candidates)} candidates...")
        for c in all_candidates:
            if c["score"] is None:
                print(f"  --- {c['version']} ---")
                score, _ = await evaluate_prompt(
                    target, c["prompt"], val_tasks, c["version"], judge_client,
                )
                c["score"] = score
                print(f"  {c['version']} score: {score:.3f}")

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = all_candidates[:args.beam_width]

        print(f"\n  Beam after round {rnd}:")
        for c in beam:
            print(f"    {c['version']:20s} score={c['score']:.3f}")
            log.append({"round": rnd, "version": c["version"], "score": c["score"]})

        if beam[0]["score"] > best["score"]:
            best = beam[0]
            print(f"  New best: {best['version']} ({best['score']:.3f})")
        print()

    # Save results
    print("=" * 60)
    print(f"  BEST: {best['version']} — score {best['score']:.3f} (seed was {seed_score:.3f})")
    print(f"  Delta: {best['score'] - seed_score:+.3f}")
    print("=" * 60)

    (RESULTS_DIR / f"{target}_best.md").write_text(best["prompt"])
    (RESULTS_DIR / f"{target}_log.json").write_text(json.dumps(log, indent=2))

    # Write best prompt back to config
    write_subagent_prompt(target, best["prompt"])
    print(f"\nBest prompt written to config.py and saved to {RESULTS_DIR / f'{target}_best.md'}")


def main():
    parser = argparse.ArgumentParser(description="Multi-prompt APO for subagents")
    parser.add_argument("--target", required=True, choices=["explorer", "analyst", "writer"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_multi_apo(args))


if __name__ == "__main__":
    main()
