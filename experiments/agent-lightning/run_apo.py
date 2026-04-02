"""APO (Automatic Prompt Optimization) for protoResearcher's SOUL.md.

Implements textual gradient beam search directly against our local vLLM,
bypassing Agent Lightning's internal plumbing (designed for its proxy/trace
pipeline, not external agent APIs).

Algorithm:
  1. Evaluate seed prompt on validation tasks
  2. For each beam round:
     a. Sample rollout results for gradient computation (critiques)
     b. Generate candidate prompts by applying textual gradient edits
     c. Evaluate candidates on validation tasks
     d. Keep top-k (beam width)
  3. Save best prompt

Usage:
    cd ~/dev/lab
    .venv/bin/python experiments/agent-lightning/run_apo.py
    .venv/bin/python experiments/agent-lightning/run_apo.py --rounds 3 --dry-run
    .venv/bin/python experiments/agent-lightning/run_apo.py --use-llm-judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import llm_judge
import bootstrap_fewshot

# --- Config ---
VLLM_URL = "http://localhost:8000/v1"
GATEWAY_URL = "http://100.101.189.45:4000/v1"
DEFAULT_CRITIC_MODEL = "cli/claude-sonnet-4-6"
RESEARCHER_URL = "http://localhost:7872/api/chat"
SOUL_PATH = Path("/home/ava/dev/protoResearcher/config/SOUL.md")
TASKS_PATH = Path("/home/ava/dev/protoResearcher/evals/tasks.json")
RESULTS_DIR = Path(__file__).parent / "results"

GRADIENT_PROMPT = """\
You are an expert prompt engineer optimizing a system prompt for an AI research assistant.

## Current System Prompt
{current_prompt}

## Task Results
The agent was given these tasks and produced these results:

{task_results}

## Instructions
Analyze what went wrong or could be improved. Write a concise critique (3-5 bullet points) \
focusing on specific, actionable changes to the system prompt that would improve task performance. \
Focus on: response structure, completeness, pattern coverage, and task understanding.

Critique:"""

EDIT_PROMPT = """\
You are an expert prompt engineer. Apply the following critique to improve a system prompt.

## Current System Prompt
{current_prompt}

## Critique
{critique}

## Instructions
Rewrite the COMPLETE system prompt incorporating the critique's suggestions. \
Keep the core identity and capabilities intact — only modify behavior guidance, \
communication style, or task handling based on the critique. \
Output ONLY the new system prompt, nothing else."""


def load_tasks() -> list[dict]:
    tasks = json.loads(TASKS_PATH.read_text())
    usable = []
    for t in tasks:
        if not t["prompt"] or t["prompt"].startswith("/"):
            continue
        if "discord_feed" in t.get("expected_tools", []):
            continue
        usable.append(t)
    return usable


def score_response(response: str, task: dict) -> float:
    if not response or not response.strip():
        return 0.0
    score = 0.0
    has_content = len(response.strip()) > 20
    if has_content:
        score += 0.2
    structure_markers = ["**", "##", "- ", "1.", "```", "| "]
    if any(m in response for m in structure_markers):
        score += 0.2
    patterns = task.get("expected_patterns", [])
    if patterns:
        matches = sum(1 for p in patterns if p.lower() in response.lower())
        score += 0.6 * (matches / len(patterns))
    else:
        score += 0.6 if has_content else 0.0
    return round(score, 3)


async def run_task(prompt: str, session_id: str, timeout: float = 120) -> str:
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
    soul_content: str, tasks: list[dict], label: str = "",
    use_llm_judge: bool = False, judge_client=None,
) -> tuple[float, list[dict]]:
    """Write SOUL.md, run tasks, return (avg_score, results)."""
    SOUL_PATH.write_text(soul_content)
    results = []
    for task in tasks:
        sid = f"apo-{label}-{task['id']}-{int(time.time())}"
        response = await run_task(task["prompt"], sid)

        if use_llm_judge:
            judge_score = await llm_judge.judge_response(task, response, client=judge_client)
            reward = judge_score.composite
        else:
            reward = score_response(response, task)

        results.append({
            "task_id": task["id"],
            "prompt": task["prompt"][:80],
            "score": reward,
            "response_preview": response[:200],
        })
        print(f"    {task['id']:25s} score={reward:.3f}")
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    return avg, results


async def generate_critique(
    client: AsyncOpenAI, model: str, current_prompt: str, task_results: list[dict]
) -> str:
    results_text = "\n".join(
        f"- Task: {r['prompt']}\n  Score: {r['score']}\n  Response: {r['response_preview']}"
        for r in task_results
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": GRADIENT_PROMPT.format(
                current_prompt=current_prompt, task_results=results_text
            ),
        }],
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


async def apply_edit(
    client: AsyncOpenAI, model: str, current_prompt: str, critique: str
) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": EDIT_PROMPT.format(
                current_prompt=current_prompt, critique=critique
            ),
        }],
        max_tokens=4096,
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""


async def run_apo(args):
    all_tasks = load_tasks()
    mid = len(all_tasks) // 2 + 1
    train_tasks = all_tasks[:mid]
    val_tasks = all_tasks[mid:] if len(all_tasks[mid:]) >= 2 else all_tasks

    print(f"Tasks: {len(all_tasks)} total, {len(train_tasks)} train, {len(val_tasks)} val")
    print(f"Rounds: {args.rounds}, Beam width: {args.beam_width}, Branch factor: {args.branch_factor}")
    print(f"Critic: {args.critic_model} via gateway, Agent: 27B via protoResearcher")
    print(f"Scorer: {'LLM judge (Sonnet 4.6)' if args.use_llm_judge else 'pattern matching'}")
    print()

    # Critic uses the gateway (stronger model) — key from env or Infisical
    import os
    gw_key = os.environ.get("GATEWAY_API_KEY", "not-needed")
    client = AsyncOpenAI(base_url=GATEWAY_URL, api_key=gw_key)
    critic_model = args.critic_model
    judge_client = llm_judge._make_client() if args.use_llm_judge else None
    seed_prompt = SOUL_PATH.read_text()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Bootstrap: inject few-shot examples into seed prompt (DSPy-inspired)
    if args.bootstrap:
        examples = bootstrap_fewshot.bootstrap_examples(
            args.bootstrap_rollouts, top_k=args.bootstrap_k,
        )
        if examples:
            seed_prompt = bootstrap_fewshot.inject_fewshot(seed_prompt, examples)
            print(f"Bootstrapped {len(examples)} few-shot examples into seed prompt "
                  f"(+{sum(len(bootstrap_fewshot.format_fewshot_block([e])) for e in examples)} chars)")
        else:
            print("Warning: --bootstrap enabled but no usable examples found")
        print()

    # Save original (with bootstrap if applied)
    (RESULTS_DIR / "original_soul.md").write_text(seed_prompt)

    # --- Evaluate seed ---
    print("=== Evaluating seed prompt ===")
    seed_score, seed_results = await evaluate_prompt(
        seed_prompt, val_tasks, "seed",
        use_llm_judge=args.use_llm_judge, judge_client=judge_client,
    )
    print(f"  Seed score: {seed_score:.3f}\n")

    beam = [{"prompt": seed_prompt, "score": seed_score, "version": "seed"}]
    best = {"prompt": seed_prompt, "score": seed_score, "version": "seed"}

    log = [{"round": 0, "version": "seed", "score": seed_score}]

    for rnd in range(1, args.rounds + 1):
        print(f"=== Round {rnd}/{args.rounds} ===")

        new_candidates = []
        for parent in beam:
            # Get training rollout results for gradient
            print(f"  Evaluating parent '{parent['version']}' on train tasks for gradient...")
            _, train_results = await evaluate_prompt(
                parent["prompt"], train_tasks, f"r{rnd}-train",
                use_llm_judge=args.use_llm_judge, judge_client=judge_client,
            )

            for b in range(args.branch_factor):
                # Generate critique (textual gradient)
                print(f"  Generating critique {b+1}/{args.branch_factor}...")
                critique = await generate_critique(client, critic_model, parent["prompt"], train_results)

                # Apply edit to create candidate
                print(f"  Applying edit...")
                candidate_prompt = await apply_edit(client, critic_model, parent["prompt"], critique)

                if not candidate_prompt or len(candidate_prompt) < 100:
                    print(f"  [skip] Generated prompt too short ({len(candidate_prompt or '')} chars)")
                    continue

                version = f"r{rnd}-p{beam.index(parent)}-b{b}"
                new_candidates.append({
                    "prompt": candidate_prompt,
                    "score": None,
                    "version": version,
                    "critique": critique[:200],
                })

        # Evaluate all candidates on validation
        all_candidates = beam + new_candidates
        print(f"\n  Evaluating {len(all_candidates)} candidates on validation...")
        for c in all_candidates:
            if c["score"] is None:
                print(f"  --- {c['version']} ---")
                score, _ = await evaluate_prompt(
                    c["prompt"], val_tasks, c["version"],
                    use_llm_judge=args.use_llm_judge, judge_client=judge_client,
                )
                c["score"] = score
                print(f"  {c['version']} score: {score:.3f}")

        # Select top-k
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = all_candidates[: args.beam_width]

        print(f"\n  Beam after round {rnd}:")
        for c in beam:
            print(f"    {c['version']:20s} score={c['score']:.3f}")
            log.append({"round": rnd, "version": c["version"], "score": c["score"]})

        # Update best
        if beam[0]["score"] > best["score"]:
            best = beam[0]
            print(f"  New best: {best['version']} ({best['score']:.3f})")
        print()

    # --- Save results ---
    print("=" * 60)
    print(f"  BEST: {best['version']} — score {best['score']:.3f} (seed was {seed_score:.3f})")
    print(f"  Delta: {best['score'] - seed_score:+.3f}")
    print("=" * 60)

    (RESULTS_DIR / "best_soul.md").write_text(best["prompt"])
    (RESULTS_DIR / "log.json").write_text(json.dumps(log, indent=2))

    # Restore best prompt as active
    SOUL_PATH.write_text(best["prompt"])
    print(f"\nBest prompt written to SOUL.md and saved to {RESULTS_DIR / 'best_soul.md'}")


def main():
    parser = argparse.ArgumentParser(description="APO for protoResearcher SOUL.md")
    parser.add_argument("--rounds", type=int, default=2, help="Beam search rounds")
    parser.add_argument("--beam-width", type=int, default=2, help="Top-k prompts kept per round")
    parser.add_argument("--branch-factor", type=int, default=2, help="Candidates per parent")
    parser.add_argument("--critic-model", default=DEFAULT_CRITIC_MODEL, help="Model for critique/edit (via gateway)")
    parser.add_argument("--use-llm-judge", action="store_true", help="Use Sonnet 4.6 LLM judge instead of pattern matching")
    parser.add_argument("--bootstrap", action="store_true", help="Inject few-shot examples from rollouts into seed prompt (DSPy-inspired)")
    parser.add_argument("--bootstrap-k", type=int, default=3, help="Number of few-shot examples to inject")
    parser.add_argument("--bootstrap-rollouts", type=str, nargs="+",
                        default=["experiments/agent-lightning/results/rollouts/rollouts_*.jsonl"],
                        help="Rollout JSONL files for bootstrap")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate seed only")
    args = parser.parse_args()

    if args.dry_run:
        tasks = load_tasks()
        judge_client = llm_judge._make_client() if args.use_llm_judge else None
        scorer = "LLM judge" if args.use_llm_judge else "pattern matching"
        seed = SOUL_PATH.read_text()
        if args.bootstrap:
            examples = bootstrap_fewshot.bootstrap_examples(
                args.bootstrap_rollouts, top_k=args.bootstrap_k,
            )
            if examples:
                seed = bootstrap_fewshot.inject_fewshot(seed, examples)
                print(f"Bootstrapped {len(examples)} few-shot examples")
        print(f"Dry run — evaluating seed on {len(tasks)} tasks (scorer: {scorer})")
        score, results = asyncio.run(evaluate_prompt(
            seed, tasks, "dryrun",
            use_llm_judge=args.use_llm_judge, judge_client=judge_client,
        ))
        print(f"\nSeed score: {score:.3f}")
        for r in results:
            print(f"  {r['task_id']:25s} {r['score']:.3f}")
        return

    asyncio.run(run_apo(args))


if __name__ == "__main__":
    main()
