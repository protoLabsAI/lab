"""APO (Automatic Prompt Optimization) for protoCLI's system prompt.

Optimizes the system prompt for coding task performance using textual
gradient beam search. Same algorithm as agent-lightning APO but targeting
protoCLI with coding tasks and binary pass/fail scoring.

Injection: QWEN_SYSTEM_MD env var overrides protoCLI's system prompt.
Scoring: Task success (exit_code 0 + expected output) = 1.0, else 0.0.

Usage:
    cd ~/dev/lab
    export GATEWAY_API_KEY=<from-infisical>

    # Dry run — evaluate seed prompt on coding tasks
    uv run python experiments/proto-bench/run_apo.py --dry-run

    # Full APO run
    uv run python experiments/proto-bench/run_apo.py --rounds 2 --beam-width 2

    # With LLM judge (richer signal than binary pass/fail)
    uv run python experiments/proto-bench/run_apo.py --rounds 2 --use-llm-judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

PROTO_CLI = os.environ.get("PROTO_CLI", "proto")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://ava:4000/v1")
CRITIC_MODEL = "cli/claude-sonnet-4-6"
SEED_PROMPT_PATH = Path(__file__).parent / "templates" / "seed_prompt.md"
RESULTS_DIR = Path(__file__).parent / "results"

# Coding tasks for evaluation — mix of simple edits, bug fixes, and multi-step
EVAL_TASKS = [
    {
        "id": "create_function",
        "prompt": "Create a file called /tmp/proto_test/fizzbuzz.py that implements a fizzbuzz function. It should take an integer n and return a list of strings from 1 to n, replacing multiples of 3 with 'Fizz', 5 with 'Buzz', and both with 'FizzBuzz'.",
        "check": lambda: _file_contains("/tmp/proto_test/fizzbuzz.py", "def fizzbuzz"),
        "difficulty": "simple",
    },
    {
        "id": "fix_bug",
        "prompt": "There's a bug in /tmp/proto_test/broken.py — the function returns the wrong result for negative numbers. Read it, find the bug, and fix it.",
        "setup": lambda: _write_file("/tmp/proto_test/broken.py", '''def absolute_value(n):\n    """Return the absolute value of n."""\n    if n > 0:\n        return n\n    else:\n        return n  # BUG: should be -n\n'''),
        "check": lambda: _file_contains("/tmp/proto_test/broken.py", "return -n") or _file_contains("/tmp/proto_test/broken.py", "return abs("),
        "difficulty": "simple",
    },
    {
        "id": "read_and_summarize",
        "prompt": "Read /tmp/proto_test/data.json and tell me: how many items are there, and what is the average price?",
        "setup": lambda: _write_file("/tmp/proto_test/data.json", json.dumps([
            {"name": "Widget A", "price": 10.00},
            {"name": "Widget B", "price": 20.00},
            {"name": "Widget C", "price": 30.00},
        ])),
        "check_output": lambda out: "3" in out and ("20" in out or "20.0" in out),
        "difficulty": "simple",
    },
    {
        "id": "write_test",
        "prompt": "Write a pytest test file at /tmp/proto_test/test_utils.py that tests the add and multiply functions in /tmp/proto_test/utils.py. Each function should have at least 2 test cases.",
        "setup": lambda: _write_file("/tmp/proto_test/utils.py", '''def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n'''),
        "check": lambda: _file_contains("/tmp/proto_test/test_utils.py", "def test_") and _file_contains("/tmp/proto_test/test_utils.py", "assert"),
        "difficulty": "medium",
    },
    {
        "id": "refactor_class",
        "prompt": "Refactor /tmp/proto_test/user.py to use dataclasses instead of the manual __init__ and __repr__. Keep the validate_email method.",
        "setup": lambda: _write_file("/tmp/proto_test/user.py", '''class User:\n    def __init__(self, name, email, age):\n        self.name = name\n        self.email = email\n        self.age = age\n\n    def __repr__(self):\n        return f"User(name={self.name}, email={self.email}, age={self.age})"\n\n    def validate_email(self):\n        return "@" in self.email and "." in self.email\n'''),
        "check": lambda: _file_contains("/tmp/proto_test/user.py", "dataclass") and _file_contains("/tmp/proto_test/user.py", "validate_email"),
        "difficulty": "medium",
    },
]


def _write_file(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)


def _file_contains(path: str, substring: str) -> bool:
    try:
        return substring in Path(path).read_text()
    except FileNotFoundError:
        return False


def run_task(prompt: str, system_prompt_path: str, timeout: int = 120) -> dict:
    """Run a single task through protoCLI with the given system prompt."""
    env = os.environ.copy()
    env["QWEN_SYSTEM_MD"] = system_prompt_path

    cmd = [PROTO_CLI, "-p", prompt, "-o", "json", "--max-session-turns", "15", "-y"]

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd="/tmp/proto_test", env=env,
        )
        elapsed = time.monotonic() - t0

        # Parse result event from JSON stream
        output_text = ""
        is_error = False
        num_turns = 0
        for line in result.stdout.strip().split("\n"):
            try:
                events = json.loads(line) if line.startswith("[") else [json.loads(line)]
                for evt in (events if isinstance(events, list) else [events]):
                    if evt.get("type") == "result":
                        output_text = evt.get("result", "")
                        is_error = evt.get("is_error", False)
                        num_turns = evt.get("num_turns", 0)
            except (json.JSONDecodeError, TypeError):
                continue

        return {
            "output": output_text,
            "exit_code": result.returncode,
            "is_error": is_error,
            "num_turns": num_turns,
            "elapsed_s": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "exit_code": -1,
            "is_error": True,
            "num_turns": 0,
            "elapsed_s": timeout,
        }


def evaluate_prompt(system_prompt_path: str, tasks: list[dict], label: str = "") -> tuple[float, list[dict]]:
    """Evaluate a system prompt on all tasks. Returns (avg_score, results)."""
    Path("/tmp/proto_test").mkdir(parents=True, exist_ok=True)
    results = []

    for task in tasks:
        # Run setup if needed
        if "setup" in task:
            task["setup"]()

        result = run_task(task["prompt"], system_prompt_path, timeout=120)

        # Score: check file output or text output
        score = 0.0
        if "check" in task:
            score = 1.0 if task["check"]() else 0.0
        elif "check_output" in task:
            score = 1.0 if task["check_output"](result["output"]) else 0.0
        elif result["exit_code"] == 0 and not result["is_error"]:
            score = 0.5  # Completed but no specific check

        results.append({
            "task_id": task["id"],
            "difficulty": task["difficulty"],
            "score": score,
            "elapsed_s": result["elapsed_s"],
            "num_turns": result["num_turns"],
            "output_preview": result["output"][:200],
        })
        print(f"    {task['id']:25s} score={score:.1f} ({result['elapsed_s']}s, {result['num_turns']} turns)")

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    return avg, results


GRADIENT_PROMPT = """\
You are an expert prompt engineer optimizing a system prompt for an AI coding agent (protoCLI).

## Current System Prompt
{current_prompt}

## Task Results
The agent was given these coding tasks and produced these results:

{task_results}

## Instructions
Analyze what went wrong or could be improved. Write a concise critique (3-5 bullet points) \
focusing on specific, actionable changes to the system prompt that would improve coding task performance. \
Focus on: tool usage patterns, error recovery, file operation reliability, and task completion.

Critique:"""

EDIT_PROMPT = """\
You are an expert prompt engineer. Apply the following critique to improve a system prompt for an AI coding agent.

## Current System Prompt
{current_prompt}

## Critique
{critique}

## Instructions
Rewrite the COMPLETE system prompt incorporating the critique's suggestions. \
Keep the core identity and capabilities intact — only modify behavior guidance, \
tool usage patterns, or task handling based on the critique. \
Output ONLY the new system prompt, nothing else."""


async def generate_critique(client: AsyncOpenAI, model: str, prompt: str, results: list[dict]) -> str:
    results_text = "\n".join(
        f"- Task: {r['task_id']} ({r['difficulty']})\n  Score: {r['score']}\n  Output: {r['output_preview']}"
        for r in results
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": GRADIENT_PROMPT.format(
            current_prompt=prompt, task_results=results_text,
        )}],
        max_tokens=1024, temperature=0.7,
    )
    return resp.choices[0].message.content or ""


async def apply_edit(client: AsyncOpenAI, model: str, prompt: str, critique: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": EDIT_PROMPT.format(
            current_prompt=prompt, critique=critique,
        )}],
        max_tokens=8192, temperature=0.4,
    )
    return resp.choices[0].message.content or ""


async def run_apo(args):
    tasks = EVAL_TASKS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Tasks: {len(tasks)}")
    print(f"Rounds: {args.rounds}, Beam: {args.beam_width}, Branch: {args.branch_factor}")
    print(f"Critic: {CRITIC_MODEL}")
    print()

    gw_key = os.environ.get("GATEWAY_API_KEY", "not-needed")
    client = AsyncOpenAI(base_url=GATEWAY_URL, api_key=gw_key)

    seed_prompt = SEED_PROMPT_PATH.read_text()
    seed_path = str(RESULTS_DIR / "seed_prompt.md")
    Path(seed_path).write_text(seed_prompt)

    # Evaluate seed
    print("=== Evaluating seed prompt ===")
    seed_score, seed_results = evaluate_prompt(seed_path, tasks, "seed")
    print(f"  Seed score: {seed_score:.3f}\n")

    beam = [{"prompt": seed_prompt, "score": seed_score, "version": "seed"}]
    best = beam[0].copy()
    log = [{"round": 0, "version": "seed", "score": seed_score}]

    for rnd in range(1, args.rounds + 1):
        print(f"=== Round {rnd}/{args.rounds} ===")
        new_candidates = []

        for parent in beam:
            # Get results for gradient
            parent_path = str(RESULTS_DIR / f"r{rnd}_parent.md")
            Path(parent_path).write_text(parent["prompt"])
            _, train_results = evaluate_prompt(parent_path, tasks, f"r{rnd}-train")

            for b in range(args.branch_factor):
                print(f"  Generating critique {b+1}/{args.branch_factor}...")
                critique = await generate_critique(client, CRITIC_MODEL, parent["prompt"], train_results)
                print(f"  Applying edit...")
                candidate = await apply_edit(client, CRITIC_MODEL, parent["prompt"], critique)

                if not candidate or len(candidate) < 100:
                    continue

                version = f"r{rnd}-p{beam.index(parent)}-b{b}"
                new_candidates.append({
                    "prompt": candidate,
                    "score": None,
                    "version": version,
                    "critique": critique[:200],
                })

        # Evaluate candidates
        all_candidates = beam + new_candidates
        for c in all_candidates:
            if c["score"] is None:
                cpath = str(RESULTS_DIR / f"{c['version']}.md")
                Path(cpath).write_text(c["prompt"])
                print(f"  --- {c['version']} ---")
                score, _ = evaluate_prompt(cpath, tasks, c["version"])
                c["score"] = score

        # Select top-k
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        beam = all_candidates[:args.beam_width]

        print(f"\n  Beam after round {rnd}:")
        for c in beam:
            print(f"    {c['version']:20s} score={c['score']:.3f}")
            log.append({"round": rnd, "version": c["version"], "score": c["score"]})

        if beam[0]["score"] > best["score"]:
            best = beam[0].copy()

    # Save results
    print(f"\n{'='*60}")
    print(f"  BEST: {best['version']} — score {best['score']:.3f} (seed was {seed_score:.3f})")
    print(f"  Delta: {best['score'] - seed_score:+.3f}")
    print(f"{'='*60}")

    (RESULTS_DIR / "best_prompt.md").write_text(best["prompt"])
    (RESULTS_DIR / "apo_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nBest prompt saved to {RESULTS_DIR / 'best_prompt.md'}")


def main():
    parser = argparse.ArgumentParser(description="APO for protoCLI system prompt")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Evaluate seed only")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run — evaluating seed prompt")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        seed_path = str(RESULTS_DIR / "seed_prompt.md")
        Path(seed_path).write_text(SEED_PROMPT_PATH.read_text())
        score, results = evaluate_prompt(seed_path, EVAL_TASKS, "dryrun")
        print(f"\nSeed score: {score:.3f}")
        return

    asyncio.run(run_apo(args))


if __name__ == "__main__":
    main()
