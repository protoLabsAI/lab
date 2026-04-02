"""Harness for running protoCLI on SWE-bench and terminal-bench tasks.

Wraps protoCLI's headless mode with task execution, trace collection,
and reward computation. Designed for:
  1. APO system prompt optimization (critique → edit → evaluate)
  2. Rollout collection for DPO/GRPO training
  3. SWE-bench and terminal-bench evaluation

Architecture mirrors agent-lightning's claude_code example but uses
protoCLI's native non-interactive mode instead of Claude Code.

Usage:
    # Run SWE-bench tasks through protoCLI
    python experiments/proto-bench/proto_agent.py swebench \
      --dataset-path experiments/proto-bench/tasks/swebench_samples.jsonl \
      --model protolabs/local \
      --max-turns 25 \
      --output-dir experiments/proto-bench/results

    # Run terminal-bench tasks
    python experiments/proto-bench/proto_agent.py terminal \
      --tasks-dir experiments/proto-bench/tasks/terminal \
      --model protolabs/local

    # APO mode: evaluate system prompt variants
    python experiments/proto-bench/proto_agent.py apo \
      --dataset-path experiments/proto-bench/tasks/swebench_samples.jsonl \
      --prompt-path ~/.proto/AGENTS.md \
      --rounds 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTO_CLI = os.environ.get("PROTO_CLI", "proto")
RESULTS_DIR = Path(__file__).parent / "results"
PROTO_HOME = Path.home() / ".proto"


def run_proto_headless(
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
    max_turns: int = 25,
    timeout: int = 600,
    env_extra: dict[str, str] | None = None,
) -> dict:
    """Run protoCLI in non-interactive (headless) mode.

    Returns dict with: output, exit_code, elapsed_s, tool_calls (from telemetry).
    """
    cmd = [PROTO_CLI, "-p", prompt, "-o", "json", "--max-session-turns", str(max_turns)]
    if model:
        cmd.extend(["--model", model])

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        elapsed = time.monotonic() - t0

        # Parse JSON output if available
        output = result.stdout
        try:
            parsed = json.loads(output)
            text = parsed.get("text", output)
        except (json.JSONDecodeError, TypeError):
            text = output
            parsed = {}

        return {
            "output": text,
            "raw_stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "elapsed_s": round(elapsed, 1),
            "parsed": parsed,
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "raw_stdout": "",
            "stderr": "TIMEOUT",
            "exit_code": -1,
            "elapsed_s": timeout,
            "parsed": {},
        }


def load_swebench_tasks(path: str, limit: int | None = None) -> list[dict]:
    """Load SWE-bench instances from JSONL."""
    tasks = []
    with open(path) as f:
        for line in f:
            tasks.append(json.loads(line))
    if limit:
        tasks = tasks[:limit]
    return tasks


def swebench_prompt(instance: dict) -> str:
    """Format a SWE-bench instance as a prompt for protoCLI."""
    return (
        f"Fix the following issue in the repository.\n\n"
        f"Repository: {instance.get('repo', 'unknown')}\n"
        f"Issue: {instance.get('problem_statement', instance.get('issue', ''))}\n\n"
        f"Instructions:\n"
        f"1. Read the relevant code to understand the issue\n"
        f"2. Implement the fix\n"
        f"3. Verify your fix works\n"
        f"4. Create a git patch with your changes"
    )


def extract_patch(result: dict, cwd: str) -> str:
    """Extract git diff from the working directory after protoCLI runs."""
    try:
        diff = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True, cwd=cwd, timeout=10,
        )
        return diff.stdout.strip()
    except Exception:
        return ""


async def run_swebench(args):
    """Run SWE-bench tasks through protoCLI."""
    tasks = load_swebench_tasks(args.dataset_path, limit=args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks: {len(tasks)}")
    print(f"Model: {args.model}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {output_dir}")
    print()

    results = []
    for i, task in enumerate(tasks):
        instance_id = task.get("instance_id", f"task_{i}")
        print(f"[{i+1}/{len(tasks)}] {instance_id}...", end=" ", flush=True)

        prompt = swebench_prompt(task)

        # TODO: Set up repo checkout in container/tmpdir for each instance
        # For now, run in current directory (needs proper SWE-bench setup)
        result = run_proto_headless(
            prompt=prompt,
            model=args.model,
            max_turns=args.max_turns,
            timeout=args.timeout,
        )

        patch = ""  # extract_patch(result, cwd) when repo setup is ready

        entry = {
            "instance_id": instance_id,
            "model_patch": patch,
            "output": result["output"][:2000],
            "exit_code": result["exit_code"],
            "elapsed_s": result["elapsed_s"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(entry)

        status = "PATCH" if patch else ("OK" if result["exit_code"] == 0 else "FAIL")
        print(f"{status} ({result['elapsed_s']}s)")

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"swebench_{ts}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {out_path}")

    # Summary
    successes = sum(1 for r in results if r["exit_code"] == 0)
    patches = sum(1 for r in results if r["model_patch"])
    print(f"Completed: {successes}/{len(results)}")
    print(f"Patches generated: {patches}/{len(results)}")


async def run_apo(args):
    """APO mode: optimize protoCLI's system prompt on coding tasks."""
    # Reuse the agent-lightning APO pattern but with protoCLI as the agent
    # and SWE-bench/terminal-bench pass/fail as the reward signal
    print("APO mode for protoCLI system prompt optimization")
    print(f"Prompt path: {args.prompt_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Rounds: {args.rounds}")
    print()
    print("TODO: Wire up APO beam search with protoCLI execution")
    print("Pattern: experiments/agent-lightning/run_apo.py adapted for coding tasks")
    print("Key difference: reward = task pass/fail (binary) instead of LLM judge")


def main():
    parser = argparse.ArgumentParser(description="protoCLI benchmark harness")
    subparsers = parser.add_subparsers(dest="command")

    # SWE-bench
    swe = subparsers.add_parser("swebench", help="Run SWE-bench tasks")
    swe.add_argument("--dataset-path", required=True)
    swe.add_argument("--model", default="protolabs/local")
    swe.add_argument("--max-turns", type=int, default=25)
    swe.add_argument("--timeout", type=int, default=600)
    swe.add_argument("--output-dir", default=str(RESULTS_DIR))
    swe.add_argument("--limit", type=int, default=None)

    # Terminal-bench
    tb = subparsers.add_parser("terminal", help="Run terminal-bench tasks")
    tb.add_argument("--tasks-dir", required=True)
    tb.add_argument("--model", default="protolabs/local")

    # APO
    apo = subparsers.add_parser("apo", help="APO system prompt optimization")
    apo.add_argument("--dataset-path", required=True)
    apo.add_argument("--prompt-path", default=str(PROTO_HOME / "AGENTS.md"))
    apo.add_argument("--model", default="protolabs/local")
    apo.add_argument("--rounds", type=int, default=2)

    args = parser.parse_args()

    if args.command == "swebench":
        asyncio.run(run_swebench(args))
    elif args.command == "apo":
        asyncio.run(run_apo(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
