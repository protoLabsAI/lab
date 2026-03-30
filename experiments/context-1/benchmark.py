"""Quick benchmark for Context-1 on vLLM.

Tests raw generation speed and validates the Harmony format works.

Usage:
    python -m experiments.context-1.benchmark [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

from .format import (
    START,
    END,
    MESSAGE,
    CHANNEL,
    CALL,
    RETURN,
    STOP_TOKENS,
    build_prompt,
    parse_assistant_output,
)
from .tools import TOOL_DEFS

SYSTEM_PROMPT = """\
You are a retrieval subagent. Your role is to find relevant documents from a corpus.
Break queries into subqueries and use search tools to find information."""

TEST_PROMPTS = [
    # Simple completion test
    {
        "name": "basic_generation",
        "prompt": f"{START}system{MESSAGE}You are a helpful assistant.{END}{START}user{MESSAGE}What is the capital of France?{END}{START}assistant",
        "max_tokens": 128,
    },
    # Tool call test
    {
        "name": "tool_call_generation",
        "prompt": None,  # built dynamically
        "max_tokens": 512,
        "query": "Find papers about FP8 quantization for MoE models",
    },
    # Multi-turn throughput
    {
        "name": "throughput_test",
        "prompt": f"{START}system{MESSAGE}You are a helpful assistant. Write a detailed response.{END}{START}user{MESSAGE}Explain how mixture of experts models work, including the routing mechanism, load balancing, and common training challenges.{END}{START}assistant",
        "max_tokens": 1024,
    },
]


def run_completion(
    prompt: str,
    url: str,
    model: str = "local",
    max_tokens: int = 256,
) -> dict:
    """Run a single completion and return timing + output."""
    t0 = time.time()
    resp = httpx.post(
        f"{url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stop": STOP_TOKENS + [END],
            "skip_special_tokens": False,
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})

    return {
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": round(elapsed, 3),
        "tok_per_s": round(
            usage.get("completion_tokens", 0) / max(elapsed, 0.001), 1
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Context-1 benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="local")
    parser.add_argument("--runs", type=int, default=3, help="Runs per test")
    args = parser.parse_args()

    print(f"Benchmarking Context-1 at {args.url}")
    print(f"Model: {args.model}, Runs: {args.runs}")

    # Check model is up
    try:
        resp = httpx.get(f"{args.url}/v1/models", timeout=5)
        models = resp.json()
        print(f"Available models: {[m['id'] for m in models['data']]}")
    except Exception as e:
        print(f"ERROR: Cannot reach vLLM at {args.url}: {e}")
        return

    print()

    for test in TEST_PROMPTS:
        name = test["name"]
        max_tokens = test["max_tokens"]

        # Build prompt
        if test["prompt"] is None:
            prompt = build_prompt(
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFS,
                history=[{"role": "user", "content": test["query"]}],
            )
        else:
            prompt = test["prompt"]

        print(f"--- {name} (max_tokens={max_tokens}) ---")
        print(f"  Prompt length: ~{len(prompt)} chars")

        results = []
        for i in range(args.runs):
            r = run_completion(prompt, args.url, args.model, max_tokens)
            results.append(r)
            print(
                f"  Run {i+1}: {r['completion_tokens']} tokens, "
                f"{r['tok_per_s']} tok/s, {r['elapsed_s']}s, "
                f"finish={r['finish_reason']}"
            )

        # Averages
        avg_tps = round(sum(r["tok_per_s"] for r in results) / len(results), 1)
        avg_elapsed = round(sum(r["elapsed_s"] for r in results) / len(results), 3)
        avg_tokens = round(sum(r["completion_tokens"] for r in results) / len(results))

        print(f"  AVG: {avg_tps} tok/s, {avg_tokens} tokens, {avg_elapsed}s")

        # Show first run output
        first_text = results[0]["text"][:300]
        print(f"  Output preview: {first_text!r}")

        # Parse tool calls if present
        if test["name"] == "tool_call_generation":
            parsed = parse_assistant_output(results[0]["text"])
            tool_calls = [m for m in parsed if m.get("tool_call")]
            analysis = [m for m in parsed if m.get("channel") == "analysis"]
            print(f"  Parsed: {len(analysis)} analysis, {len(tool_calls)} tool calls")
            for tc in tool_calls:
                print(f"    -> {tc['tool_call']['name']}({tc['tool_call']['arguments'][:80]})")

        print()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
