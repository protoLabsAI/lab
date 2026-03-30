"""Context-1 retrieval agent harness.

Implements the observe-reason-act loop for Context-1:
1. Build prompt with system message, tools, and history
2. Generate with vLLM (stop on <|call|> or <|return|>)
3. Parse tool calls or final answer
4. Execute tools, feed results back
5. Repeat until final answer or budget exhausted

Usage:
    python -m experiments.context-1.harness \
        --query "What papers discuss FP8 quantization for MoE models?" \
        --corpus /path/to/corpus.json \
        --vllm-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from .format import (
    CALL,
    RETURN,
    STOP_TOKENS,
    ToolDef,
    build_prompt,
    parse_assistant_output,
)
from .tools import TOOL_DEFS, DocumentStore, ToolExecutor

SYSTEM_PROMPT = """\
You are a retrieval subagent in a multi-agent system. Your ONLY role is to \
find and retrieve relevant documents from a corpus using the provided tools. \
You must ALWAYS use search_corpus or grep_corpus before providing any answer. \
Never answer without first searching. You do NOT answer questions yourself — \
you only find and retrieve relevant documents.

Strategy:
1. Break the query into key concepts and develop search strategies for each
2. Execute searches via tool calls (use search_corpus and grep_corpus)
3. After each round, evaluate: what do I know? what should I search next? what should I prune?
4. When you have found sufficient relevant documents, provide your final answer listing the retrieved chunks

Be thorough — try multiple query formulations, synonyms, and related terms. \
Prune irrelevant chunks to stay within token budget."""

MAX_TURNS = 20
SOFT_THRESHOLD_RATIO = 0.75  # suggest pruning at 75% budget
HARD_CUTOFF_RATIO = 0.90  # reject non-prune tools at 90%


def generate(
    prompt: str,
    vllm_url: str,
    model: str = "local",
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> dict:
    """Call vLLM completions API (not chat — we format the prompt ourselves)."""
    # Use token IDs for stop tokens — string matching doesn't work for special tokens
    # <|call|> = 200012, <|return|> = 200002
    STOP_TOKEN_IDS = [200012, 200002]

    resp = httpx.post(
        f"{vllm_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop_token_ids": STOP_TOKEN_IDS,
            "skip_special_tokens": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    return {
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def run_agent(
    query: str,
    store: DocumentStore,
    vllm_url: str = "http://localhost:8000",
    model: str = "local",
    token_budget: int = 24000,
    verbose: bool = True,
) -> dict:
    """Run the full retrieval agent loop.

    Returns:
        dict with keys: query, retrieved_chunks, turns, total_tokens, elapsed_s, history
    """
    executor = ToolExecutor(store, token_budget=token_budget)
    history: list[dict] = [{"role": "user", "content": query}]
    all_messages: list[dict] = []

    t0 = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn in range(MAX_TURNS):
        # Build prompt
        prompt = build_prompt(
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFS,
            history=history,
            token_usage=executor.token_usage,
        )

        if verbose:
            used, budget = executor.token_usage
            print(f"\n--- Turn {turn + 1} [{used:,}/{budget:,} tokens] ---")

        # Check hard cutoff
        used, budget = executor.token_usage
        if used / budget > HARD_CUTOFF_RATIO:
            if verbose:
                print("  [HARD CUTOFF] Token budget nearly exhausted, forcing conclusion.")
            # Inject system nudge
            history.append({
                "role": "user",
                "content": "[System: Token budget nearly exhausted. Please provide your final answer with the chunks you've found, or prune to make room.]",
            })

        # Generate
        result = generate(prompt, vllm_url, model=model)
        raw_text = result["text"]
        total_prompt_tokens += result["usage"].get("prompt_tokens", 0)
        total_completion_tokens += result["usage"].get("completion_tokens", 0)

        if verbose:
            print(f"  Model output ({len(raw_text)} chars): {raw_text[:200]}...")

        # Parse output
        messages = parse_assistant_output(raw_text)
        if not messages:
            if verbose:
                print("  [WARN] No parseable output, retrying...")
            continue

        # Process all messages first, then decide if we're done.
        # In Harmony format, the model can generate analysis → final → tool call
        # in a single pass. Only treat as done if there are NO tool calls.
        has_tool_call = False
        final_answer = None

        for msg in messages:
            all_messages.append(msg)
            channel = msg.get("channel", "")

            if msg.get("tool_call"):
                has_tool_call = True
                tc = msg["tool_call"]

                # Enforce hard cutoff — only allow prune
                if (
                    used / budget > HARD_CUTOFF_RATIO
                    and tc["name"] != "prune_chunks"
                ):
                    tool_result = json.dumps({
                        "error": "Token budget exceeded. Only prune_chunks is allowed."
                    })
                else:
                    tool_result = executor.execute(tc["name"], tc["arguments"])

                if verbose:
                    print(f"  -> {tc['name']}({tc['arguments'][:80]}...) = {tool_result[:120]}...")

                # Add to history
                history.append({
                    "role": "assistant",
                    "channel": channel or "commentary",
                    "content": tc["arguments"],
                    "tool_call": tc,
                })
                history.append({
                    "role": "tool_result",
                    "tool_name": tc["name"],
                    "content": tool_result,
                })

            elif channel == "analysis":
                history.append({
                    "role": "assistant",
                    "channel": "analysis",
                    "content": msg["content"],
                })
                if verbose:
                    print(f"  [analysis] {msg['content'][:150]}...")

            elif channel == "final":
                # Store as candidate final answer — but don't return yet
                history.append({
                    "role": "assistant",
                    "channel": "final",
                    "content": msg["content"],
                })
                final_answer = msg["content"]
                if verbose:
                    print(f"  [final] {msg['content'][:150]}...")

        # After processing all messages: if there were tool calls, continue the loop.
        # If no tool calls and we have a final answer (or finish was <|return|>), we're done.
        if not has_tool_call and final_answer is not None:
            if verbose:
                print(f"\n=== FINAL ANSWER ===\n{final_answer[:500]}")

            elapsed = time.time() - t0
            retrieved = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content,
                }
                for c in executor.active_chunks.values()
            ]

            return {
                "query": query,
                "answer": final_answer,
                "retrieved_chunks": retrieved,
                "turns": turn + 1,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "elapsed_s": round(elapsed, 2),
                "history": history,
            }

        # Soft threshold nudge
        if used / budget > SOFT_THRESHOLD_RATIO and not has_tool_call:
            history.append({
                "role": "user",
                "content": "[System: Token budget is getting high. Consider pruning irrelevant chunks or wrapping up.]",
            })

    # Exceeded max turns
    elapsed = time.time() - t0
    retrieved = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content}
        for c in executor.active_chunks.values()
    ]
    return {
        "query": query,
        "answer": "[Max turns exceeded]",
        "retrieved_chunks": retrieved,
        "turns": MAX_TURNS,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "elapsed_s": round(elapsed, 2),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Context-1 retrieval agent harness")
    parser.add_argument("--query", required=True, help="Query to search for")
    parser.add_argument("--corpus", required=True, help="Path to corpus (JSON or directory)")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM base URL")
    parser.add_argument("--model", default="local", help="Model name on vLLM")
    parser.add_argument("--token-budget", type=int, default=24000, help="Token budget")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    # Load corpus
    store = DocumentStore()
    corpus_path = Path(args.corpus)
    if corpus_path.is_dir():
        n = store.load_from_directory(corpus_path)
        print(f"Loaded {n} documents from {corpus_path}")
    elif corpus_path.suffix == ".json":
        n = store.load_from_json(corpus_path)
        print(f"Loaded {n} documents from {corpus_path}")
    elif corpus_path.suffix == ".db":
        n = store.load_from_protoresearcher(corpus_path)
        print(f"Loaded {n} items from protoResearcher DB")
    else:
        print(f"Unknown corpus format: {corpus_path}")
        sys.exit(1)

    print(f"Store: {len(store.documents)} docs, {len(store.chunks)} chunks")

    result = run_agent(
        query=args.query,
        store=store,
        vllm_url=args.vllm_url,
        model=args.model,
        token_budget=args.token_budget,
        verbose=not args.quiet,
    )

    print(f"\n{'='*60}")
    print(f"Query: {result['query']}")
    print(f"Turns: {result['turns']}")
    print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
    print(f"Tokens: {result['total_prompt_tokens']} prompt + {result['total_completion_tokens']} completion")
    print(f"Time: {result['elapsed_s']}s")
    print(f"Answer: {result['answer'][:300]}")


if __name__ == "__main__":
    main()
