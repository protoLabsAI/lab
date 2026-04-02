"""Few-shot bootstrap for APO (inspired by DSPy MIPROv2).

Selects high-scoring rollouts and formats them as few-shot examples
to inject into the system prompt before the APO critique-edit loop.

The key insight from DSPy: few-shot examples in the prompt are often
higher-leverage than instruction editing alone. By showing the model
what a great response looks like, we anchor the optimization around
quality patterns rather than abstract instructions.

Usage:
    from bootstrap_fewshot import bootstrap_examples, inject_fewshot

    # Select top examples from rollouts
    examples = bootstrap_examples("results/rollouts/rollouts_*.jsonl", top_k=3)

    # Inject into system prompt
    augmented_prompt = inject_fewshot(original_prompt, examples)
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path


def load_rollouts(patterns: list[str] | str) -> list[dict]:
    """Load rollouts from JSONL files."""
    if isinstance(patterns, str):
        patterns = [patterns]
    records = []
    for pat in patterns:
        for fpath in glob(pat):
            with open(fpath) as f:
                for line in f:
                    records.append(json.loads(line))
    return records


def bootstrap_examples(
    rollout_patterns: list[str] | str,
    top_k: int = 3,
    min_reward: float = 0.5,
    diverse: bool = True,
    max_response_len: int = 800,
) -> list[dict]:
    """Select high-scoring rollouts as few-shot examples.

    Args:
        rollout_patterns: Glob patterns for rollout JSONL files.
        top_k: Number of examples to select.
        min_reward: Minimum reward threshold.
        diverse: If True, prefer one example per task category.
        max_response_len: Truncate responses longer than this.

    Returns:
        List of dicts with keys: task_prompt, response, reward, category, task_id.
    """
    rollouts = load_rollouts(rollout_patterns)

    # Filter by minimum reward
    candidates = [r for r in rollouts if r["reward"] >= min_reward]
    if not candidates:
        print(f"[bootstrap] No rollouts above min_reward={min_reward}, lowering to 0.3")
        candidates = [r for r in rollouts if r["reward"] >= 0.3]

    if not candidates:
        print("[bootstrap] No usable rollouts found")
        return []

    # Sort by reward descending
    candidates.sort(key=lambda r: r["reward"], reverse=True)

    if diverse:
        # Pick top example from each category first, then fill remaining
        selected = []
        seen_categories = set()
        seen_tasks = set()

        # First pass: one per category
        for r in candidates:
            cat = r["category"]
            if cat not in seen_categories and len(selected) < top_k:
                selected.append(r)
                seen_categories.add(cat)
                seen_tasks.add(r["task_id"])

        # Second pass: fill remaining slots, prefer unseen tasks
        for r in candidates:
            if len(selected) >= top_k:
                break
            if r["task_id"] not in seen_tasks:
                selected.append(r)
                seen_tasks.add(r["task_id"])

        # Third pass: if still short, take whatever's best
        for r in candidates:
            if len(selected) >= top_k:
                break
            if r not in selected:
                selected.append(r)
    else:
        selected = candidates[:top_k]

    # Format for injection
    examples = []
    for r in selected:
        response = r["response"]
        if len(response) > max_response_len:
            response = response[:max_response_len] + "\n[... truncated for brevity]"
        examples.append({
            "task_prompt": r["prompt"],
            "response": response,
            "reward": r["reward"],
            "category": r["category"],
            "task_id": r["task_id"],
        })

    print(f"[bootstrap] Selected {len(examples)} examples "
          f"(rewards: {', '.join(f'{e['reward']:.3f}' for e in examples)})")
    return examples


def format_fewshot_block(examples: list[dict]) -> str:
    """Format examples as a markdown block for prompt injection."""
    if not examples:
        return ""

    lines = [
        "\n## Reference Examples",
        "Below are examples of high-quality responses. Use these as reference for "
        "the expected quality, structure, and depth — but adapt your response to "
        "each specific task.\n",
    ]

    for i, ex in enumerate(examples, 1):
        lines.append(f"### Example {i} (score: {ex['reward']:.2f}, {ex['category']})")
        lines.append(f"**Task:** {ex['task_prompt']}")
        lines.append(f"\n**Response:**\n{ex['response']}\n")
        lines.append("---\n")

    return "\n".join(lines)


def inject_fewshot(prompt: str, examples: list[dict]) -> str:
    """Inject few-shot examples into a system prompt.

    Appends the examples block at the end of the prompt, before any
    closing section if present.
    """
    if not examples:
        return prompt

    block = format_fewshot_block(examples)

    # If there's a clear closing section marker, inject before it
    closing_markers = ["## Final Notes", "## Remember", "---\n\nRemember"]
    for marker in closing_markers:
        if marker in prompt:
            idx = prompt.index(marker)
            return prompt[:idx] + block + "\n" + prompt[idx:]

    # Otherwise append at end
    return prompt + "\n" + block


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview few-shot bootstrap examples")
    parser.add_argument("--rollouts", type=str, nargs="+",
                        default=["experiments/agent-lightning/results/rollouts/rollouts_*.jsonl"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-reward", type=float, default=0.5)
    parser.add_argument("--show-prompt", action="store_true",
                        help="Show full augmented prompt")
    args = parser.parse_args()

    examples = bootstrap_examples(args.rollouts, top_k=args.top_k, min_reward=args.min_reward)

    if args.show_prompt:
        seed = Path("/home/ava/dev/protoResearcher/config/SOUL.md").read_text()
        augmented = inject_fewshot(seed, examples)
        print(augmented)
    else:
        block = format_fewshot_block(examples)
        print(block)
        print(f"\nBlock length: {len(block)} chars")
