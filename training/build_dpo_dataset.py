#!/usr/bin/env python3
"""Build DPO dataset from WildBench inference results.

Takes "chosen" responses (from a strong model like GPT-5.4-mini) and
"rejected" responses (from the target model like Qwen3.5-9B) and formats
them for LLaMA-Factory DPO training.

Usage:
    python build_dpo_dataset.py \
        --chosen ../evals/results/wildbench/gpt-5.4-mini.json \
        --rejected ../evals/results/wildbench/local.json \
        --output dpo_wildbench_9b.json

Output format (LLaMA-Factory sharegpt_dpo):
[
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}  // multi-turn history
    ],
    "chosen": {"from": "gpt", "value": "chosen response"},
    "rejected": {"from": "gpt", "value": "rejected response"}
  }
]
"""

import json
import sys
from pathlib import Path

import click


@click.command()
@click.option("--chosen", required=True, type=click.Path(exists=True), help="JSON from strong model")
@click.option("--rejected", required=True, type=click.Path(exists=True), help="JSON from target model")
@click.option("--output", required=True, type=click.Path(), help="Output DPO dataset JSON")
@click.option("--min-chosen-len", default=50, help="Skip pairs where chosen is too short")
@click.option("--min-rejected-len", default=50, help="Skip pairs where rejected is too short")
def main(chosen, rejected, output, min_chosen_len, min_rejected_len):
    """Build DPO dataset from WildBench chosen/rejected pairs."""
    with open(chosen) as f:
        chosen_data = json.load(f)
    with open(rejected) as f:
        rejected_data = json.load(f)

    # Index by session_id
    chosen_map = {d["session_id"]: d for d in chosen_data}
    rejected_map = {d["session_id"]: d for d in rejected_data}

    common_ids = set(chosen_map.keys()) & set(rejected_map.keys())
    click.echo(f"Chosen: {len(chosen_data)} responses ({chosen_data[0].get('model', '?')})")
    click.echo(f"Rejected: {len(rejected_data)} responses ({rejected_data[0].get('model', '?')})")
    click.echo(f"Common session IDs: {len(common_ids)}")

    dataset = []
    skipped_chosen = 0
    skipped_rejected = 0
    skipped_error = 0

    for sid in sorted(common_ids):
        c = chosen_map[sid]
        r = rejected_map[sid]

        # Skip if either had errors
        if c.get("error") or r.get("error"):
            skipped_error += 1
            continue

        chosen_text = c.get("output", "")
        rejected_text = r.get("output", "")

        if len(chosen_text) < min_chosen_len:
            skipped_chosen += 1
            continue
        if len(rejected_text) < min_rejected_len:
            skipped_rejected += 1
            continue

        # Build conversation history (all turns except the last user message)
        messages = c["conversation_input"]
        conversations = []
        for msg in messages[:-1]:
            role = "human" if msg["role"] == "user" else "gpt"
            conversations.append({"from": role, "value": msg["content"]})

        # Last user message is the prompt for chosen/rejected
        last_user = messages[-1]
        conversations.append({"from": "human", "value": last_user["content"]})

        dataset.append({
            "conversations": conversations,
            "chosen": {"from": "gpt", "value": chosen_text},
            "rejected": {"from": "gpt", "value": rejected_text},
        })

    # Save
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    click.echo(f"\nDataset: {len(dataset)} DPO pairs")
    click.echo(f"Skipped: {skipped_error} errors, {skipped_chosen} short chosen, {skipped_rejected} short rejected")
    click.echo(f"Saved to: {output}")


if __name__ == "__main__":
    main()
