"""DPO training for Qwen 3.5 9B on protoResearcher rollouts.

Converts collected rollouts into preference pairs (chosen=high-reward,
rejected=low-reward) and trains with TRL DPOTrainer + LoRA.

This is the offline alternative to online GRPO — it doesn't need a live
reward function or vLLM server. Just rollouts with rewards.

Pair construction strategy:
  - Within-task pairs: same task_id, reward spread >= threshold
  - Chosen: higher reward response, Rejected: lower reward response
  - Optionally augmented with cross-task pairs within same category

Setup:
    source ~/dev/quant-env/bin/activate  # transformers 5.5, TRL 1.0

    # Dry run (validate data + config, no GPU needed)
    python experiments/agent-lightning/train_dpo.py \
      --rollouts experiments/agent-lightning/results/rollouts/rollouts_*.jsonl \
      --dry-run

    # Full training on GPU 1 (GPU 0 can stay serving vLLM)
    CUDA_VISIBLE_DEVICES=1 python experiments/agent-lightning/train_dpo.py \
      --rollouts experiments/agent-lightning/results/rollouts/rollouts_*.jsonl

    # With cross-task pairs for more data
    CUDA_VISIBLE_DEVICES=1 python experiments/agent-lightning/train_dpo.py \
      --rollouts experiments/agent-lightning/results/rollouts/rollouts_*.jsonl \
      --cross-task
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "/mnt/data/training/researcher/dpo-9b"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MIN_REWARD_SPREAD = 0.1


def load_rollouts(paths: list[str]) -> list[dict]:
    """Load rollout JSONL files."""
    records = []
    for path in paths:
        for fpath in glob(path):
            with open(fpath) as f:
                for line in f:
                    records.append(json.loads(line))
    return records


def build_preference_pairs(
    rollouts: list[dict],
    min_spread: float = MIN_REWARD_SPREAD,
    cross_task: bool = False,
) -> list[dict]:
    """Convert rollouts into DPO preference pairs.

    Within-task pairs: for each task_id, pair high-reward vs low-reward responses.
    Cross-task pairs: within same category, pair high vs low (weaker signal but more data).
    """
    pairs = []

    # Within-task pairs (strongest signal)
    by_task = defaultdict(list)
    for r in rollouts:
        by_task[r["task_id"]].append(r)

    for task_id, task_rollouts in by_task.items():
        # Sort by reward descending
        sorted_rollouts = sorted(task_rollouts, key=lambda x: x["reward"], reverse=True)

        for i in range(len(sorted_rollouts)):
            for j in range(i + 1, len(sorted_rollouts)):
                chosen = sorted_rollouts[i]
                rejected = sorted_rollouts[j]

                spread = chosen["reward"] - rejected["reward"]
                if spread < min_spread:
                    continue

                # Skip if either is zero-reward (tool failure, not preference signal)
                if chosen["reward"] <= 0.0:
                    continue

                pairs.append({
                    "prompt": chosen["prompt"],
                    "chosen": chosen["response"],
                    "rejected": rejected["response"],
                    "chosen_reward": chosen["reward"],
                    "rejected_reward": rejected["reward"],
                    "spread": spread,
                    "task_id": task_id,
                    "category": chosen["category"],
                    "pair_type": "within_task",
                })

    # Cross-task pairs (weaker but more data)
    if cross_task:
        by_category = defaultdict(list)
        for r in rollouts:
            if r["reward"] > 0.0:  # Skip tool failures
                by_category[r["category"]].append(r)

        for category, cat_rollouts in by_category.items():
            sorted_cat = sorted(cat_rollouts, key=lambda x: x["reward"], reverse=True)
            # Top quartile vs bottom quartile
            n = len(sorted_cat)
            if n < 4:
                continue
            top = sorted_cat[: n // 4]
            bottom = sorted_cat[-(n // 4):]

            for chosen in top:
                for rejected in bottom:
                    if chosen["task_id"] == rejected["task_id"]:
                        continue  # Already covered by within-task
                    spread = chosen["reward"] - rejected["reward"]
                    if spread < min_spread:
                        continue
                    pairs.append({
                        "prompt": chosen["prompt"],  # Use chosen's task prompt
                        "chosen": chosen["response"],
                        "rejected": rejected["response"],
                        "chosen_reward": chosen["reward"],
                        "rejected_reward": rejected["reward"],
                        "spread": spread,
                        "task_id": f"{chosen['task_id']}_vs_{rejected['task_id']}",
                        "category": category,
                        "pair_type": "cross_task",
                    })

    return pairs


def pairs_to_dataset(pairs: list[dict]) -> Dataset:
    """Convert preference pairs to HuggingFace Dataset for DPOTrainer.

    DPOTrainer expects columns: prompt, chosen, rejected.
    The prompt is formatted as a chat message for the tokenizer.
    """
    records = []
    for p in pairs:
        # Format as chat messages (DPOTrainer with chat template)
        records.append({
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="DPO training for Qwen 9B from rollouts")
    parser.add_argument("--rollouts", type=str, nargs="+", required=True,
                        help="Glob patterns for rollout JSONL files")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (KL penalty). Lower = more aggressive preference learning")
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length for prompt + response")
    parser.add_argument("--min-spread", type=float, default=MIN_REWARD_SPREAD,
                        help="Minimum reward spread for pair construction")
    parser.add_argument("--cross-task", action="store_true",
                        help="Include cross-task pairs within same category")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and config but don't train")
    args = parser.parse_args()

    # Load and build pairs
    rollouts = load_rollouts(args.rollouts)
    print(f"Loaded {len(rollouts)} rollouts")

    pairs = build_preference_pairs(rollouts, min_spread=args.min_spread, cross_task=args.cross_task)
    print(f"Built {len(pairs)} preference pairs (min spread: {args.min_spread})")

    if not pairs:
        print("No viable preference pairs. Need more rollouts or lower --min-spread.")
        return

    # Stats
    within = [p for p in pairs if p["pair_type"] == "within_task"]
    cross = [p for p in pairs if p["pair_type"] == "cross_task"]
    print(f"  Within-task: {len(within)}")
    if cross:
        print(f"  Cross-task:  {len(cross)}")

    spreads = [p["spread"] for p in pairs]
    print(f"  Spread range: [{min(spreads):.3f}, {max(spreads):.3f}]")
    print(f"  Avg spread:   {sum(spreads)/len(spreads):.3f}")

    by_cat = defaultdict(int)
    for p in pairs:
        by_cat[p["category"]] += 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count} pairs")
    print()

    # Build dataset
    dataset = pairs_to_dataset(pairs)
    print(f"Dataset: {len(dataset)} examples")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print()

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # DPO config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        # Training settings
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_steps=25,
        save_total_limit=3,
        warmup_steps=2,
        report_to="none",
        seed=42,
        # DPO-specific
        loss_type="sigmoid",  # Standard DPO loss
        remove_unused_columns=False,
    )

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — config validated, not training.")
        print(f"  Pairs: {len(pairs)} ({len(within)} within-task, {len(cross)} cross-task)")
        print(f"  LoRA rank: {args.lora_r}, alpha: {LORA_ALPHA}")
        print(f"  Batch: {args.batch_size} x {training_args.gradient_accumulation_steps} grad_accum")
        print(f"  Effective batch: {args.batch_size * training_args.gradient_accumulation_steps}")
        print(f"  LR: {args.lr}, beta: {args.beta}, epochs: {args.epochs}")
        print(f"  Max length: {args.max_length}")
        print(f"  Loss: {training_args.loss_type}")
        print("=" * 60)

        # Preview a pair
        if pairs:
            p = pairs[0]
            print(f"\nSample pair ({p['pair_type']}, {p['category']}):")
            print(f"  Task: {p['task_id']}")
            print(f"  Prompt: {p['prompt'][:100]}...")
            print(f"  Chosen reward:   {p['chosen_reward']:.3f} ({len(p['chosen'])} chars)")
            print(f"  Rejected reward: {p['rejected_reward']:.3f} ({len(p['rejected'])} chars)")
            print(f"  Spread: {p['spread']:.3f}")
        return

    # Load tokenizer (needed for DPO to format chat templates)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize trainer
    trainer = DPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save final adapter
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete. Adapter saved to {args.output_dir}")

    # Save training metadata
    meta = {
        "model": args.model,
        "pairs": len(pairs),
        "within_task": len(within),
        "cross_task": len(cross),
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "loss_type": training_args.loss_type,
        "avg_spread": round(sum(spreads) / len(spreads), 3),
    }
    meta_path = Path(args.output_dir) / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
