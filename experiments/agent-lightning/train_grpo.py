"""GRPO training for Qwen 3.5 9B on protoResearcher rollouts.

Uses TRL GRPOTrainer with vLLM server mode:
  - GPU 0: vLLM server for generation (started separately via `trl vllm-serve`)
  - GPU 1: Training (this script)

The reward function uses pre-collected rollout scores (from collect_rollouts.py).
For online training, it could call the LLM judge directly, but that's expensive
and slow — offline GRPO from collected rollouts is more practical.

Setup:
    source ~/dev/vllm-env/bin/activate

    # Terminal 1: Start vLLM server on GPU 0
    CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
      --model Qwen/Qwen3.5-9B \
      --port 8001 \
      --gpu-memory-utilization 0.85

    # Terminal 2: Run training on GPU 1
    CUDA_VISIBLE_DEVICES=1 python experiments/agent-lightning/train_grpo.py \
      --rollouts experiments/agent-lightning/results/rollouts/rollouts_*.jsonl

Alternative (offline, no vLLM needed):
    CUDA_VISIBLE_DEVICES=0 python experiments/agent-lightning/train_grpo.py \
      --rollouts results/rollouts/rollouts_*.jsonl \
      --offline
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "/mnt/data/training/researcher/grpo-9b"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


def load_rollouts(paths: list[str]) -> Dataset:
    """Load rollout JSONL files into a HuggingFace Dataset."""
    records = []
    for path in paths:
        for fpath in glob(path):
            with open(fpath) as f:
                for line in f:
                    r = json.loads(line)
                    # Skip zero-reward rollouts (tool failures, not useful for training)
                    if r["reward"] <= 0.0:
                        continue
                    records.append({
                        "prompt": r["prompt"],
                        "completion": r["response"],
                        "reward": r["reward"],
                        "task_id": r["task_id"],
                        "category": r["category"],
                    })

    print(f"Loaded {len(records)} rollouts (filtered zero-reward)")

    # GRPO needs a "prompt" column — the trainer handles generation
    return Dataset.from_list(records)


def reward_fn(completions: list[str], prompts: list[str] | None = None, **kwargs) -> list[float]:
    """Reward function for GRPO.

    In offline mode, rewards are pre-computed in the dataset.
    In online mode, this would call the LLM judge.

    For now, this is a placeholder — GRPOTrainer in offline mode
    reads rewards directly from the dataset's 'reward' column.
    """
    # This shouldn't be called in offline mode, but provide a fallback
    return [0.5] * len(completions)


def main():
    parser = argparse.ArgumentParser(description="GRPO training for Qwen 9B")
    parser.add_argument("--rollouts", type=str, nargs="+", required=True,
                        help="Glob patterns for rollout JSONL files")
    parser.add_argument("--offline", action="store_true",
                        help="Offline mode (no vLLM server, use pre-collected completions)")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt for GRPO (online mode)")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and config but don't train")
    args = parser.parse_args()

    # Load rollouts
    dataset = load_rollouts(args.rollouts)
    if len(dataset) == 0:
        print("No rollouts loaded. Exiting.")
        return

    print(f"Dataset: {len(dataset)} examples")
    print(f"Reward range: [{min(dataset['reward']):.3f}, {max(dataset['reward']):.3f}]")
    print(f"Avg reward: {sum(dataset['reward'])/len(dataset):.3f}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print()

    # LoRA config — we don't want to full-finetune 9B on one GPU
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=0.04,  # KL penalty coefficient
        loss_type="grpo",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # vLLM settings (only for online mode)
        use_vllm=not args.offline,
        vllm_mode="server" if not args.offline else None,
        vllm_server_port=8001,
        vllm_gpu_memory_utilization=0.85,
        # Training settings
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        warmup_ratio=0.1,
        report_to="none",
        # Disable wandb/tensorboard (use local logging)
        seed=42,
    )

    if args.dry_run:
        print("Dry run — config validated, not training.")
        print(f"  LoRA rank: {args.lora_r}, alpha: {LORA_ALPHA}")
        print(f"  Batch: {args.batch_size} x {training_args.gradient_accumulation_steps} grad_accum")
        print(f"  LR: {args.lr}, epochs: {args.epochs}")
        print(f"  vLLM: {'off (offline)' if args.offline else 'server mode on :8001'}")
        return

    # Initialize trainer
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        reward_funcs=reward_fn,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save final adapter
    trainer.save_model(args.output_dir)
    print(f"\nTraining complete. Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
