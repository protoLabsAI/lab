"""
Gemma 4 26B MoE — LoRA fine-tuning v3: Full multi-turn agentic dataset.

Datasets:
- NVIDIA Nemotron RL Agentic (97K) — multi-turn with reward signals
- Salesforce APIGen-MT-5k (5K) — gold-standard multi-turn tool use
- ToolACE-Qwen-cleaned (10.5K) — diverse API coverage
- NousResearch Hermes FC v1 (1.9K multi-turn subset)

Config (from community research):
- LR: 2e-4 (Unsloth MoE recommendation)
- LoRA: r=16, alpha=32, +embed_tokens/lm_head
- Router frozen, bf16, gradient checkpointing
- MLflow tracking at ava:5050

Usage:
    source ~/dev/moe-train-env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python train_v3.py [--max-samples N] [--epochs N]
"""

import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_HOME"] = "/mnt/models/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MLFLOW_TRACKING_URI"] = "http://ava:5050"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "gemma4-moe-agentic"

import argparse
import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
import mlflow


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples", type=int, default=None, help="Cap total training examples")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    return p.parse_args()


# ─── Dataset Formatters ──────────────────────────────

def format_apigen(example, tokenizer):
    """APIGen-MT: system + tools string + conversations (from/value)."""
    messages = []
    system = example.get("system", "")
    tools_raw = example.get("tools", "")
    tools = []
    if isinstance(tools_raw, str) and tools_raw.strip():
        try:
            parsed = json.loads(tools_raw)
            tools = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            pass

    if tools:
        tool_desc = "\n\nAvailable tools:\n"
        for t in tools:
            tool_desc += f"\n- {t.get('name','?')}: {t.get('description','')}"
        system += tool_desc
    if system:
        messages.append({"role": "system", "content": system.strip()})

    for turn in example.get("conversations", []):
        role = turn["from"]
        content = turn["value"]
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": content})

    if len(messages) < 3:  # need at least system + user + assistant
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def format_toolace(example, tokenizer):
    """ToolACE-Qwen: JSON string conversations (role/content) + tools."""
    try:
        convs = json.loads(example["conversations"])
        tools = json.loads(example["tools"])
    except (json.JSONDecodeError, TypeError):
        return None

    messages = []
    if tools:
        tool_desc = "Available tools:\n"
        for t in tools:
            tool_desc += f"\n- {t.get('name','?')}: {t.get('description','')}"
        messages.append({"role": "system", "content": tool_desc})

    has_tool_turn = False
    for turn in convs:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if content is None:
            content = ""
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            has_tool_turn = True
            messages.append({"role": "user", "content": f"[Tool Result]: {content}"})

    if not has_tool_turn or len(messages) < 4:
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def format_hermes(example, tokenizer):
    """Hermes FC v1: conversations (from/value), tools as string."""
    convs = example.get("conversations", [])
    messages = []
    has_tool = False

    for turn in convs:
        role = turn.get("from", "")
        content = turn.get("value", "")
        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            has_tool = True
            messages.append({"role": "user", "content": f"[Tool Result]: {content}"})

    if not has_tool or len(messages) < 4:
        return None
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def format_nemotron(example, tokenizer):
    """Nemotron RL: OpenAI Responses API format with reasoning/function_call/function_call_output items."""
    rcp = example.get("responses_create_params", {})
    input_items = rcp.get("input", [])
    expected = example.get("expected_action", {})

    if not input_items or not expected:
        return None

    messages = []
    has_tool = False

    for item in input_items:
        item_type = item.get("type", item.get("role", ""))

        if item_type == "system":
            messages.append({"role": "system", "content": item.get("content", "")})
        elif item_type == "user":
            messages.append({"role": "user", "content": item.get("content", "")})
        elif item_type == "message" and item.get("role") == "assistant":
            # Assistant text message
            content = item.get("content", "")
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "output_text"]
                content = "\n".join(parts)
            if content:
                messages.append({"role": "assistant", "content": str(content)})
        elif item_type == "reasoning":
            # Skip reasoning traces — we want the model to learn to produce them, not memorize them
            pass
        elif item_type == "function_call":
            # Tool call by assistant
            name = item.get("name", "unknown")
            args = item.get("arguments", "{}")
            messages.append({"role": "assistant", "content": f"[Calling {name}({args})]"})
        elif item_type == "function_call_output":
            # Tool result
            has_tool = True
            output = item.get("output", "")
            messages.append({"role": "user", "content": f"[Tool Result]: {output}"})

    # Add expected action as final assistant turn
    exp_type = expected.get("type", "")
    exp_content = expected.get("content", "")
    if exp_type == "function_call":
        name = expected.get("name", "unknown")
        args = expected.get("arguments", "{}")
        messages.append({"role": "assistant", "content": f"[Calling {name}({args})]"})
    elif exp_content:
        messages.append({"role": "assistant", "content": exp_content})

    if not has_tool or len(messages) < 4:
        return None

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


# ─── Main ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("Loading Gemma 4 26B MoE...")
    model, tokenizer = FastModel.from_pretrained(
        model_name="google/gemma-4-26B-A4B-it",
        max_seq_length=2048,
        load_in_4bit=False,
        dtype=None,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        use_gradient_checkpointing="unsloth", random_state=42,
    )
    model.print_trainable_parameters()

    # ─── Load and format all datasets ─────────────────
    all_texts = []

    print("\n1. Loading APIGen-MT-5k...")
    ds = load_dataset("Salesforce/APIGen-MT-5k", split="train")
    count = 0
    for ex in ds:
        text = format_apigen(ex, tokenizer)
        if text and len(text) > 100:
            all_texts.append(text)
            count += 1
    print(f"   APIGen: {count} multi-turn examples")

    print("2. Loading ToolACE-Qwen-cleaned...")
    ds = load_dataset("tryumanshow/ToolACE-Qwen-cleaned", split="train")
    count = 0
    for ex in ds:
        text = format_toolace(ex, tokenizer)
        if text and len(text) > 100:
            all_texts.append(text)
            count += 1
    print(f"   ToolACE: {count} multi-turn examples")

    # Hermes FC v1 skipped — all single-turn, no multi-turn tool use

    print("3. Loading NVIDIA Nemotron RL (filtering multi-turn with tools)...")
    ds = load_dataset("nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1", split="train")
    count = 0
    for ex in ds:
        text = format_nemotron(ex, tokenizer)
        if text and len(text) > 100:
            all_texts.append(text)
            count += 1
    print(f"   Nemotron: {count} multi-turn examples")

    # Shuffle
    random.seed(42)
    random.shuffle(all_texts)

    # Cap if requested
    if args.max_samples and len(all_texts) > args.max_samples:
        all_texts = all_texts[:args.max_samples]

    print(f"\nTotal training examples: {len(all_texts)}")

    # Create dataset
    full_dataset = Dataset.from_dict({"text": all_texts})

    # Train/eval split
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ─── Training ─────────────────────────────────────
    num_steps = (len(train_dataset) * args.epochs) // 4  # batch_size=1, grad_accum=4
    print(f"\nStarting training: {num_steps} steps, lr={args.lr}, r={args.rank}, alpha={args.alpha}")

    mlflow.set_experiment("gemma4-moe-agentic")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir="/mnt/data/training/gemma4-moe-v3",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=250,
            max_seq_length=2048,
            dataset_text_field="text",
            seed=42,
            report_to="mlflow",
            dataloader_num_workers=0,
            dataset_num_proc=1,
        ),
    )

    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

    output_path = "/mnt/data/training/gemma4-moe-v3/lora-adapter"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Adapter saved to {output_path}")


if __name__ == "__main__":
    main()
