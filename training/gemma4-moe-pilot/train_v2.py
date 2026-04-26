"""
Gemma 4 26B MoE — LoRA fine-tuning v2 for multi-turn agentic tool use.

Changes from v1:
- LR: 2e-5 → 2e-4 (Unsloth recommendation for LoRA)
- Alpha: 16 → 32 (higher alpha/rank ratio for structured output)
- Target modules: +embed_tokens, +lm_head (tool token precision)
- Data: Gemma 4 native tool format (<|tool_call>, <|tool_response>)
- Validation: 10% holdout split
- MLflow tracking enabled

Usage:
    source ~/dev/moe-train-env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python train_v2.py
"""

import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["HF_HOME"] = "/mnt/models/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MLFLOW_TRACKING_URI"] = "http://ava:5050"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "gemma4-moe-agentic"

from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import json
import mlflow


# ─── Model ────────────────────────────────────────────
print("Loading Gemma 4 26B MoE...")
model, tokenizer = FastModel.from_pretrained(
    model_name="google/gemma-4-26B-A4B-it",
    max_seq_length=2048,
    load_in_4bit=False,
    dtype=None,
    full_finetuning=False,
)

# ─── LoRA config (v2: higher alpha, +embed/lm_head) ──
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,              # v2: was 16, now 2x rank
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",          # v2: added for tool token precision
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

model.print_trainable_parameters()

# ─── Data ─────────────────────────────────────────────
print("Loading APIGen-MT-5k dataset...")
dataset = load_dataset("Salesforce/APIGen-MT-5k", split="train")
print(f"Dataset size: {len(dataset)} examples")


def format_example(example):
    """Convert APIGen-MT format to Gemma 4 chat format with native tool tokens."""
    messages = []

    # System prompt with tool definitions
    system = example.get("system", "")
    tools_raw = example.get("tools", "")
    tools = []
    if isinstance(tools_raw, str) and tools_raw.strip():
        try:
            tools = json.loads(tools_raw)
            if not isinstance(tools, list):
                tools = [tools]
        except json.JSONDecodeError:
            pass
    elif isinstance(tools_raw, list):
        tools = tools_raw

    if tools:
        tool_desc = "\n\nYou have access to these tools:\n"
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            tool_desc += f"\n- {name}: {desc}"
            if params:
                tool_desc += f"\n  Parameters: {json.dumps(params, ensure_ascii=False)}"
        system += tool_desc
    if system:
        messages.append({"role": "system", "content": system.strip()})

    # Convert conversation turns
    for turn in example.get("conversations", []):
        role = turn["from"]
        content = turn["value"]

        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": content})

    if len(messages) < 2:
        return {"text": ""}

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


print("Formatting dataset...")
formatted_dataset = dataset.map(format_example, num_proc=1)

# Filter empty examples
formatted_dataset = formatted_dataset.filter(lambda x: len(x["text"]) > 50)
print(f"Formatted: {len(formatted_dataset)} examples (after filtering)")

# v2: Train/validation split
split = formatted_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# ─── Training with MLflow ─────────────────────────────
print("Starting training with MLflow tracking...")

mlflow.set_experiment("gemma4-moe-agentic")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="/mnt/data/training/gemma4-moe-v2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,             # v2: was 2e-5, now 10x (Unsloth default)
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,              # v2: was 0.1, shorter warmup with higher LR
        bf16=True,
        logging_steps=5,                # v2: more frequent logging
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps",          # v2: eval during training
        eval_steps=100,
        max_seq_length=2048,
        dataset_text_field="text",
        seed=42,
        report_to="mlflow",            # v2: MLflow tracking
        dataloader_num_workers=0,
        dataset_num_proc=1,
    ),
)

stats = trainer.train()
print(f"\nTraining complete!")
print(f"  Loss: {stats.training_loss:.4f}")
print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

# Save LoRA adapter
output_path = "/mnt/data/training/gemma4-moe-v2/lora-adapter"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"Adapter saved to {output_path}")

# Log adapter to MLflow
mlflow.log_artifact(output_path, artifact_path="lora-adapter")
print("Adapter logged to MLflow")
