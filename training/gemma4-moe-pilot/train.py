"""
Gemma 4 26B MoE — LoRA fine-tuning pilot for multi-turn agentic tool use.

Target: Improve multi-turn function calling (BFCL multi_turn: 6% → 50%+)
Method: bf16 LoRA on Unsloth (NOT QLoRA — bitsandbytes can't handle MoE fused tensors)
Hardware: Single RTX PRO 6000 Blackwell (96GB VRAM)
Data: Salesforce APIGen-MT-5k (multi-turn tool use, 99% human-verified)

Usage:
    source ~/dev/moe-train-env/bin/activate
    CUDA_VISIBLE_DEVICES=0 python train.py
"""

import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # Prevent mixed bf16/fp32 crash
os.environ["HF_HOME"] = "/mnt/models/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import json


# ─── Model ────────────────────────────────────────────
print("Loading Gemma 4 26B MoE...")
model, tokenizer = FastModel.from_pretrained(
    model_name="google/gemma-4-26B-A4B-it",
    max_seq_length=2048,
    load_in_4bit=False,       # bf16 LoRA, NOT QLoRA
    dtype=None,               # auto-detect (bf16)
    full_finetuning=False,
)

# ─── LoRA config ──────────────────────────────────────
# Target attention + expert FFN layers, freeze router
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # expert FFN
    ],
    # Router (gate) is NOT targeted — frozen by default in Unsloth
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Print trainable params
model.print_trainable_parameters()

# ─── Data ─────────────────────────────────────────────
print("Loading APIGen-MT-5k dataset...")
dataset = load_dataset("Salesforce/APIGen-MT-5k", split="train")
print(f"Dataset size: {len(dataset)} examples")

# Format into chat messages for Gemma 4
def format_example(example):
    """Convert APIGen-MT format to Gemma 4 chat format.

    APIGen-MT format:
    - system: string (system prompt)
    - tools: list of dicts (name, description, parameters)
    - conversations: list of {from: human|assistant|tool, value: string}
    """
    messages = []

    # System prompt with tool definitions
    system = example.get("system", "")
    tools_raw = example.get("tools", "")
    # tools is a JSON string containing a list of tool dicts
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
            tool_desc += f"\n- {tool.get('name', 'unknown')}: {tool.get('description', '')}"
            params = tool.get("parameters", {})
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
            # Tool results go as user messages (Gemma 4 convention)
            messages.append({"role": "user", "content": content})

    if len(messages) < 2:
        return {"text": ""}

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

print("Formatting dataset...")
formatted_dataset = dataset.map(format_example, num_proc=1)

# ─── Training ─────────────────────────────────────────
print("Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    args=SFTConfig(
        output_dir="/mnt/data/training/gemma4-moe-pilot",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        max_seq_length=2048,
        dataset_text_field="text",
        seed=42,
        report_to="none",
        dataloader_num_workers=0,    # prevent tokenizer fork deadlock
        dataset_num_proc=1,
    ),
)

stats = trainer.train()
print(f"\nTraining complete!")
print(f"  Loss: {stats.training_loss:.4f}")
print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

# Save LoRA adapter
model.save_pretrained("/mnt/data/training/gemma4-moe-pilot/lora-adapter")
tokenizer.save_pretrained("/mnt/data/training/gemma4-moe-pilot/lora-adapter")
print(f"Adapter saved to /mnt/data/training/gemma4-moe-pilot/lora-adapter")
