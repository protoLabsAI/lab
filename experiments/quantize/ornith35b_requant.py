#!/usr/bin/env python3
"""Agents-A1 NVFP4 requant — aligned to llm-compressor main's official
qwen3_5 W4A4 example (post-#2848 MoE loading fix; the 0.10.1.dev48 build we
first used saved sequential per-expert layout that nothing could load).

Needs the dual-GPU window (stop vllm + vllm-replica-b + embed-b first):
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python a1_requant.py
"""

import os

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "deepreinforce-ai/Ornith-1.0-35B"
OUT = "/mnt/models/quantized/Ornith-1.0-35B-NVFP4"
NUM_CALIBRATION_SAMPLES = 128  # window budget: 256@4096 (official) ≈ 4x longer
MAX_SEQUENCE_LENGTH = 2048

with load_context(Qwen3_5MoeForConditionalGeneration):
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
        "re:.*mtp.*",
    ],
)

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]"
)
ds = ds.select_columns(["messages"]).shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
# A1 stripped its mtp.* — graft the draft head from the same-generation base
# via the official util (replaces our manual sidecar for this artifact)
save_mtp_tensors_to_checkpoint(source_model="Qwen/Qwen3.5-35B-A3B", dest_dir=OUT)
print(f"saved to {OUT}")
# post-steps (from /quant-release): fix_vl_keys.py (verify still needed on new
# version), aux-config copy, MTP sidecar install, expert-format verification
# (expect FUSED experts.gate_up_proj-style keys — the whole point of this redo)
