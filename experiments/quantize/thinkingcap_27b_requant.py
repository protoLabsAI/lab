#!/usr/bin/env python3
"""ThinkingCap-Qwen3.6-27B NVFP4 requant — dense hybrid-GDN + VL + MTP.

BottleCapAI's brevity-finetune of Qwen3.6-27B. Same model class as our Qwen3.5-2B
requant (`Qwen3_5ForConditionalGeneration`: dense text backbone + vision tower + GDN
linear-attention), just 27B and — unlike the 2B — it SHIPS an MTP head bundled in
`model-base-aux.safetensors` (mtp.fc + mtp.layers.0.*). So vs the 2B recipe:
  - keep `re:.*mtp.*` in the ignore list (stays bf16) AND graft it into the output
    via save_mtp_tensors_to_checkpoint(source_model=<ThinkingCap itself>) — the head
    is already here, no same-generation base download needed.
  - vision tower / GDN / lm_head / embed stay bf16 (VL + DeltaNet have no sm120 W4A4 kernel).

Run on the freed GPU1 window (GPU0 is busy generating):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python thinkingcap_27b_requant.py
"""

import os

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "bottlecapai/ThinkingCap-Qwen3.6-27B"
OUT = "/mnt/models/quantized/ThinkingCap-Qwen3.6-27B-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
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
        "re:.*embed_tokens$",
        "re:.*linear_attn.*",  # keep the GDN hybrid layers in bf16
        "re:.*mtp.*",          # keep the bundled draft head in bf16 (grafted below)
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
    data_collator=data_collator,
)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
# ThinkingCap ships its own mtp.* (model-base-aux.safetensors) — graft it into the
# quantized artifact (bf16 draft head + merged index) so the sidecar travels.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=OUT)
print(f"saved to {OUT}")
