#!/usr/bin/env python3
"""ThinkingCap-Qwen3.6-27B-HERETIC NVFP4 requant — abliterated variant.

Same recipe as `thinkingcap_27b_requant.py` (dense hybrid-GDN + VL + MTP) but on the
ABLITERATED base at /mnt/data/abliterate (the protoPen brain). Two differences:
  - MAIN weights = the local abliterated checkpoint (not the HF base).
  - The abliteration merge DROPPED the MTP head (0 mtp.* tensors), so the in-script
    graft (source_model=self) can't work. We quantize the main weights here, then graft
    the already-extracted head from the base NVFP4 in a separate post-step (graft_mtp.py) —
    no 54G base re-download. MTP drafts, target verifies → lossless even with the base head.
  - vision tower / GDN / lm_head / embed / mtp stay bf16 (no sm120 W4A4 kernel).

Run on the freed GPU0 window:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/ava/dev/quant-env/bin/python thinkingcap_27b_heretic_requant.py
"""
import os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MAIN = "/mnt/data/abliterate/ThinkingCap-Qwen3.6-27B-heretic"
PROC_SRC = "/mnt/models/quantized/ThinkingCap-Qwen3.6-27B-NVFP4"  # has processor/chat_template
OUT = "/mnt/models/quantized/ThinkingCap-Qwen3.6-27B-heretic-NVFP4"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048

with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MAIN, torch_dtype="auto", device_map="auto"
    )
processor = AutoProcessor.from_pretrained(PROC_SRC)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*embed_tokens$",
        "re:.*linear_attn.*",  # keep the GDN hybrid layers in bf16
        "re:.*mtp.*",          # (none present, but keep for parity)
    ],
)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.select_columns(["messages"]).shuffle(seed=42)

def preprocess_function(example):
    messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in example["messages"]]
    return processor.apply_chat_template(
        messages, return_tensors="pt", padding=False, truncation=True,
        max_length=MAX_SEQUENCE_LENGTH, tokenize=True, add_special_tokens=False,
        return_dict=True, add_generation_prompt=False)

ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)

def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES, data_collator=data_collator)

model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
print(f"saved main NVFP4 to {OUT}  (run graft_mtp.py next to add the MTP head)")
