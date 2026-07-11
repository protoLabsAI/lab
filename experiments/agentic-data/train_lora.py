"""LoRA-SFT a Qwen3.5 student on the agentic-distill corpus (transformers Trainer + peft).

  ~/dev/quant-env/bin/python train_lora.py --data <jsonl> --out <dir> [--subset N --smoke]

Assistant-masked loss done EXPLICITLY (Qwen3.5's chat template has no `{% generation %}`
block, so TRL's assistant_only_loss can't work). We tokenize each example with the real
template — so train format == inference format — then mask every token that isn't inside an
assistant turn, via the incremental-prefix method. Full-sequence loss on multi-turn agentic
data degraded the base (arm-A v1: claw 0.642 -> 0.419); masking is the fix.

Hybrid-GDN VL model: LoRA targets MLP + standard-attention projections only (not the GDN
linear-attn internals). Trajectories are flattened (tool_calls -> text) before templating.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, Qwen3_5ForConditionalGeneration,
                          Trainer, TrainingArguments)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _flatten(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        c = m.get("content") or ""
        if m.get("tool_calls"):
            c = (c + " " + " ".join(json.dumps(t) for t in m["tool_calls"])).strip()
        if c.strip() or m["role"] == "assistant":
            out.append({"role": m["role"], "content": c})
    return out


_HDR = {}  # cache role-header token ids per tokenizer


def _asst_header(tok) -> list[int]:
    if "h" not in _HDR:
        _HDR["im_start"] = tok.convert_tokens_to_ids("<|im_start|>")
        _HDR["im_end"] = tok.convert_tokens_to_ids("<|im_end|>")
        _HDR["h"] = tok.encode("assistant\n", add_special_tokens=False)
    return _HDR["h"]


def build_example(messages: list[dict], tok, max_len: int) -> dict | None:
    """Tokenize the full conversation once; label only the tokens inside each assistant
    turn — from just after the `<|im_start|>assistant\n` header to its `<|im_end|>`."""
    try:
        full = tok.apply_chat_template(messages, tokenize=True)
    except Exception:
        return None
    if not isinstance(full, list):  # transformers 5.x may return a BatchEncoding
        full = full["input_ids"]
    if full and not isinstance(full[0], int):  # nested [[...]] or tensor
        full = list(full[0]) if not hasattr(full, "tolist") else full.tolist()
    hdr = _asst_header(tok)
    im_start, im_end = _HDR["im_start"], _HDR["im_end"]
    labels = [-100] * len(full)
    p = 0
    while p < len(full):
        if full[p] == im_start and full[p + 1: p + 1 + len(hdr)] == hdr:
            start = p + 1 + len(hdr)
            q = start
            while q < len(full) and full[q] != im_end:
                q += 1
            for j in range(start, min(q + 1, len(full))):  # content + closing <|im_end|>
                labels[j] = full[j]
            p = q + 1
        else:
            p += 1
    if not any(l != -100 for l in labels):
        return None
    return {"input_ids": full[:max_len], "labels": labels[:max_len],
            "attention_mask": [1] * min(len(full), max_len)}


def load_dataset_masked(path: str, subset: int | None, tok, max_len: int) -> Dataset:
    import random
    rows = [json.loads(l) for l in open(path)]
    random.Random(0).shuffle(rows)
    if subset:
        rows = rows[:subset]
    out = []
    for r in rows:
        ex = build_example(_flatten(r["messages"]), tok, max_len)
        if ex and any(l != -100 for l in ex["labels"]):
            out.append(ex)
    print(f"tokenized {len(out)}/{len(rows)} examples with ≥1 assistant span")
    return Dataset.from_list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    ds = load_dataset_masked(a.data, a.subset or (200 if a.smoke else None), tok, a.max_seq)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
        target_modules=TARGETS, task_type="CAUSAL_LM", bias="none"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out,
        per_device_train_batch_size=1 if a.smoke else a.bs,
        gradient_accumulation_steps=1 if a.smoke else a.grad_accum,
        num_train_epochs=a.epochs, max_steps=15 if a.smoke else -1,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="no" if a.smoke else "epoch",
        bf16=True, gradient_checkpointing=True, report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False})

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    if not a.smoke:
        trainer.save_model(a.out)
        print(f"saved LoRA adapter -> {a.out}")
    else:
        print("SMOKE OK — masked training loop ran end-to-end")


if __name__ == "__main__":
    main()
