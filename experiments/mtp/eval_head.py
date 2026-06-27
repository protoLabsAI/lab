#!/usr/bin/env python
"""Offline acceptance *proxy* for an MTP head — localizes a regression without serving.

Computes, over a sample of the corpus, how often the head's top-1 next-next-token
prediction (under distill.py's training forward) matches the actual next-next token.
This is a greedy proxy for vLLM's acceptance rate.

Why it's the key diagnostic:
  - Run on the GRAFT checkpoint. If proxy ~= the graft's measured vLLM acceptance (0.763
    for ornith-9b), our training forward is parity-correct with serving -> any distill
    regression is pure optimization (fix: lower lr + early-stop).
  - If the graft's proxy != its vLLM acceptance, our forward has an op-level mismatch with
    vLLM's MTP serving forward -> fix the forward before training again.
  - Run on both graft and distilled to see which forward each is better under.

Usage (GPU; quant-env):
  python eval_head.py --config configs/ornith-9b.yaml --ckpt /mnt/data/checkpoints/ornith-9b-mtp-graft --n 200
  python eval_head.py --config configs/ornith-9b.yaml --ckpt /mnt/data/checkpoints/ornith-9b-mtp      --n 200
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import yaml

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill import MTPHead, build_examples, collate, find_base_parts, load_head_init  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint whose mtp.* head to evaluate")
    ap.add_argument("--n", type=int, default=200, help="number of corpus samples to score")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["distill"]
    device = "cuda"

    config = AutoConfig.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    pad_id = tokenizer.pad_token_id or config.get_text_config().pad_token_id

    base = AutoModelForCausalLM.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    ).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    text_model, lm_head = find_base_parts(base)

    head = MTPHead(config, d["full_attention_layer_idx"]).to(device=device, dtype=torch.float32).eval()
    load_head_init(head, args.ckpt)  # the head under test (graft init or distilled)

    exs = build_examples(cfg["corpus"]["out"], tokenizer, d["max_seq_len"], limit=args.n)
    shift = d.get("position_shift", 1)
    print(f"[eval_head] ckpt={args.ckpt}  scoring {len(exs)} samples (shift={shift})")

    match = total = 0
    with torch.no_grad():
        for i in range(len(exs)):
            input_ids, attn, comp_start = collate(exs[i:i+1], pad_id, device)
            base_hidden = text_model(input_ids=input_ids, attention_mask=attn,
                                     use_cache=False).last_hidden_state
            B, T, H = base_hidden.shape
            hid = base_hidden[:, : T - (shift + 1), :]
            next_ids = input_ids[:, shift : T - 1]
            labels = input_ids[:, shift + 1 :]
            mask = attn[:, shift : T - 1]
            col = torch.arange(T - (shift + 1), device=device).unsqueeze(0)
            comp_mask = (col >= (comp_start.unsqueeze(1) - shift)) & (mask.bool())

            out = head(text_model, next_ids, hid.to(torch.float32), mask)
            logits = F.linear(out, lm_head.weight.to(out.dtype))
            pred = logits.argmax(-1)
            m = comp_mask
            match += int((pred[m] == labels[m]).sum())
            total += int(m.sum())

    proxy = match / total if total else 0.0
    print(f"\n[eval_head] greedy acceptance PROXY = {proxy:.3f}  ({match}/{total} next-next tokens)")
    print("[eval_head] compare to the head's measured vLLM acceptance to localize the issue "
          "(see eval_head.py docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
