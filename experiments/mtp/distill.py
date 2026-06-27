#!/usr/bin/env python
"""Distill a grafted Qwen3.5 MTP head onto a fine-tune's own hidden states.

The grafted head (Qwen's) loads on the fine-tune but drafts poorly: it was co-trained
against base-Qwen's residual stream, which the fine-tune moved. We freeze the base and
train ONLY the 15 mtp.* tensors so the head re-aligns to the fine-tune's hidden states.
Quality is never at risk (the base verifies every speculated token); we are buying back
the acceptance rate that makes the speedup real.

Objective (matches vLLM's serving forward, model_executor/models/qwen3_5_mtp.py):
  for position t:
    e = pre_fc_norm_embedding( embed(tok_{t+1}) )
    h = pre_fc_norm_hidden( base_post_norm_hidden_t )
    x = fc( cat([e, h]) )                 # 2H -> H
    x = mtp_decoder_layer(x)              # one full_attention layer (rope + causal)
    x = norm(x)
    logits = lm_head(x)                   # frozen, shared
    loss   += CE(logits, tok_{t+2})

The head's decoder layer / rope / causal mask reuse the BASE model's own HF modules, so
the trained head matches the served forward. Run on GPU; use --smoke for a 1-step check.

Env: quant-env (transformers 5.5, per the venv rule -- do NOT train in vllm-env).
GPU: needs ~20 GB for the frozen bf16 base + activations; free GPU1 first
     (stop vllm-replica-b -- see experiments/mtp/README.md).

Usage:
  python distill.py --config configs/ornith-9b.yaml            # full run
  python distill.py --config configs/ornith-9b.yaml --smoke    # 1 batch fwd/bwd sanity
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file, save_file

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    create_causal_mask,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mtp_lib import MTP_TENSORS  # noqa: E402

MTP_SHARD = "model-mtp.safetensors"


# --------------------------------------------------------------------------- head
class MTPHead(nn.Module):
    """One Qwen3.5 MTP layer, built from the base's own HF modules for serving parity.

    Holds exactly the 15 mtp.* params (decoder layer + fc + 3 norms). embed_tokens,
    lm_head, rotary and the causal-mask builder are borrowed from the frozen base at
    forward time -- never trained, never duplicated into this module's state_dict.
    """

    def __init__(self, config, full_attention_layer_idx: int):
        super().__init__()
        text_cfg = config.get_text_config()
        h, eps = text_cfg.hidden_size, text_cfg.rms_norm_eps
        assert text_cfg.layer_types[full_attention_layer_idx] == "full_attention", (
            f"layer_idx {full_attention_layer_idx} is not a full_attention slot"
        )
        # standalone decoder layer needs an explicit attn impl (else dispatch warns/None)
        if getattr(text_cfg, "_attn_implementation", None) in (None, "None"):
            text_cfg._attn_implementation = "sdpa"
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(text_cfg, full_attention_layer_idx)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)

    def forward(self, base, next_ids, base_hidden, attention_mask):
        e = self.pre_fc_norm_embedding(base.embed_tokens(next_ids))
        hh = self.pre_fc_norm_hidden(base_hidden)
        x = self.fc(torch.cat([e, hh], dim=-1))

        B, T = next_ids.shape
        dev = x.device
        pos = torch.arange(T, device=dev).view(1, 1, -1).expand(4, B, -1)
        text_pos, rope_pos = pos[0], pos[1:]
        position_embeddings = base.rotary_emb(x, rope_pos)
        causal = create_causal_mask(
            config=base.config, inputs_embeds=x, attention_mask=attention_mask,
            past_key_values=None, position_ids=text_pos,
        )
        x = self.layers[0](
            x, position_embeddings=position_embeddings, attention_mask=causal,
            position_ids=text_pos, use_cache=False,
        )
        return self.norm(x)

    def export_state_dict(self, dtype) -> dict[str, torch.Tensor]:
        """Map this module's params -> on-disk mtp.* names, in `dtype`."""
        sd = self.state_dict()
        # nn.Linear / module names already line up with mtp.* under an "mtp." prefix.
        out = {f"mtp.{k}": v.to(dtype).contiguous().cpu() for k, v in sd.items()}
        assert sorted(out) == sorted(MTP_TENSORS), (
            f"export mismatch:\n got {sorted(out)}\n exp {sorted(MTP_TENSORS)}"
        )
        return out


def load_head_init(head: MTPHead, init_ckpt: str) -> None:
    """Load grafted mtp.* tensors (Qwen init) into the head module."""
    head_sd = {}
    idx = json.load(open(os.path.join(init_ckpt, "model.safetensors.index.json")))
    shard = idx["weight_map"]["mtp.fc.weight"]
    tensors = load_file(os.path.join(init_ckpt, shard))
    for k, v in tensors.items():
        if k.startswith("mtp."):
            head_sd[k[len("mtp."):]] = v
    missing, unexpected = head.load_state_dict(head_sd, strict=False)
    assert not unexpected, f"unexpected keys loading head init: {unexpected}"
    if missing:
        print(f"  (head init missing {len(missing)} keys, will train from scratch: {missing[:3]}...)")


# --------------------------------------------------------------------------- data
def build_examples(corpus_path, tokenizer, max_seq_len, limit=None):
    """Yield (input_ids, completion_start) for each corpus sample.

    completion_start marks where the model's OWN tokens begin; we only score the head on
    the completion region (the distribution it must draft at serve time).
    """
    exs = []
    with open(corpus_path) as fh:
        for line in fh:
            obj = json.loads(line)
            msgs, text = obj["messages"], obj["text"]
            prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            if not isinstance(prompt_ids, list):  # tf 5.x may return a BatchEncoding/dict
                prompt_ids = prompt_ids["input_ids"]
            comp_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + comp_ids)[:max_seq_len]
            if len(ids) - len(prompt_ids) < 4:  # need >=4 completion tokens for a t+2 label
                continue
            exs.append((ids, len(prompt_ids)))
            if limit and len(exs) >= limit:
                break
    return exs


def collate(batch, pad_id, device):
    maxlen = max(len(ids) for ids, _ in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    comp_start = torch.zeros(B, dtype=torch.long)
    for i, (ids, cs) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
        comp_start[i] = cs
    return input_ids.to(device), attn.to(device), comp_start.to(device)


# --------------------------------------------------------------------------- train
def find_base_parts(model):
    """Locate (text_model, lm_head) across the multimodal wrapper."""
    lm_head = model.get_output_embeddings()
    m = model
    for attr in ("model", "language_model", "model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "embed_tokens"):
            return getattr(m, attr), lm_head
        if hasattr(m, attr):
            m = getattr(m, attr)
    # fall back: walk to the thing with embed_tokens + rotary_emb + norm
    cand = model
    while not (hasattr(cand, "embed_tokens") and hasattr(cand, "rotary_emb")):
        subs = [v for v in vars(cand).get("_modules", {}).values()]
        cand = next(s for s in subs if hasattr(s, "embed_tokens") or hasattr(s, "language_model"))
        if hasattr(cand, "language_model"):
            cand = cand.language_model
    return cand, lm_head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="1-step fwd/bwd on a tiny slice, then exit")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["distill"]
    device = "cuda"
    head_dtype = getattr(torch, d.get("head_dtype", "bfloat16"))

    print(f"[distill] base={d['base_ckpt']}  out={d['out']}")
    config = AutoConfig.from_pretrained(d["base_ckpt"])
    tokenizer = AutoTokenizer.from_pretrained(d["base_ckpt"])
    pad_id = tokenizer.pad_token_id or config.get_text_config().pad_token_id

    # frozen base (HF drops mtp.* on load automatically)
    base_model = AutoModelForCausalLM.from_pretrained(
        d["base_ckpt"], dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    text_model, lm_head = find_base_parts(base_model)
    print(f"[distill] base text model: {type(text_model).__name__}; lm_head {tuple(lm_head.weight.shape)}")

    # head: fp32 master params for stable optimization; reuses base modules at forward
    head = MTPHead(config, d["full_attention_layer_idx"]).to(device=device, dtype=torch.float32)
    load_head_init(head, d["init_from"])
    n_train = sum(p.numel() for p in head.parameters())
    print(f"[distill] training {n_train/1e6:.1f}M head params (base frozen)")

    exs = build_examples(cfg["corpus"]["out"], tokenizer,
                         d["max_seq_len"], limit=8 if args.smoke else None)
    print(f"[distill] {len(exs)} training examples")
    if not exs:
        print("no examples -- generate the corpus first (gen_corpus.py)")
        return 1

    opt = torch.optim.AdamW(head.parameters(), lr=d["lr"], weight_decay=d.get("weight_decay", 0.0))
    micro_bsz = 1
    accum = max(1, d["batch_tokens"] // d["max_seq_len"])
    total_steps = max(1, (len(exs) // micro_bsz // accum) * d["epochs"])
    warmup = int(total_steps * d.get("warmup_ratio", 0.03))

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    os.makedirs(d["out"], exist_ok=True)
    shift = d.get("position_shift", 1)
    step, micro, running = 0, 0, 0.0
    opt.zero_grad()
    for ep in range(d["epochs"]):
        for i in range(0, len(exs), micro_bsz):
            batch = exs[i : i + micro_bsz]
            input_ids, attn, comp_start = collate(batch, pad_id, device)
            with torch.no_grad():
                base_hidden = text_model(input_ids=input_ids, attention_mask=attn,
                                         use_cache=False).last_hidden_state  # (B,T,H) post-norm

            # align: hidden_t + embed(tok_{t+shift}) -> predict tok_{t+shift+1}
            B, T, H = base_hidden.shape
            hid = base_hidden[:, : T - (shift + 1), :]
            next_ids = input_ids[:, shift : T - 1]
            labels = input_ids[:, shift + 1 :].clone()
            mask = attn[:, shift : T - 1]
            # only score the completion region + valid (non-pad) positions
            col = torch.arange(T - (shift + 1), device=device).unsqueeze(0)
            comp_mask = (col >= (comp_start.unsqueeze(1) - shift)) & (mask.bool())
            labels[~comp_mask] = -100

            out = head(text_model, next_ids, hid.to(torch.float32), mask)
            logits = F.linear(out, lm_head.weight.to(out.dtype))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            )
            (loss / accum).backward()
            running += loss.item()
            micro += 1

            if args.smoke:
                print(f"[smoke] loss={loss.item():.4f}  logits={tuple(logits.shape)}  "
                      f"scored={int((labels!=-100).sum())} tokens  OK")
                return 0

            if micro % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = d["lr"] * lr_at(step)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"  step {step}/{total_steps}  loss {running/accum/10:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
                    running = 0.0

    # write final checkpoint: copy base shards + retuned head shard, patch index
    print(f"[distill] writing {d['out']}")
    _write_final(d["base_ckpt"], d["out"], head, head_dtype)
    print("[distill] done. validate with experiments/mtp/validate.sh")
    return 0


def _write_final(base_ckpt, out, head, head_dtype):
    import shutil
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(base_ckpt, "model.safetensors.index.json")))
    for f in os.listdir(base_ckpt):
        src = os.path.join(base_ckpt, f)
        if not os.path.isfile(os.path.realpath(src)):
            continue
        if f == MTP_SHARD:  # replaced below
            continue
        dst = os.path.join(out, f)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(os.path.realpath(src), dst)
        except OSError:
            shutil.copy2(os.path.realpath(src), dst)
    head_sd = head.export_state_dict(head_dtype)
    save_file(head_sd, os.path.join(out, MTP_SHARD), metadata={"format": "pt"})
    for k in head_sd:
        idx["weight_map"][k] = MTP_SHARD
    json.dump(idx, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
