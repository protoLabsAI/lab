"""White-box KD LoRA trainer: masked-CE + skew-forward-KL over cached teacher top-k.

Extends train_lora.py. The teacher (Ornith-35B) top-k logprobs were precomputed once
(`precompute_teacher_logits.py`) and cached per-trajectory as .npz {pos, topk_ids, topk_lp}.
Loss = (1-β)·CE_masked + β·SkewForwardKL_topk, both assistant-masked.

Off-policy (fixed teacher-generated corpus) → forward/skew-forward KL is correct (reverse-KL's
edge is an on-policy phenomenon; see DistiLLM-2). Skew: KL(t ‖ α·t+(1-α)·s), α small, kills the
zero-prob blow-ups of raw forward-KL over a 248K vocab. Top-k renormalized on both sides.

  ~/dev/quant-env/bin/python train_lora_kd.py --data <jsonl> --cache <npz_dir> --out <dir> \
      --beta 0.5 --alpha 0.1 --lora-r 64 --epochs 2
"""
from __future__ import annotations
import argparse, json, os
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")
import numpy as np, torch, torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, Qwen3_5ForConditionalGeneration, Trainer, TrainingArguments)
from train_lora import build_example, _flatten, TARGETS


def load_dataset_kd(path, cache_dir, tok, max_len, subset):
    import random
    rows = [json.loads(l) for l in open(path)]
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)          # SAME seed as train_lora → cache idx alignment
    if subset:
        order = order[:subset]
    out = []
    for idx in order:
        ex = build_example(_flatten(rows[idx]["messages"]), tok, max_len)
        if not ex or not any(l != -100 for l in ex["labels"]):
            continue
        npz = os.path.join(cache_dir, f"{idx:06d}.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        T, k = len(ex["input_ids"]), z["topk_ids"].shape[1]
        tids = np.full((T, k), -1, np.int64)
        tlp = np.full((T, k), -1e4, np.float32)
        for j, p in enumerate(z["pos"]):
            if p < T:
                tids[p] = z["topk_ids"][j]
                tlp[p] = z["topk_lp"][j].astype(np.float32)
        out.append({**ex, "t_ids": tids.tolist(), "t_lp": tlp.tolist()})
    print(f"KD dataset: {len(out)} trajectories with teacher cache")
    return Dataset.from_list(out)


def kd_collate(feats, pad_id):
    T = max(len(f["input_ids"]) for f in feats)
    k = len(feats[0]["t_ids"][0])
    B = len(feats)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    lab = torch.full((B, T), -100, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    tid = torch.full((B, T, k), -1, dtype=torch.long)
    tlp = torch.full((B, T, k), -1e4, dtype=torch.float32)
    for b, f in enumerate(feats):
        n = len(f["input_ids"])
        ids[b, :n] = torch.tensor(f["input_ids"]); lab[b, :n] = torch.tensor(f["labels"])
        am[b, :n] = 1
        tid[b, :n] = torch.tensor(f["t_ids"]); tlp[b, :n] = torch.tensor(f["t_lp"])
    return {"input_ids": ids, "labels": lab, "attention_mask": am,
            "teacher_ids": tid, "teacher_lp": tlp}


class KDTrainer(Trainer):
    def __init__(self, *a, beta=0.5, alpha=0.1, **kw):
        super().__init__(*a, **kw); self.beta = beta; self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        t_ids = inputs.pop("teacher_ids"); t_lp = inputs.pop("teacher_lp")
        labels = inputs["labels"]
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = out.logits                                   # (B,T,V)
        # CE (masked) — standard -1 shift
        sl = logits[:, :-1, :].contiguous(); slab = labels[:, 1:].contiguous()
        ce = F.cross_entropy(sl.view(-1, sl.size(-1)), slab.view(-1), ignore_index=-100)
        # KD: teacher dist stored at token position t aligns to student pred at t-1
        pl = logits[:, :-1, :]                                # predicts token t (t>=1)
        tid = t_ids[:, 1:, :]; tlp = t_lp[:, 1:, :]           # teacher targets for token t
        valid = (tid[..., 0] >= 0) & (slab >= 0)              # assistant positions w/ cache
        if valid.any():
            idx = tid.clamp(min=0)                            # (B,T-1,k)
            s_top = torch.gather(pl, -1, idx)                 # student logits at teacher ids
            s_top = s_top.masked_fill(tid < 0, -1e4)
            s = torch.log_softmax(s_top.float(), dim=-1)      # student logprob over top-k
            t = torch.log_softmax(tlp.masked_fill(tid < 0, -1e4), dim=-1)  # teacher over top-k
            tp = t.exp()
            m = self.alpha * tp + (1 - self.alpha) * s.exp() + 1e-8
            skl = (tp * (t - m.log())).sum(-1)                # skew-forward-KL per position
            kd = skl[valid].mean()
        else:
            kd = torch.zeros((), device=logits.device)
        loss = (1 - self.beta) * ce + self.beta * kd
        self._last = {"ce": ce.item(), "kd": float(kd)}
        return (loss, out) if return_outputs else loss

    def log(self, logs, *a, **k):
        if hasattr(self, "_last"): logs.update(self._last)
        super().log(logs, *a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--base", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--subset", type=int, default=None); ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lora-r", type=int, default=64); ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.1); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    ds = load_dataset_kd(a.data, a.cache, tok, a.max_seq, a.subset or (50 if a.smoke else None))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM", bias="none"))
    model.print_trainable_parameters()
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=1 if a.smoke else a.bs,
        gradient_accumulation_steps=1 if a.smoke else a.grad_accum, num_train_epochs=a.epochs,
        max_steps=10 if a.smoke else -1, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="no" if a.smoke else "epoch", bf16=True, gradient_checkpointing=True,
        report_to="none", remove_unused_columns=False, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = KDTrainer(model=model, args=args, train_dataset=ds, beta=a.beta, alpha=a.alpha,
                        data_collator=lambda f: kd_collate(f, tok.pad_token_id or 0))
    trainer.train()
    if not a.smoke:
        trainer.save_model(a.out); print(f"saved KD LoRA -> {a.out}")
    else:
        print("KD SMOKE OK")


if __name__ == "__main__":
    main()
