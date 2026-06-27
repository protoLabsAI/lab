# MTP toolkit — graft + distill in-checkpoint MTP heads

Reusable pipeline for giving a Qwen3.5-family fine-tune a native **MTP (Multi-Token
Prediction)** speculative-decode head when the fine-tune shipped without one.

**The product is the recipe**, not just one set of weights. The same scripts retarget to
any Qwen3.5-derived checkpoint (the bf16→FP8 follow-up, other fine-tunes) by swapping a
config.

## Why this exists

A native Qwen3.5 checkpoint ships **15 `mtp.*` tensors** — one `full_attention` decoder
layer + a 2H→H fusion + three RMSNorms — that share the base `embed_tokens`/`lm_head`.
vLLM's `mtp` speculative method reads this head straight from the served checkpoint and
gives a **lossless** decode speedup (the base verifies every drafted token, so quality
cannot change; only tok/s moves). MTP pays off on **dense** models; it *hurts* MoE
(routing overhead > speculation savings), so this is specifically a dense-model play.

Many fine-tunes drop the head. First target: **DeepReinforce's Ornith-1.0-9B** (dense
Qwen3.5-9B fine-tune) — verified **0 of 760** tensors are `mtp.*`, vs **15** in base
`Qwen/Qwen3.5-9B`. So it serves plain (~75 tok/s single-stream); native Qwen3.5-9B+MTP
was +22% at ~79% acceptance.

**You can't just slot Qwen's head on.** Shapes match (it loads), but the head was
co-trained against base-Qwen's residual stream; the fine-tune moved those hidden states,
so acceptance collapses → little/no speedup. The head is coupled to the weights, not a
portable accessory. So: **graft, then re-distill the head against the fine-tune's own
hidden states.**

## Pipeline

```
graft.py        donor mtp.* ──▶ target checkpoint   (verbatim copy, +1 shard, offline/CPU)
gen_corpus.py   served target ──▶ corpus.jsonl       (the model's OWN generations = teacher)
distill.py      freeze base, train ONLY mtp.*        (re-align head to target hidden states; GPU)
eval_head.py    offline acceptance proxy             (localize regressions w/o serving; GPU)
validate.sh     serve + acceptance + tok/s + eval    (off-gateway; lossless check)
```

## Results so far

First run (**Ornith-1.0-9B**, 2026-06-27): the **graft alone wins** — Qwen3.5-9B's head
transfers at **0.763 acceptance / ~111 tok/s (+49%), lossless**, overturning the
"transplant collapses" assumption (Ornith is a light fine-tune). Distillation v1 *regressed*
it (0.721) — next iteration gated on `eval_head.py` forward-parity diagnosis. Full honest
writeup + numbers: [`runs/ornith-9b/RESULTS.md`](runs/ornith-9b/RESULTS.md),
draft post: [`runs/ornith-9b/BLOG.md`](runs/ornith-9b/BLOG.md).

Per-target settings live in `configs/<name>.yaml`; outputs/results in `runs/<name>/`.

### 1. Graft (done for ornith-9b)

```bash
HF_HOME=/mnt/models/huggingface ~/dev/vllm-env/bin/python graft.py \
  --donor Qwen/Qwen3.5-9B --target deepreinforce-ai/Ornith-1.0-9B \
  --out /mnt/data/checkpoints/ornith-9b-mtp-graft --dtype bfloat16
```
Appends `model-mtp.safetensors` (~487 MB bf16) + patches the index; base shards are
hardlinked unchanged. Result is immediately serveable (naive Qwen head — low accept).

### 2. Corpus

Serve the target **off-gateway on :8005** (see GPU note), assemble a prompt JSONL
(agentic + chat + code, to match the serving distribution — e.g. pull claw task prompts
+ custom-coding prompts), then:
```bash
~/dev/vllm-env/bin/python gen_corpus.py --url http://localhost:8005/v1 --model ornith-9b \
  --prompts prompts.jsonl --out /mnt/data/datasets/ornith-9b-mtp/corpus.jsonl --n 20000
```

### 3. Distill (GPU; quant-env per the venv rule)

```bash
~/dev/quant-env/bin/python distill.py --config configs/ornith-9b.yaml --smoke   # 1-step sanity
~/dev/quant-env/bin/python distill.py --config configs/ornith-9b.yaml           # full run
```
Freezes the base (HF drops `mtp.*` on load automatically), trains only the 487 MB head.
Objective matches vLLM's serving forward (see `distill.py` docstring): predict tok_{t+2}
from base post-norm hidden_t + embed(tok_{t+1}). Head trained in fp32, **saved bf16**
(keep the head bf16 even on an FP8 base — quantizing it risks acceptance, same logic as
keeping SSM/router/embed bf16 in Ornith-35B-FP8).

### 4. Validate (GPU; off-gateway)

```bash
bash validate.sh /mnt/data/checkpoints/ornith-9b-mtp 8005 1
```
Reports acceptance rate + decode tok/s, and the lossless-eval commands (re-run the 9B
challenger row — expect ~identical scores; only speed changes).

## GPU / production note ⚠️

Daily driver = 2× Ornith-35B-FP8 replicas (`vllm` :8000 GPU0, `vllm-replica-b` :8003
GPU1) + Fish(:8092) + embed(:8001/:8004). Both cards are ~full. Bench/train the 9B on
**GPU1 only, after stopping replica-b**, and serve **off-gateway on :8005** (NOT :8003 —
that's gateway-routed and would contaminate production):
```bash
sudo systemctl stop vllm-replica-b
# ... do GPU work on CUDA_VISIBLE_DEVICES=1 / :8005 ...
sudo systemctl reset-failed vllm-replica-b && sudo systemctl start vllm-replica-b
```
Stopping replica-b halves smart-lane throughput while down — keep the window tight.
Kill stuck vLLM by >40 GB GPU-mem (not cmdline grep). See
[[feedback_orphan_kill_by_gpu_mem]].

## Recon findings (so the next session doesn't re-derive them)

- **HF transformers ignores `mtp.*` on load** (`_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]`
  in `modeling_qwen3_5.py`) — there's no built-in MTP training forward, so distill.py
  implements the head, reusing the base's HF modules (`Qwen3_5DecoderLayer`,
  `Qwen3_5RMSNorm`, `rotary_emb`, `create_causal_mask`) for serving parity.
- **The MTP consumes post-final-norm hidden** — vLLM's main model returns
  `self.norm(hidden_states)`, and the MTP re-normalizes via `pre_fc_norm_hidden`. Confirmed
  in `qwen3_5.py` / `qwen3_next.py` and `qwen3_5_mtp.py`.
- **Ornith-9B is Qwen3.5 hybrid** (linear_attention Mamba layers + full_attention every
  4th); the MTP layer itself is `full_attention` (gated attn, `attn_output_gate=true` →
  q_proj 8192=2×4096; head_dim 256; GQA 32/4). `mtp_num_hidden_layers=1`,
  `mtp_use_dedicated_embeddings=false` (shares base embed).
- Serve with `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`. vLLM
  0.22.1 supports mtp/eagle/eagle3/dflash/medusa/ngram; `speculators` NOT installed (only
  needed for the EAGLE-3 follow-up).

## Spec-decode ladder (this toolkit covers MTP; siblings noted)

1. **MTP** (here) — cheapest, in-checkpoint, scales under concurrency. Fills the gap
   DeepReinforce left. Do bf16 first, then FP8+MTP as the edge follow-up.
2. **EAGLE-3** — separate feature-conditioned draft + tree, highest ceiling; needs
   `speculators` + a trained draft. Different mechanism → its own experiment, not this dir.
3. **dFlash** — block-diffusion draft, single-stream only (doesn't scale). See
   `experiments/dflash/`.

Related memory: `project_ornith_9b_mtp`, `project_ornith_daily_driver`, `project_lab_focus`.
