# Ornith-1.0-9B MTP — proof run (2026-06-27)

First instance of the `experiments/mtp/` toolkit. Quick proof run (Josh-approved short
window): does grafting + distilling a Qwen3.5 MTP head onto Ornith-1.0-9B recover a usable
speculative-decode acceptance rate?

## Setup

- **Base**: `deepreinforce-ai/Ornith-1.0-9B` (dense Qwen3.5-9B fine-tune, ships 0 MTP tensors).
- **Donor head**: `Qwen/Qwen3.5-9B` (15 `mtp.*` tensors).
- **Corpus**: 3942 of Ornith-9B's **own** no-think generations (~1.8M tokens) on
  WildBench + ToolACE seed prompts. No external/proprietary data (self-distillation).
- **Distill**: froze base, trained only the 487 MB head, 1 epoch, lr 2e-4, 492 steps.
  Train loss 2.0 → ~1.0–1.1.
- **Measurement**: served off-gateway on :8005 with
  `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, probed acceptance
  over ~9k draft tokens, no-think (matched to training). Plain-9B baseline ~75 tok/s
  single-stream (challenger bench).

## Results

| Variant | Acceptance | Single-stream tok/s | Notes |
|---|:---:|:---:|---|
| Plain Ornith-9B (no MTP) | — | ~75 | challenger baseline |
| **Graft (Qwen head, untrained)** | **0.763** | **111.6** | **+49% — lossless, zero training** |
| Distilled head (v1) | 0.721 | 97.5 | regressed vs graft |

## Findings (honest)

1. **The graft alone is the win.** Qwen3.5-9B's MTP head transfers to Ornith-1.0-9B nearly
   intact — **0.763 acceptance**, ~**111 tok/s (+49%)**, **lossless by construction** (the
   base verifies every drafted token; output distribution is provably unchanged, so quality
   == plain Ornith regardless of head quality). This **overturns the prior assumption** that
   a fine-tune's moved hidden states would collapse a donor head's acceptance — Ornith-1.0
   is a light enough fine-tune that base-Qwen's residual-stream distribution is preserved.
   The shippable artifact is the graft: no training, no data.

2. **Distillation v1 did NOT beat the graft — it regressed it** (0.763 → 0.721). Training on
   Ornith's own outputs moved the head *away* from a good initialization. Two real suspects
   (the earlier "rope position offset" idea is ruled out — a *uniform* position shift cancels
   in relative rope, so it can't matter):
   - **Train/serve forward op-mismatch.** If `distill.py`'s forward isn't bit-aligned with
     vLLM's MTP serving forward, lower training loss (measured under our forward) need not
     mean higher acceptance (measured under vLLM's). Optimizing the wrong forward drifts a
     good init downward. Candidates: SDPA-vs-vLLM-attention numerics, the `norm`/residual
     fusion ordering, fp32-train/bf16-serve.
   - **Over-training from a strong init.** lr 2e-4 × 492 steps on a 243M head, no early stop,
     narrow no-think single-epoch corpus — easy to overshoot when the init is already 0.763.

   **Diagnostic (run next window):** `eval_head.py` computes an offline acceptance *proxy*
   (next-next-token argmax-match rate under our forward). If the **graft's** proxy ≈ 0.76
   (matching its vLLM acceptance), our forward is parity-correct and the regression is pure
   optimization → fix with low-lr + early-stop. If the graft's proxy ≠ 0.76, we have a
   forward op-bug to fix before any training is meaningful.

## Next iteration (distillation, to beat the graft)

- **Fix forward parity first** — make `distill.py`'s position_ids / rope match vLLM's MTP
  serving forward exactly; re-probe. This is the highest-leverage fix.
- **Init-and-gently-tune** — start from the graft (already 0.763), low lr, early-stop on a
  held-out acceptance proxy, so we can only improve on the init.
- **Broader corpus** — include thinking-on generations (daily driver serves thinking-on);
  more tokens; multiple epochs only with early stop.
- **Target**: beat 0.763 (native Qwen3.5-9B+MTP is ~0.79). If distillation can't clear the
  graft after the parity fix, the honest conclusion is "graft suffices for Ornith" and the
  publishable artifact is the graft + this transfer finding.

## Artifacts

- Graft checkpoint: `/mnt/data/checkpoints/ornith-9b-mtp-graft` (serveable, 0.763).
- Distilled v1: `/mnt/data/checkpoints/ornith-9b-mtp` (0.721 — superseded by graft for now).
- Corpus: `/mnt/data/datasets/ornith-9b-mtp/corpus.jsonl`.
- Toolkit: `experiments/mtp/` (graft.py, gen_corpus.py, distill.py, validate.sh).
