# A Qwen3.5 MTP head transfers to a fine-tune for free (+49%, lossless)

DeepReinforce shipped Ornith-1.0-9B — a dense Qwen3.5-9B fine-tune — without MTP
weights. We verified it: **0 of 760 tensors** are `mtp.*`, where base `Qwen/Qwen3.5-9B`
ships **15**. So Ornith-9B serves plain (~75 tok/s single-stream); the native
Multi-Token-Prediction speedup is just absent.

The folk wisdom says you can't fix that by copying the base model's MTP head over. The
head reads the model's residual stream and is co-trained against that exact distribution;
fine-tuning moves the hidden states, so a transplanted head should draft garbage and
acceptance should collapse. We expected to have to re-distill the head against Ornith's
own hidden states to make it useful.

We measured it instead. The folk wisdom is wrong here.

## The graft

The 15 `mtp.*` tensors are self-contained on disk — one `full_attention` decoder layer, a
2H→H fusion (`fc`), three RMSNorms — sharing the base `embed_tokens`/`lm_head`. They don't
reference any base-model tensor names; the coupling is purely at runtime, where `fc` fuses
the base hidden state with the next token's embedding. So grafting is a verbatim copy:

```bash
python graft.py --donor Qwen/Qwen3.5-9B \
                --target deepreinforce-ai/Ornith-1.0-9B \
                --out ornith-9b-mtp-graft --dtype bfloat16
```

Append one 487 MB shard, patch the index, done. Then serve with vLLM's native method:

```bash
vllm serve ornith-9b-mtp-graft \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## The numbers

Measured on 2× RTX PRO 6000 Blackwell, vLLM 0.22.1, single GPU, ~9k drafted tokens:

| Variant | Acceptance | tok/s (single-stream) |
|---|:---:|:---:|
| Plain Ornith-9B (no MTP) | — | ~75 |
| **Graft (Qwen head, zero training)** | **0.763** | **111.6** |

**0.763 acceptance** — within a whisker of native Qwen3.5-9B+MTP (~0.79). A 487 MB head we
copied from a *different model* drafts for the fine-tune almost as well as if it had been
trained for it. That's **+49% decode throughput** for the cost of a file copy.

And it is **lossless** — not approximately, exactly. MTP speculative decoding has the base
model verify every drafted token; rejected drafts fall back to the base's own sampling. The
output distribution is provably identical to plain Ornith-9B. A bad head only costs you
speed (low acceptance), never quality. So a transplanted head is *safe* even before you know
how well it drafts.

## Why it works

Ornith-1.0 is a light fine-tune. Whatever DeepReinforce did to Qwen3.5-9B, it left the
residual-stream distribution close enough to base that a head trained on base internals
still predicts the next-next token well. The "moved hidden states break the head"
intuition is real in principle — it just takes a heavier fine-tune than this to bite.

The practical upshot: **for any light fine-tune of a base that ships an MTP head, try the
graft first.** It's free, it's lossless, and here it captured ~97% of the native
acceptance with no data and no training.

## What didn't work (yet)

We also distilled the grafted head on Ornith's own generations (self-distillation — no
external data) to try to beat the graft. It *regressed* it (0.763 → 0.721), despite training
loss dropping cleanly. Optimizing our training-time forward moved the head away from a
strong initialization — a sign the training objective and vLLM's serving forward aren't
yet bit-for-bit aligned. When your initialization is already this good, "just fine-tune it"
is a way to make it worse. That's the next iteration; the graft ships as-is.

## Reproduce

Toolkit (donor-agnostic, retargets any Qwen3.5 fine-tune via a config):
`experiments/mtp/` — `graft.py`, `gen_corpus.py`, `distill.py`, `validate.sh`.
Honest run log: `experiments/mtp/runs/ornith-9b/RESULTS.md`.
