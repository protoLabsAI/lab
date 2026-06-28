# EAGLE-3 on Ornith-1.0-9B

Sibling to `experiments/mtp/`. Where MTP is a 1-token in-checkpoint head, **EAGLE-3** is a
separate trained draft model (feature-conditioned on the target's hidden states at multiple
layers) that drafts a *tree* the target verifies in one pass → longer accepted runs → higher
ceiling. Both are lossless (the target verifies every token).

## Graft baseline (2026-06-28) — zero training

Same move as the MTP graft: take an existing **Qwen3.5-9B** EAGLE-3 draft and run it against
the Ornith-9B fine-tune. Draft: `BLR2/Eagle3-Qwen3.5-9B` (`LlamaForCausalLMEagle3`, 1 layer,
763 MB, reads target aux hidden states at layers [1,15,28]). Served Ornith-9B + draft via
`--speculative-config '{"method":"eagle3","model":"BLR2/Eagle3-Qwen3.5-9B","num_speculative_tokens":N}'`
on :8005 (Blackwell). Acceptance/tok-s probe (no-think single-stream):

| Variant | accepted/step | single-stream tok/s | vs plain |
|---|:---:|:---:|:---:|
| plain Ornith-9B (no spec) | — | ~75 | — |
| MTP (distilled, in-checkpoint) | 0.76 | ~121 | +60% |
| **EAGLE-3 graft, n=3** | 1.21 | **138.8** | **+85%** |
| EAGLE-3 graft, n=5 | 1.37 | 136.3 | +82% |

**Finding: EAGLE-3 beats MTP on Ornith-9B even *untuned*** (graft 138.8 vs MTP 121). The
Qwen3.5-9B draft transfers to the fine-tune (Ornith is light enough — same as the MTP-graft
result). `n=5` accepts slightly more per step but the extra verify cost cancels it — the
graft's acceptance (~1.2–1.4/step) is too low to exploit longer drafts, so **n=3 ≈ ceiling**.

## Headline: train an Ornith-specific draft

EAGLE-3 hits 2–4 accepted/step when the draft is trained *for the target*. The graft's 1.2 is
the transfer tax (base-Qwen draft on moved Ornith hidden states). Training a draft on Ornith's
own generations should push accepted-length ~1.2 → ~2–3 → **tok/s ~170–220**, and *then* longer
drafts (n=5–8) pay off.

- **Trainer**: SpecForge (SGLang's EAGLE-3 trainer) or `speculators` — **neither installed**;
  setup needed (dedicated env to avoid the vllm-env transformers pin).
- **Data**: Ornith-9B self-generations already on disk — `/mnt/data/datasets/ornith-9b-mtp/corpus.jsonl`
  (3942 samples, the MTP corpus). Train the 1-layer draft to predict from target aux hidden
  states at the chosen layers.
- **Validate**: serve + probe acceptance/tok-s vs the graft (138.8) and MTP (121); confirm
  lossless via the eval suite (re-run the 9B row — expect identical, only speed changes).
- **Publish**: draft weights + recipe → HF `protoLabsAI`, same playbook as the MTP head; the
  GGUF/llama.cpp path is a separate avaLab follow-up (llama.cpp uses `--model-draft`).

## Spec-decode ladder (Ornith-9B, dense)

plain ~75 → MTP (in-checkpoint, shipped) 121 → **EAGLE-3 graft 138.8** → EAGLE-3 trained
(projected ~170–220). All lossless. EAGLE-3 is the higher ceiling; MTP is the cheaper/simpler
in-checkpoint option. (On the MoE 35B daily driver, neither is a win — spec-decode is a
dense-model play; MTP was −11% there.)

Run logs: `results/{local→ornith-9b-eagle3}_…`. Related: `experiments/mtp/`, `project_ornith_9b_mtp`.
