---
license: mit
base_model: ornith-ai/Ornith-1.5-9B
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - speculative-decoding
  - mtp
  - multi-token-prediction
  - qwen3.5
  - vision
pipeline_tag: image-text-to-text
---

# Ornith-1.5-9B MTP — GGUF (llama.cpp speculative decoding)

GGUF builds of [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B) with a
**distilled MTP draft head baked into the trunk** — llama.cpp does lossless multi-token
self-speculative decoding out of the box, no separate draft model to wire up. Every file here
carries the `nextn` head, so `--spec-type draft-mtp` just works.

Ornith ships 1.5-9B with `mtp_num_hidden_layers: 1` in `config.json` but **none of the `mtp.*`
weights** — 0 of 760 tensors. So the stock GGUFs (official and third-party) have no MTP head and
can't speculate. These do.

- **Blackwell (RTX 50xx / PRO 6000): use `NVFP4`.** 6.5 GB, and the fastest rung here —
  **299 tok/s with MTP (1.38×)**. MTP's verify step is nearly free on FP4 tensor cores and
  costs real time on the K-quant dequant path, so the two compound.
- **6 GB card? Use `IQ4_XS`** (5.45 GB). It is smaller *than* `Q4_K_M`, faster, and takes a
  bigger MTP gain (1.21× vs 1.06×). `IQ3_M` (4.67 GB) and `IQ2_M` (3.87 GB) go lower and stay
  coherent — both still recall a needle exactly at 32K.
- **Use `Q8_0` for reference quality.** It takes the largest *relative* MTP gain, because
  its baseline is the most bandwidth-bound (1.57×–1.77× depending on prompt mix) — but it is
  still slower in absolute terms than `NVFP4`.

> **`Q4_K_M` is no longer the low-VRAM recommendation.** An earlier version of this card said it
> was. `IQ4_XS` beats it on size, speed and MTP gain — measured, table below.

> Want the base with no MTP head? `ornith-ai/Ornith-1.5-9B-GGUF`.

## Files

| File | Size | Form | Use |
|---|---:|---|---|
| `Ornith-1.5-9B-MTP-NVFP4.gguf` | 6.5 GB | bundled | **Blackwell: fastest rung (299 tok/s, 1.38×)** |
| `Ornith-1.5-9B-MTP-Q8_0.gguf` | 9.8 GB | bundled | largest *relative* MTP gain, reference quality |
| `Ornith-1.5-9B-MTP-Q6_K.gguf` | 7.6 GB | bundled | near-lossless quant |
| `Ornith-1.5-9B-MTP-Q5_K_M.gguf` | 6.6 GB | bundled | balanced quality |
| `Ornith-1.5-9B-MTP-Q4_K_M.gguf` | 5.8 GB | bundled | superseded by `IQ4_XS` — see above |
| `Ornith-1.5-9B-MTP-IQ4_XS.gguf` | 5.45 GB | bundled (imatrix) | **best low-VRAM rung**, fits 6 GB |
| `Ornith-1.5-9B-MTP-IQ3_M.gguf` | 4.67 GB | bundled (imatrix) | 6 GB with room for context |
| `Ornith-1.5-9B-MTP-IQ2_M.gguf` | 3.87 GB | bundled (imatrix) | smallest; still coherent |
| `Ornith-1.5-9B-MTP-BF16.gguf` | 18.4 GB | bundled (master) | re-quantize from this |
| `mmproj-Ornith-1.5-9B-BF16.gguf` | 922 MB | vision projector | required for image input |
| `mtp-head/mtp-Ornith-1.5-9B-head-Q8_0.gguf` | 2.4 GB | standalone head | attach to a base GGUF via `--model-draft` |

"Bundled" = trunk + `nextn` head in one file. The standalone head is **not a model** — loading
`mtp-head/…` directly will crash. It exists only to pair with a base Ornith-1.5-9B GGUF.

Ornith-1.5-9B is a **vision** model; pair any rung with `mmproj-…` for image input. Verified
working with MTP enabled.

## Run

```bash
llama-server --model Ornith-1.5-9B-MTP-Q8_0.gguf \
  --mmproj mmproj-Ornith-1.5-9B-BF16.gguf \
  --n-gpu-layers 99 --ctx-size 8192 --flash-attn on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3
```

`--spec-draft-n-max` is the draft depth: **2** maximizes acceptance, **3** maximizes throughput,
**4 regresses**. Same shape our 1.0 head showed, reproduced independently here.

**Standalone draft** — pair the small head with any base Ornith-1.5-9B GGUF:

```bash
llama-server --model ornith-1.5-9b-Q4_K_M.gguf \
  --model-draft mtp-head/mtp-Ornith-1.5-9B-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 --n-gpu-layers 99 --flash-attn on --jinja
```

## The finding: a grafted head was NOT good enough this time

Our [Ornith-1.0-9B head](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF) transferred
from base Qwen3.5-9B almost intact — Ornith-1.0 was a light enough fine-tune that base-Qwen's
residual stream survived, and the raw graft hit 0.74–0.76 acceptance with zero training.

**Ornith-1.5 breaks that.** Its end-to-end RL self-improvement loop moved the hidden states much
further, and the same graft lands 0.09–0.13 lower at *every* draft depth. Re-distilling the head
against Ornith-1.5's own generations (KL distribution-match, 492 steps) recovers all of it:

    n-max   graft   distilled   Δ        1.0's shipped head
    -----   -----   ---------   ------   ------------------
      2     0.636     0.767     +0.131         0.766
      3     0.528     0.663     +0.135         0.651
      4     0.473     0.583     +0.110         0.565

The lesson generalizes: **how well an MTP head transplants is a function of how far the fine-tune
moved the residual stream.** A light SFT keeps the donor head usable; a heavy RL loop does not.
Measure acceptance before assuming a graft is enough — the head loads and generates correctly
either way, so nothing but the acceptance rate tells you.

Objective matters too: on 1.0, hard-CE distillation *regressed* the graft (0.763 → 0.721) by
over-sharpening the argmax. MTP acceptance is rejection sampling against the target, so it rewards
distribution match, not token fit. KL is the correct objective; this build uses it.

## Benchmarks

**RTX PRO 6000 Blackwell (sm120), ctx 8192, flash-attn, greedy, 6-prompt code+general mix,
`-n 200`, quiet GPU. Single-stream (C=1) — see the caveat below.**

### Q8_0, n-max sweep

| config | decode tok/s | acceptance | speedup |
|---|---:|---:|---:|
| base (no MTP) | 149.6 | — | 1.00× |
| MTP n-max 2 | 252.6 | **0.767** | 1.69× |
| **MTP n-max 3** | **264.5** | 0.663 | **1.77×** |
| MTP n-max 4 | 256.2 | 0.583 | 1.71× |

### Across the full ladder, n-max 3

Every row below was measured in **one session on one box with one prompt mix**, so the rows are
comparable to each other:

| rung | size | base tok/s | +MTP tok/s | speedup | acceptance |
|---|---:|---:|---:|---:|---:|
| **NVFP4** | 6.53 GB | 216.1 | **299.1** | 1.38× | 0.599 |
| IQ2_M | 3.87 GB | 260.7 | 276.8 | 1.06× | 0.558 |
| IQ4_XS | 5.45 GB | 228.4 | 276.9 | 1.21× | 0.525 |
| IQ3_M | 4.67 GB | 233.8 | 257.0 | 1.10× | 0.507 |
| Q6_K | 7.56 GB | 171.6 | 249.7 | 1.46× | 0.541 |
| Q8_0 | 9.79 GB | 150.5 | 236.4 | 1.57× | 0.543 |
| Q4_K_M | 5.78 GB | 203.4 | 215.7 | 1.06× | 0.544 |

**NVFP4 is the fastest rung outright**, and it is not simply "4-bit is small": IQ4_XS and IQ2_M
are *smaller* and still slower with MTP on. FP4 sits on Blackwell's tensor-core GEMM path, where
MTP's parallel verify is nearly free, while K-quants and i-quants pay that verify on the dequant
path. The speedup *ratio* still grows with precision (Q8_0 1.57×) because Q8_0's baseline is the
most bandwidth-bound — but ratio and absolute speed point at different files, and what you want
to run is the fast one.

> **Why Q8_0 reads 1.57× here and 1.77× in the sweep above:** different prompt mix, and
> acceptance moved with it (0.543 vs 0.663). MTP speedup is a function of how predictable your
> text is, so compare rows *within* a table, never across the two.

### ⚠️ Q4_K_M + MTP regresses on creative prose

Acceptance tracks predictability, and it collapses on open-ended prose (0.310 at n-max 3). On
Q4_K_M that is below the break-even point — the verify costs more than speculation saves:

    Q4_K_M, n-max 3     no-MTP    +MTP     acceptance
    code                 206.6    256.4      0.702
    math                 206.8    286.3      0.820
    structured           206.6    239.1      0.631
    creative prose       206.6    159.4      0.310   <- 23% SLOWER

On Q8_0 prose still nets positive (195.8 vs 149.6) because the baseline is slower to begin with.
**If your workload is mostly long-form creative writing on Q4_K_M, run without `--spec-type`.**

### Methodology caveat — read before quoting these

These are **single-stream (C=1)** numbers, which our house rule normally bars from a model card,
because speculative-decoding wins are known to compress or invert under concurrent load (our
[dFlash finding](https://protolabs.studio/lab): a +43% single-stream win became 3× *slower* than
MTP at C=32). We publish C=1 here because llama.cpp/GGUF deployment is overwhelmingly single-user
and local, which makes C=1 the honest representative regime for this artifact — **but do not carry
these numbers over to a batched server.** No concurrency sweep was run for this release.

## "Lossless" — read this

MTP speculative decoding is **distribution-lossless**: every drafted token is verified against the
target, so the output distribution is unchanged. It is **not bitwise-identical** to plain decode at
greedy/temp 0 — the batched verify computes target logits in a different floating-point reduction
order than sequential decode, which can flip a greedy argmax and fork the text. Both outputs are
equally valid; this is expected llama.cpp behavior, not a defect of these weights.

## Troubleshooting: `wrong number of tensors expected 442 got 427`

The gap is the 15 `mtp.*` head tensors. This happens if you convert the **base**
`ornith-ai/Ornith-1.5-9B` directly without grafting a head first: the base keeps
`mtp_num_hidden_layers: 1` in `config.json` but ships none of the `mtp.*` weights, so the converter
declares a `blk.32` MTP layer and leaves those 15 tensors empty.

**Fix:** graft the head into the trunk before converting, then convert with no `--mtp` flag. (Only
4 of the 15 land as `blk.32.nextn.*`; the other 11 become ordinary `blk.32.*`, so `grep nextn`
shows 4 but the head is complete.) Or run the stock base GGUF with
`--model-draft mtp-head/mtp-Ornith-1.5-9B-head-Q8_0.gguf`.

**MTP-baked GGUFs need recent runtimes.** Old Ollama (≤~0.30) fails with `layer 32 missing
attn_qkv`; update and re-pull.

## How these were built

```bash
# 1. graft Qwen3.5-9B's 15 mtp.* tensors into the Ornith-1.5 trunk
python graft.py --donor Qwen/Qwen3.5-9B --target ornith-ai/Ornith-1.5-9B \
                --out ornith-1.5-9b-mtp-graft --dtype bfloat16
# 2. corpus = Ornith-1.5's OWN generations (3942 samples, no-think, T=0.7)
python gen_corpus.py --url <served-1.5> --model ornith15 --out corpus.jsonl
# 3. distill: freeze base, train ONLY the 15 mtp.* tensors, KL objective
python distill.py --config configs/ornith-1.5-9b.yaml     # 492 steps, loss 0.889 -> 0.357
# 4. convert (remaps mtp.* -> blk.32.nextn.* automatically) + quantize
python convert_hf_to_gguf.py ornith-1.5-9b-mtp --outfile ...-BF16.gguf --outtype bf16
llama-quantize ...-BF16.gguf ...-Q4_K_M.gguf Q4_K_M
```

Recipe: [`experiments/mtp/`](https://github.com/protoLabsAI/lab) — the scripts retarget to any
Qwen3.5-family fine-tune by swapping a config.

## The i-quant rungs, and what the MTP head does at low bit depth

The `IQ` rungs are i-quants (importance-matrix calibrated) with the **MTP head pinned to Q8_0**.
That pin is load-bearing for a non-obvious reason: a plain forward pass never activates the
`nextn` head, so the importance matrix contains **no data at all** for those 15 tensors. Left
unpinned they would be i-quantized blind — on the one tensor group where that costs the most,
since a degraded draft loses acceptance on every token. (`output.weight` and `token_embd.weight`
are likewise absent from the imatrix and fall back to llama.cpp's defaults.)

Calibration corpus: ~2 MB, 70% Ornith-1.5-9B's own generations (agentic/business, coding,
general chat — the same corpus the MTP head was distilled against) and 30% literary prose. The
prose share is deliberate: instruct output is register-narrow and creative writing is the first
thing to go at 3 bits.

**MTP does not invert at low bit depth.** The Q8_0 → Q4_K_M decay (1.57× → 1.06× on the ladder mix) looks like a
trend heading for a regression at 3 and 2 bits. It is not one — acceptance holds in a 0.51–0.60
band across the entire ladder and every rung is net-positive with the head on.

### Coherence at low bit depth

Needle recall + degeneration detectors at 4K / 16K / 32K context, plus a verifiable word problem:

| rung | 4K | 16K | 32K | word problem (answer 17:05) |
|---|---|---|---|---|
| NVFP4 | clean | clean | clean | correct, with a distance check |
| IQ4_XS | clean | clean | clean | correct, with a distance check |
| IQ3_M | clean | clean | clean | correct, with a distance check |
| IQ2_M | clean | clean | clean | correct, with a distance check |

Needle exact at every depth on every rung — no repetition loops, no mid-word garbage. IQ2_M
solves the two-train problem correctly *and verifies its own answer* at 2.7 bpw.

**Want the vLLM build?** [`protoLabsAI/Ornith-1.5-9B-NVFP4`](https://huggingface.co/protoLabsAI/Ornith-1.5-9B-NVFP4)
— W4A4 NVFP4 for vLLM on Blackwell, same distilled MTP head, vision verified against the bf16
source.

## Provenance & license

- **Base:** `ornith-ai/Ornith-1.5-9B` (MIT) — a dense Qwen3.5-9B-architecture hybrid
  (linear + full attention) VL fine-tune, trained with end-to-end RL self-improvement.
- **MTP head:** grafted from `Qwen/Qwen3.5-9B` (Apache-2.0), then KL-distilled against
  Ornith-1.5-9B's own hidden states.
- These GGUFs derive from both; **MIT**. Built by [protoLabs.studio](https://protolabs.studio).
