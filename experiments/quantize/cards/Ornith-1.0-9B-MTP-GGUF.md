GGUF builds of [`deepreinforce-ai/Ornith-1.0-9B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B)
with the KL-distilled MTP draft head from
[`protoLabsAI/Ornith-1.0-9B-MTP`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP)
**baked into the trunk** — llama.cpp does lossless multi-token self-speculative decoding out of
the box, no separate draft model to wire up. Every file here carries the `nextn` head, so
`--spec-type draft-mtp` just works.

Two things to know before you pick a file:

- **Ampere and older:** use `Q4_K_M`. It is smaller *and* faster than everything else here.
- **Blackwell (RTX 50xx / PRO 6000):** use `NVFP4`. MTP compounds with NVFP4's tensor-core GEMMs
  where it only partially helps K-quants — so `NVFP4` is the fastest rung on this hardware by
  ~28%, despite being a hair larger than `Q4_K_M`. The measured mechanism is below.

> Want the base with no MTP head? `deepreinforce-ai/Ornith-1.0-9B-GGUF`.

## Files

| File | Size | Form | Use |
|---|---:|---|---|
| `Ornith-1.0-9B-MTP-NVFP4.gguf` | 6.6 GB | bundled | **Blackwell: fastest rung** (306 tok/s +MTP) |
| `Ornith-1.0-9B-MTP-Q8_0.gguf` | 9.8 GB | bundled | reference quality / largest relative MTP gain |
| `Ornith-1.0-9B-MTP-Q6_K.gguf` | 7.6 GB | bundled | near-lossless quant |
| `Ornith-1.0-9B-MTP-Q5_K_M.gguf` | 6.6 GB | bundled | balanced quality |
| `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | 5.8 GB | bundled | **Ampere: fastest rung** |
| `Ornith-1.0-9B-MTP-IQ4_XS.gguf` | 5.5 GB | bundled (imatrix) | low VRAM, near-Q4 quality |
| `Ornith-1.0-9B-MTP-IQ3_M.gguf` | 4.7 GB | bundled (imatrix) | lower VRAM |
| `Ornith-1.0-9B-MTP-IQ2_M.gguf` | 3.9 GB | bundled (imatrix) | very low VRAM (~5 GB to serve) |
| `Ornith-1.0-9B-MTP-BF16.gguf` | 18.4 GB | bundled (master) | re-quantize from this |
| `mtp-head/mtp-Ornith-1.0-9B-head-Q8_0.gguf` | 2.4 GB | standalone head | attach to a base GGUF via `--model-draft` |

"Bundled" = trunk + `nextn` head in one file. The **IQ** rungs are i-quants (importance-matrix
calibrated) with the **MTP head pinned to Q8_0** so acceptance holds on the low-bit trunk
(measured ~0.81–0.84 on IQ2_M–IQ4_XS, on par with the k-quants). Serve them exactly like the
k-quants.

The standalone head is **not a model** — loading `mtp-head/…` directly will crash. It exists only
to pair with a base Ornith-9B GGUF via `--model-draft`.

Requires **llama.cpp ≥ b9616** (Qwen3.5 `qwen35` arch + `--spec-type draft-mtp`). The `NVFP4`
rung additionally needs a build with NVFP4 support (`GGML_TYPE_NVFP4`, type 40) — both landed
in llama.cpp spring 2026. LM Studio (recent) and Ollama (≥ ~0.31) inherit MTP through llama.cpp;
older Ollama fails with `layer 32 missing attn_qkv` → update and re-pull.

## Run

**Bundled (recommended)** — the head travels in the file:

```bash
llama-server --model Ornith-1.0-9B-MTP-Q4_K_M.gguf \
  --n-gpu-layers 99 --ctx-size 8192 --flash-attn on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 3
```

**One-command pull:**

```bash
llama-server -hf protoLabsAI/Ornith-1.0-9B-MTP-GGUF:NVFP4  --spec-type draft-mtp -ngl 99  # Blackwell
llama-server -hf protoLabsAI/Ornith-1.0-9B-MTP-GGUF:Q4_K_M --spec-type draft-mtp -ngl 99  # everything else
```

**Standalone draft** — pair the small head with any base Ornith-9B GGUF:

```bash
llama-server --model ornith-1.0-9b-Q4_K_M.gguf \
  --model-draft mtp-head/mtp-Ornith-1.0-9B-head-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --n-gpu-layers 99 --ctx-size 8192 --flash-attn on --jinja
```

`--spec-draft-n-max` is the draft depth: **2** maximizes acceptance, **3** maximizes throughput,
4 starts to regress. Tune per workload.

## The NVFP4 × MTP finding (why Blackwell is different)

`NVFP4` weights sit on the tensor-core FP4 GEMM path; K-quants dequantize to a compute path each
step. MTP's cost is a per-step parallel *verify* of the drafted tokens — and that verify is
nearly free on NVFP4's GEMMs but costs ~+28% on the K-quant dequant path. So MTP's speedup is
effectively **multiplicative with NVFP4 and only partial with K-quants**.

Measured, acceptance-controlled (`-n 200` greedy, 6 diverse prompts, ranges shown; Ampere
single-run indicative):

    file        size    Ampere A6000        Blackwell (sm120)
                        no-MTP   +MTP       no-MTP   +MTP
    ---------   ------  ------   ------     ------   ----------------
    Q4_K_M      5.8 GB  104.6    153.4      205.1    239  (216–252)
    NVFP4       6.6 GB   70.7     84.8      201.5    306  (287–330)

On Blackwell the *worst* NVFP4 prompt (287) beats the *best* Q4_K_M prompt (252); draft
acceptance is near-equal on both files at the same prompt/box, so the gap is verify cost, not
acceptance. On Ampere the FP4 path has no tensor-core backing, so Q4_K_M wins outright — use it.
Code prompts run hottest (330), creative prose lowest (287): MTP acceptance tracks predictability.

The FP8 × spec-decode compounding is documented upstream (TensorRT-LLM); the FP4-vs-K-quant
verify-cost split is, as far as we've found, new data.

## Benchmarks (RTX A6000, ctx 8192, flash-attn, greedy; 6-prompt code+general mix)

### n-max sweep, Q8_0

| config | decode tok/s | acceptance | speedup |
|---|---:|---:|---:|
| base (no MTP) | 71.0 | — | 1.00× |
| MTP n-max 2 | 118.3 | 0.766 | 1.67× |
| **MTP n-max 3** | **122.6** | 0.651 | **1.73×** |
| MTP n-max 4 | 120.8 | 0.565 | 1.70× |

Per-token acceptance at n-max 2 (**0.766**) matches the vLLM reference for this head (0.762).

### Across quants, n-max 3

| quant | base tok/s | +MTP tok/s | speedup | acceptance |
|---|---:|---:|---:|---:|
| Q4_K_M | 105.4 | **145.3** | 1.38× | 0.659 |
| Q8_0 | 71.0 | 122.6 | 1.73× | 0.651 |

Acceptance is quant-stable (~0.65 at n-max 3 even with a Q4 trunk). Q4_K_M is fastest in absolute
terms on Ampere; the *relative* MTP gain grows with precision (Q8's bandwidth-bound baseline has
more to gain from the parallel verify).

Full rows behind every number: [`protoLabsAI/lab-benchmarks`](https://huggingface.co/datasets/protoLabsAI/lab-benchmarks)
· charts at [protolabs.studio/lab](https://protolabs.studio/lab).

## "Lossless" — read this

MTP speculative decoding is **distribution-lossless**: every drafted token is verified against the
target, so the output distribution is unchanged. It is **not bitwise-identical** to plain decode at
greedy/temp 0 — the batched verify computes target logits in a different floating-point reduction
order than sequential decode, which can flip a greedy argmax and fork the text. Both outputs are
equally valid; this is expected llama.cpp behavior, not a defect of these weights.

## Troubleshooting: `wrong number of tensors expected 442 got 427`

(or `got 426` on smaller quants — the gap is the 15 `mtp.*` head tensors.)

This happens if you convert the **base** `deepreinforce-ai/Ornith-1.0-9B` directly without grafting
the head first. The base keeps `mtp_num_hidden_layers: 1` in `config.json` but ships **none** of the
`mtp.*` weights, so the converter declares a `blk.32` MTP layer while leaving those 15 tensors empty
→ llama.cpp expects 442 and finds 427.

**Fix:** graft the head into the trunk before converting, then convert with no `--mtp` flag. (Only 4
of the 15 head tensors are named `blk.32.nextn.*`; the other 11 land as ordinary `blk.32.*` tensors,
so `grep nextn` shows 4 but the head is complete.) Or skip grafting entirely and run the base GGUF
with `--model-draft mtp-head/mtp-Ornith-1.0-9B-head-Q8_0.gguf` — functionally identical.

## How these were built

```bash
# 1. graft the mtp.* head into the base trunk (15 tensors, 1 nextn layer)
python graft.py --donor protoLabsAI/Ornith-1.0-9B-MTP \
                --target deepreinforce-ai/Ornith-1.0-9B --out ./ornith-9b-mtp-kl
# 2. convert (remaps mtp.* -> blk.32.nextn.* automatically)
python convert_hf_to_gguf.py ./ornith-9b-mtp-kl --outfile out/...-BF16.gguf --outtype bf16
# 3. quantize (NVFP4 rung converted from the gate-verified vLLM NVFP4 quant, same scales)
llama-quantize out/...-BF16.gguf out/...-Q4_K_M.gguf Q4_K_M
```

The `graft.py` recipe and the KL-distillation details live in the head repo
[`protoLabsAI/Ornith-1.0-9B-MTP`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP). The NVFP4
rung is converted from [`protoLabsAI/Ornith-1.0-9B-NVFP4`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-NVFP4)
— full quality/coherence receipts on that card.

## Want a different size or format?

Open a Community discussion — requests usually ship within 48h. That's how most of the quants here
got made.

## Provenance & license

- **Base:** `deepreinforce-ai/Ornith-1.0-9B` (MIT) — a Qwen3.5-9B hybrid (linear + full attention) fine-tune.
- **MTP head:** `protoLabsAI/Ornith-1.0-9B-MTP` (MIT) — KL-distilled against Ornith's own hidden states.
- These GGUFs derive from both; **MIT**. Built by [protoLabs.studio](https://protolabs.studio).
