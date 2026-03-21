# Session Handoff — 2026-03-21 (Afternoon)

## What We Did This Session

### 1. Qwen3.5 Small Model Evaluation (0.8B → 9B)
Evaluated the full Qwen3.5 family for fine-tuning base selection:

- **Qwen3.5-9B** selected as fine-tune base over OmniCoder-9B (same arch, better tool use)
- **Qwen3.5-4B** is the breakout — matches 9B on 6/8 eval dimensions
- **4B INT4 (AWQ)** runs at 294 tok/s, 3GB, quality intact
- **Capability cliff at 4B→2B**: sub-4B models can't reliably call tools (0/4 claw pass^3)
- **2B surprised on reasoning**: 5/5 on logic/math, matching 9B — can think but can't act
- OmniCoder-9B nuked (inferior), Qwen3.5-27B bf16 nuked (redundant with INT4)
- Base models (0.8B, 2B, 4B) downloaded for pretraining experiments

### 2. Comprehensive Eval Suite Built
Created 6 new eval suites (30 tests) and a profile runner:

| Suite | Tests | Domain |
|-------|:-----:|--------|
| `instruction_following` | 5 | Constraint adherence, format compliance |
| `reasoning` | 5 | Math, logic puzzles, deduction, pattern recognition |
| `safety` | 5 | Refusal, jailbreak resistance, PII, security review |
| `structured_output` | 5 | JSON, YAML, SQL, markdown tables, log parsing |
| `summarization` | 5 | Compression, action extraction, TL;DR |
| `coding/analysis` | 5 | Complexity, security review, regex, API design |

**Profiles:**
- `quick` — 6 claw + 6 custom suites + FC, 1 trial (~15 min)
- `full` — 20 claw + 10 custom suites + FC, 3 trials pass^3 (~60-90 min)

```bash
cd evals && ./run.sh profile --name quick --model local
cd evals && ./run.sh profile --name full --model local
```

### 3. Grand Model Comparison (17 Models)
Ran quick profile across local and cloud models:

| Model | Claw | Code | IF | Reason | SO | Summ | Safety | FC | **Total** |
|-------|:----:|:----:|:--:|:------:|:--:|:----:|:------:|:--:|:---------:|
| **GPT-5.4** | 1/6 | 10/10 | 5/5 | 5/5 | 5/5 | 3/5 | 5/5 | 8/8 | **42/49** |
| **Gemini 3 Flash** | 2/5 | 9/10 | 5/5 | 4/5 | 5/5 | 5/5 | 4/5 | 8/8 | **42/48** |
| **Qwen 9B** | 4/6 | 7/10 | 3/5 | 5/5 | 5/5 | 5/5 | 5/5 | 8/8 | **42/49** |
| **Sonnet 4.6** | 2/6 | 10/10 | 5/5 | 4/5 | 4/5 | 4/5 | 4/5 | 8/8 | **41/49** |
| **DeepSeek V3.2** | 4/6 | 9/10 | 4/5 | 5/5 | 3/5 | 4/5 | 4/5 | 8/8 | **41/49** |
| **Qwen 35B MoE** | 2/6 | 10/10 | 3/5 | 5/5 | 4/5 | 4/5 | 5/5 | 8/8 | **41/49** |
| **GPT-OSS-120B** | 0/6 | 9/10 | 5/5 | 5/5 | 5/5 | 3/5 | 4/5 | 8/8 | **39/49** |
| **Grok 4.1 Fast** | 1/6 | 8/10 | 5/5 | 5/5 | 4/5 | 3/5 | 5/5 | 8/8 | **39/49** |
| **Opus 4.6** | 2/6 | 9/10 | 3/5 | 4/5 | 4/5 | 4/5 | 4/5 | 8/8 | **38/49** |
| **Qwen 27B INT4** | 1/6 | 8/10 | 4/5 | 4/5 | 5/5 | 3/5 | 5/5 | 8/8 | **38/49** |
| **Qwen 4B INT4** | 4/6 | 7/10 | 3/5 | 5/5 | 5/5 | 2/5 | 4/5 | 8/8 | **38/49** |
| **GPT-OSS-20B** | 2/6 | 7/10 | 3/5 | 5/5 | 5/5 | 4/5 | 4/5 | 8/8 | **38/49** |
| **Haiku 4.5** | 2/6 | 9/10 | 2/5 | 5/5 | 3/5 | 3/5 | 4/5 | 8/8 | **36/49** |
| **Cydonia 24B** | 0/6 | 8/10 | 4/5 | 4/5 | 2/5 | 3/5 | 1/5 | 2/8 | **24/49** |
| **Qwen 2B** | 1/6 | 3/10 | 3/5 | 5/5 | 3/5 | 2/5 | 2/5 | 2/8 | **21/49** |
| **Qwen 0.8B** | 1/6 | 3/10 | 1/5 | 3/5 | 1/5 | 2/5 | — | 2/8 | **13/49** |

### 4. New Models Tested
- **Cydonia 24B v4.3** (Mistral base): 24/49, tool calling broken, safety 1/5. Keeping for creative/roleplay experiments only.
- **Mistral-Small-4-119B AWQ**: **crashed** — MLA attention with head_size=320 not supported on Blackwell SM 12.0 in vLLM 0.17.1. Nuked (53GB freed).
- **GPT-OSS-20B**: 38/49, slow (37 min for quick profile), tool calls output as text
- **GPT-OSS-120B**: 39/49, faster than 20B (17 min), but 0/6 claw (can't use tools through gateway)
- **Grok 4.1 Fast**: 39/49, finally completed (failed last session due to gateway restart)

### 5. Bug Fixes
- **Claw task resolver**: T10 was matching T100_reverse_decoder. Fixed with underscore boundary (`T10_*` before `T10*`).
- **vllm-swap.sh symlink**: Created `~/dev/vllm-swap.sh` → `lab/models/vllm-swap.sh`
- **sam alias**: Fixed mangled bashrc alias

---

## Currently Running

Full profile (3 trials, 20 claw tasks, all suites) on:
- **Qwen3.5-4B INT4** (local)
- **Claude Sonnet 4.6** (cloud)
- **Claude Haiku 4.5** (cloud)

Check status:
```bash
# Background task output files
ls /tmp/claude-1001/-home-ava-dev-lab/*/tasks/*.output
# Or check results dirs
ls evals/results/ | tail -10
```

---

## vLLM Optimization Research — Priority Actions

### Priority 1: Free Wins (Add to ALL configs)
```bash
--async-scheduling              # ~30% throughput gain, zero risk (Blackwell-recommended)
--enable-prefix-caching         # TTFT 4.3s→0.6s on repeated prompts
--performance-mode interactivity  # Auto-tunes scheduler for latency
```

### Priority 2: Double KV Cache Capacity
```bash
--kv-cache-dtype fp8            # 2x effective KV cache (128K→200K+ context)
```
Low risk. Test quality first. No calibration needed (defaults to scale=1.0).

### Priority 3: MTP Speculative Decoding (Test Carefully)
```bash
--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```
Qwen3.5 natively supports Multi-Token Prediction. Big latency win BUT:
- Known bug: acceptance rate collapses during tool-calling sessions (issue #36872)
- Fix exists in PR #36910 — check if merged in 0.17.1
- Only `num_speculative_tokens: 1` works, value of 2 errors out
- **DO NOT use with tool-calling workflows until verified**

### Priority 4: TP=2 NCCL Tuning
```bash
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=8
```

### Priority 5: MoE-Specific (35B, 122B)
```bash
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
```

### Not Yet Viable
- **FlashAttention 4**: Integrated in vLLM 0.17.0 for Blackwell, but `flash-attn` package may not be installed. Worth checking.
- **NVFP4 quantization**: Native Blackwell FP4 tensor cores, but Mamba-hybrid layers must stay BF16 and output can be garbled. Stick with GPTQ-Int4.
- **Mistral MLA models**: Not supported on SM 12.0 (head_size=320 unsupported by all attention backends).

### Recommended A/B Test
1. Baseline: current qwen-27b-int4 config (44 tok/s)
2. Optimized: add `--async-scheduling --enable-prefix-caching --performance-mode interactivity --kv-cache-dtype fp8`
3. Run quick profile on both, compare tok/s and quality

---

## Model Inventory (`/mnt/models` — 244GB free)

### LLM Models (servable via vllm-swap.sh)

| Model | Size | tok/s | Quick Score | Role |
|-------|------|:-----:|:-----------:|------|
| **Qwen 27B INT4** | 29GB | 44 | 38/49 | Daily driver |
| **Qwen 35B MoE BF16** | 67GB | 170 | 41/49 | Speed king (TP=2) |
| **Qwen 122B INT4** | 74GB | ~30 | — | Quality ceiling |
| **Qwen 9B BF16** | 19GB | 92 | 42/49 | Fine-tune base |
| **Qwen 4B INT4** | 3.8GB | 294 | 38/49 | Edge deploy |
| **Qwen 4B BF16** | 8.8GB | 155 | — | LoRA base |
| Qwen 2B BF16 | 4.3GB | 307 | 21/49 | Training |
| Qwen 0.8B BF16 | 1.7GB | 547 | 13/49 | Training |
| Cydonia 24B | 44GB | — | 24/49 | Creative/roleplay only |
| Llama 70B AWQ | 38GB | 38 | — | Creative/roleplay |

### Base Models (for pretraining)
- Qwen3.5-0.8B-Base (1.7GB)
- Qwen3.5-2B-Base (4.3GB)
- Qwen3.5-4B-Base (8.8GB)

### Non-LLM Models
- Lightricks/LTX-2.3 (97GB) — video gen
- FLUX.2-klein-9B + base (100GB) — image gen
- Z-Image + Turbo (51GB) — image gen
- fishaudio/s2-pro (11GB) — TTS
- Voxtral-Mini-4B (17GB) — TTS
- Qianfan-OCR + GLM-OCR (11.4GB) — OCR

---

## What YOU Need To Do Next

### Immediate
- [ ] Wait for full profile runs to complete (4B INT4, Sonnet, Haiku)
- [ ] A/B test vLLM optimization flags on daily driver
- [ ] Roll out winning flags to all vllm-swap.sh configs
- [ ] Consider nuking Cydonia 24B (44GB, scored 24/49) if not using for creative work

### Eval Suite Refinement
- [ ] Review full profile results — identify tests that are too easy (5/5 everywhere) or too hard (0/5 everywhere) and recalibrate
- [ ] The `communication` dimension scores 0.00 across all claw tasks — investigate if this grader is broken or just strict
- [ ] T04 (calendar scheduling) fails for almost every model — check if the task or rubric needs adjustment
- [ ] Install Inspect AI for HumanEval/GSM8K/ARC standardized benchmarks: `uv pip install inspect-ai`

### Training Pipeline
- [ ] Install LLaMA-Factory in training workspace
- [ ] First fine-tune: Qwen3.5-9B LoRA on protoClaw tool-use traces from Langfuse
- [ ] Create HuggingFace dataset (`ArtificialCitizens/protoclaw-agent-v1`)
- [ ] Consider 2B as fine-tune target — if we can teach it tool calling, it's a 4GB model that reasons as well as 9B

### Models to Revisit
- [ ] Mistral-Small-4-119B — retry when vLLM ships MLA fixes for SM 12.0
- [ ] Qwen3-235B-A22B GPTQ-Int4 — ~120GB, might fit TP=2 (244GB free)
- [ ] DeepSeek-R1-Distill-Llama-70B — need AWQ quant (~38GB) or run TP=2 at BF16
