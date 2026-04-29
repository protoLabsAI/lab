# Session Handoff — 2026-04-29

## Dual-GPU daily setup locked in

- GPU 0 — `vllm.service` — Qwen3.6-27B-FP8 thinking, :8000, 262K
- GPU 1 — `vllm-fast.service` — Qwen3.6-35B-A3B-FP8 instruct, :8002, 131K
- Both systemd units `enabled` (auto-start on boot)
- Tuned MoE kernels symlinked via `models/install-moe-configs.sh`
- Heretic 35B retired (HF repo deleted, 0 downloads); see CLAUDE.md "Thinking models" section for why

## Gateway-side thinking-normalizer (homelab-iac)

vLLM's `--reasoning-parser qwen3` is greedy — when the model fails to emit `</think>` (long agent loops, preserve_thinking), the answer lands in `reasoning_content` and `content` is empty. Confirmed upstream bug: [vllm#40528](https://github.com/vllm-project/vllm/issues/40528).

Fix in `stacks/ai/config/litellm/callbacks/thinking_normalizer.py` (PR #20, merged):
1. Salvage `reasoning_content → content` when content is empty
2. Strip inline `<think>...</think>` from content
3. Expose raw trace as top-level `reasoning` field (OpenRouter convention)

Follow-ups: PR #22 (null-content handling), PR #23 (drop logit_bias from `protolabs/fast` since it now uses official 35B that respects `enable_thinking:false` directly — open at session end).

## Eval-side cleanups (lab repo)

- claw-eval submodule (`a8767ef`, `4bc54c9`, `cef2e35` on `protoLabsAI/claw-eval`): tightened mock health check to `<400` to dodge node_exporter port collision; forwarded `extra_body` through to per-task graders; `Message.text` falls back to `reasoning_content` for graders that bypass the gateway.
- `evals/claw/config*.yaml` + `runners/run_claw.py`: judge defaults to `protolabs/fast` via `ava:4000` (single source of truth for heretic-era logit_bias / sampling, replaced by clean official-model defaults at gateway level).
- `runners/run_refusal.py`: dropped heretic-specific logit_bias hack from `classify_response` (was breaking the judge on the official model).

## XSTest 450 baseline on `local-fast` (official 35B, post-fix)

| | safe (250) | harmful (200) |
|---|---|---|
| comply | 67.2% | 0.5% |
| refuse | 31.2% | 99.5% |
| busted | 28 of 277 refusals are <20-char artifacts |

99.5% under-refusal is excellent. 31% over-refusal is mid-pack vs public XSTest baselines (GPT-4 ~25-30%, Llama-2-Chat 38-50%). Heaviest over-refusers: `privacy_fictional` (52%), `nons_group_real_discr` (44%), `homonyms` (32%). Improving this is system-prompt territory — not a model swap.

---

# Session Handoff — 2026-03-22

## What We Built

### 1. Comprehensive Eval Suite (128 custom tests + 72 claw + 8 FC)

| Suite | Tests | New? |
|-------|:-----:|:----:|
| **creative_writing** | 25 | +20 (narrative, character voice, genre, perspective) |
| **summarization** | 25 | +20 (news, technical, factual consistency) |
| **RAG** | 25 | +20 (faithfulness, multi-doc, domain-specific, edge cases) |
| **coding** | 10 | — |
| **instruction_following** | 5 | — |
| **reasoning** | 5 | — |
| **structured_output** | 5 | — |
| **safety** | 5 | — |
| **roleplay** | 5 | — |
| **svg_generation** | 5 | — |
| **research** | 4 | — |
| **function_call** | 8 | — |

New infrastructure: pairwise A/B judge, NIAH context window test, deepeval/ragas integration.

**Quick profile is FROZEN as baseline** — don't change test composition. Add new profiles instead.

### 2. Speed Optimization Testing

| Optimization | Result |
|-------------|--------|
| P1+P2 flags (async-scheduling, prefix-caching, FP8 KV) | +1-3% single-request (real wins under concurrency) |
| **MTP speculative decoding** | **+32% on 27B, +22% on 9B, -11% on 35B MoE** |
| NCCL tuning for TP=2 | Zero impact on decode speed |
| Prefix caching on TP=2 | TTFT -70% on 35B (1.8s→0.5s) |
| MoE FP8 env vars | Crashes on 122B FP8 quants, +2% on BF16 MoE |

### 3. Grand Model Comparison (Expanded Quick Profile)

| Model | Claw | Code | IF | Rea | SO | Summ/25 | Safe | Creat/25 | RP | FC | Total |
|-------|:----:|:----:|:--:|:---:|:--:|:-------:|:----:|:--------:|:--:|:--:|:-----:|
| **GPT-5.4** | 2/9 | 9 | 5 | 5 | 5 | 21 | 5 | 23 | 5 | 8 | **88/102** |
| **Sonnet 4.6** | 4/10 | 9 | 3 | 5 | 4 | 20 | 4 | **25** | 5 | 8 | **87/103** |
| **Gemini Flash** | 2/8 | 9 | 4 | 4 | 5 | **23** | 4 | 23 | 5 | 8 | **87/101** |
| **DeepSeek V3.2** | 2/8 | 9 | 4 | 5 | 4 | 21 | 4 | **25** | 5 | 8 | **87/101** |
| *27B INT4 MTP* | **5** | 8 | **5** | **5** | **5** | 19 | **5** | 21 | 5 | 8 | **86/103** |
| **Gemini 3.1 Pro** | 0/7 | **10** | 4 | 4 | 5 | 22 | 5 | 22 | 5 | 8 | **85/100** |
| **Grok 4.1 Fast** | 2/8 | 8 | 4 | 5 | 4 | 22 | 5 | 22 | 5 | 8 | **85/101** |
| *27B INT4* | **5** | 8 | 4 | 4 | **5** | 21 | **5** | 19 | 5 | 8 | **84/103** |
| **Haiku 4.5** | 3/8 | 9 | 2 | 4 | 3 | 22 | 4 | 23 | 5 | 8 | **83/101** |
| *35B MoE* | **6** | **10** | 4 | **5** | 4 | 19 | **5** | 11 | 4 | 8 | **76/103** |
| *9B MTP* | 5 | 8 | 4 | **5** | 4 | 19 | **5** | 11 | 3 | 8 | **72/103** |
| *4B INT4* | 5 | 7 | 3 | **5** | **5** | 14 | 4 | 2 | 3 | 8 | **56/103** |
| *Llama 8B AWQ* | 1/9 | 5 | 3 | 2 | 5 | 14 | 5 | 5 | 4 | 6 | **50/102** |

*italic = local model*

### 4. Models Tested and Eliminated

| Model | Why Eliminated |
|-------|---------------|
| OmniCoder 9B | Inferior to Qwen3.5-9B (same arch, worse tool use) |
| Mistral-Small-4-119B | MLA attention unsupported on Blackwell SM 12.0 |
| Qwen3.5-27B BF16 | Redundant with INT4 (same quality, 52GB wasted) |
| Qwen3.5-122B FP8 | Redundant with INT4 (INT4 is faster on single GPU) |
| Hermes 3 70B FP8 | FP8 crashes under sustained load, tool parser broken, all claw scores 0.00 |
| Llama 8B AWQ | 50/102 — worst model tested, Qwen 4B beats it at 1/4 the size |

### 5. Key Discoveries

**Creative writing is the great separator.** The expanded 25-test creative suite reveals massive quality gaps invisible in the old 5-test version:
- Frontier ceiling: 25/25 (Sonnet 4.6, DeepSeek V3.2)
- 27B MTP: 21/25 (competitive)
- 35B MoE: 11/25 (MoE routing hurts creative?)
- 9B: 11/25 (fine-tune target)
- 4B: 2/25 (capability cliff)

**Qwen owns everything local.** Llama 8B scored 50/102 vs Qwen 9B at 72/103. Hermes 70B couldn't even complete a run. All fine-tuning should be on Qwen architecture.

**27B MTP is 1 point behind frontier.** At 86/103 with 70 tok/s local, it's within striking distance of GPT-5.4 (88/102). The gap is creative (21 vs 23) and summarization (19 vs 21).

---

## Current Model Inventory (`/mnt/models` — 415GB free)

### Production LLMs

| Model | Size | tok/s | Quick Score | Config |
|-------|------|:-----:|:-----------:|--------|
| **Qwen 27B INT4** | 29GB | 53/70 MTP | 84-86/103 | `qwen-27b-int4` / `-mtp` |
| **Qwen 35B MoE** | 67GB | 171 | 76/103 | `qwen-35b` |
| **Qwen 9B** | 19GB | 92/112 MTP | 72/103 | `qwen-9b` / `-mtp` |
| **Qwen 4B INT4** | 3.8GB | 297 | 56/103 | `qwen-4b-int4` |
| **Qwen 122B INT4** | 74GB | ~30 (TP=2) | — | `qwen-122b-int4` |

### Training / Fine-tune Bases

| Model | Size | Purpose |
|-------|------|---------|
| Qwen 4B BF16 | 8.8GB | LoRA base |
| Qwen 4B Base | 8.8GB | Pretraining |
| Qwen 2B + Base | 8.6GB | Training experiments (5/5 reasoning at 2B!) |
| Qwen 0.8B + Base | 3.4GB | Training experiments |
| Llama 3.1-8B AWQ | 5GB | Eval baseline (poor: 50/102) |

### Creative / Experimental

| Model | Size | Notes |
|-------|------|-------|
| Cydonia 24B | 44GB | Uncensored Mistral, tool calling broken, holding for manual creative testing |
| Llama 3.3-70B AWQ | 38GB | Creative/roleplay, holding |

### Non-LLM

| Model | Size | Notes |
|-------|------|-------|
| LTX-2.3 | 97GB | Video gen (ComfyUI symlinked, keep on fast drive) |
| fishaudio/s2-pro | 11GB | TTS |

### Cold Storage (`/mnt/data/models-cold/`)

FLUX.2-klein 9B+base (100GB), Z-Image+Turbo (51GB), Voxtral-Mini-4B (17GB), OCR models (11.4GB)

---

## Dual-GPU Production Setup (Current — April 2026)

Two services run simultaneously on boot:

| Service | GPU | Port | Model | Role |
|---------|-----|------|-------|------|
| `vllm.service` | 0 | :8000 | Qwen3.6-27B-FP8 | Thinking / planning / reasoning (256K) |
| `vllm-voice.service` | 1 | :8002 | Qwen3.6-35B-A3B-FP8 | Executor / voice / agentic (256K, no thinking) |

Gateway names: `local` → :8000, `local-voice` → :8002.

No-thinking enforced via `models/templates/qwen3_nonthinking.jinja` (vLLM-level, not per-request).
TRITON_ATTN backend required on `vllm-voice` — FlashInfer JIT crashes on Blackwell SM 12.0.

MoE kernel tuned for Blackwell (2026-04-25) — see `docs/moe-tuning.md`.

```bash
sudo systemctl start vllm vllm-voice   # start dual-GPU setup
sudo systemctl stop vllm vllm-voice    # stop both
```

## Recommended Production Configs

```bash
# Daily driver — agentic work (tools, claw tasks)
bash models/vllm-swap.sh qwen-27b-int4       # 53 tok/s, reliable tools

# Daily driver — chat, creative, coding
bash models/vllm-swap.sh qwen-27b-int4-mtp   # 70 tok/s, +32% speed

# Fine-tune evaluation
bash models/vllm-swap.sh qwen-9b-mtp         # 112 tok/s, fine-tune baseline

# Edge deploy
bash models/vllm-swap.sh qwen-4b-int4        # 297 tok/s, 3.8GB

# Quality ceiling (needs both GPUs)
bash models/vllm-swap.sh qwen-122b-int4      # 30 tok/s, TP=2

# Speed king (single GPU or TP=2 for long context)
bash models/vllm-swap.sh qwen-35b            # 171 tok/s, best for batch/coding
```

---

## What YOU Need To Do Next

### Fine-Tuning (Primary Goal)
- [ ] Install LLaMA-Factory (`pip install llamafactory` in training venv)
- [ ] Accept Llama 3.1-8B-Instruct license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- [ ] Set up Langfuse online evaluators for protoClaw production traffic
- [ ] Export tool-use traces from Langfuse as training data
- [ ] First LoRA: Qwen3.5-9B on protoClaw tool-use traces
- [ ] Creative writing LoRA: Qwen3.5-9B on curated creative data (target: 11→20+/25)
- [ ] Create HuggingFace dataset: `ArtificialCitizens/protoclaw-agent-v1`

### Eval Refinement
- [ ] Run full profile (3 trials, pass^3) on 27B MTP — it's our SOTA local model
- [ ] Investigate claw scoring — many models score low, may be rubric/parser issues
- [ ] GPT-5.4-mini/nano need gateway restart with `drop_params: true` to retest
- [ ] Consider adding WildBench (1,024 tasks) as a periodic deep creative eval

### Infrastructure
- [ ] Rebuild protoClaw container for Langfuse tracing
- [ ] Set up Grafana dashboard for protoClaw metrics
- [ ] Concurrency testing — async-scheduling claims 30% but untested under load
- [ ] Install Inspect AI: `uv pip install inspect-ai` (HumanEval, GSM8K, ARC)

### Models to Revisit
- [ ] Qwen3-235B GPTQ-Int4 (~120GB) — would be our quality ceiling, 415GB free
- [ ] Mistral-Small-4-119B — retry when vLLM ships MLA fixes for SM 12.0
- [ ] Llama 3.1-8B-Instruct BF16 — for fine-tune experiments once license approved
