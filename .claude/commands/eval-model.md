---
description: Evaluate a new LLM model for the protoLabs system — download, configure, benchmark, and compare
argument-hint: <model-name-or-huggingface-id>
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Agent, WebSearch, WebFetch]
---

# Model Evaluation Pipeline

You are the protoLabs model evaluation engineer. When a user wants to evaluate a new LLM model, follow this systematic pipeline.

## Arguments

The user wants to evaluate: $ARGUMENTS

## Step 1: Model Discovery & Feasibility

1. Search HuggingFace for the model and its quantizations (FP8, GPTQ-Int4, AWQ)
2. Check file sizes — determine what fits on our hardware:
   - **Single GPU**: up to ~90GB (96GB card, need KV room)
   - **TP=2**: up to ~180GB (both cards, needs UPS/power check)
   - Prefer INT4/GPTQ for dense models, BF16 for small MoE models
3. Check architecture: dense vs MoE, param count, active params
4. Check if vLLM supports it (architecture in vLLM model registry)
5. Check disk space on `/mnt/models` before downloading

## Step 2: Download & Configure

1. Download with: `source ~/dev/vllm-env/bin/activate && HF_HOME=/mnt/models/huggingface hf download <model-id>` (`huggingface-cli` is gone from the 0.22.1 env)
2. Add a swap config to `~/dev/lab/models/vllm-swap.sh`:
   - Use `--served-model-name local` (required for gateway routing)
   - Pick the right tool-call parser:
     - Qwen3/3.5 models: `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml`
     - Llama 3.x: `--enable-auto-tool-choice --tool-call-parser llama3_json`
     - Llama 4: `--enable-auto-tool-choice --tool-call-parser llama4_json`
     - Hermes-format: `--enable-auto-tool-choice --tool-call-parser hermes`
   - Qwen3.5 VLM models need `--language-model-only` for text-only serving
   - Single GPU: `CUDA_VISIBLE_DEVICES=0`, no `--enforce-eager` (CUDA graphs work on Blackwell SM 12.0)
   - TP=2: `--tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager`
   - Don't use `--attention-backend flashinfer` (crashes on SM 12.0)

## Step 3: Swap & Verify

1. Run `bash ~/dev/lab/models/vllm-swap.sh <config-name>` and verify it starts
2. Check VRAM: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`
3. Quick smoke test: `curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"local","messages":[{"role":"user","content":"hello"}],"max_tokens":200}'`
4. Verify tool calling works with a tool-bearing request

## Step 4: Quick Profile (Smoke Test)

Run the quick evaluation profile — covers all domains in ~15 min with 1 trial:

```bash
cd ~/dev/lab/evals && ./run.sh profile --name quick --model local
```

This runs:
- **Claw-eval**: 6 agent tasks (T02,T04,T06,T08,T10,T14)
- **Custom suites**: coding (10 tests), instruction_following (5), reasoning (5), structured_output (5), summarization (5), safety (5)
- **Function calling**: basic + edge cases (8 tests)

## Step 5: Full Profile (Comprehensive)

If the quick profile looks promising, run the full evaluation:

```bash
cd ~/dev/lab/evals && ./run.sh profile --name full --model local
```

This runs (single trial — **pass^3 dropped 2026-06-29**; full breadth discriminates better than 3× repetition. Use `--trials N` only for a targeted run-to-run consistency check on a small task set):
- **Claw-eval**: 30 agent tasks, 1 trial
- **All custom suites**: coding, instruction_following, reasoning, structured_output, summarization, safety, creative_writing, roleplay, svg_generation, research, + protolabs suites
- **Function calling**: all suites

## Step 6: Speed Benchmark

**speed-test-v2 is the standard for any number that leaves the lab** (InferenceMAX-style:
client-side `vllm bench serve`, seeded random dataset, TTFT/TPOT p50/p99, goodput at
TTFT≤2s+TPOT≤50ms). Single-stream-only numbers are banned from model cards — the dFlash
lesson (single-stream +43% inverted to 3× slower at the C=4–8 fan-out prod actually runs).

```bash
bash ~/dev/lab/models/speed-test-v2.sh quick 8000 <label> [tokenizer-path]  # 2 regimes × C{1,8}, ~10 min
bash ~/dev/lab/models/speed-test-v2.sh full                                 # 4 regimes × C{1,4,8,16,32}
bash ~/dev/lab/models/speed-test-v2.sh depth                                # decode@4/16/32/64K (needs 64K server)
bash ~/dev/lab/models/speed-test.sh        # v1 single-stream — legacy continuity only
```

- Pass the model dir as 4th arg when the served name isn't a resolvable HF id (bench client needs a tokenizer).
- **Spec-decode caveat:** MTP/EAGLE acceptance craters on the random-data bench (~31% vs ~76% real text) —
  the bench *understates* spec-decode speed; measure accept% separately on real prompts and say which is which.
- **depth tier** needs `--max-model-len ≥ 65536`; pair it with `evals/graders/verify_coherence.py`
  (same server session) so you never publish a tok/s@64K for a model that's babbling at 64K.

## Step 7: Report & Decision

Compile results into a comparison table. Decide:
- **Keep**: Add to vllm-swap.sh permanently, update CLAUDE.md model inventory
- **Nuke**: `rm -rf /mnt/models/huggingface/hub/models--<org>--<model>/`

## Baseline Comparison (Qwen3.5 Family)

| Model | Size | tok/s | Claw pass^3 | Coding | Role |
|-------|------|:-----:|:-----------:|:------:|------|
| **4B INT4** | 3GB | 294 | 2-3/4 | 2/5 | Edge deploy speed demon |
| **4B BF16** | 8GB | 155 | 3/4 | 2/5 | Edge deploy, LoRA base |
| **9B BF16** | 19GB | 92 | 2/4 | 3/5 | Fine-tune base (best coding) |
| **27B INT4** | 14GB | 44 | 3/4 | — | Daily driver, all-rounder |
| **35B MoE BF16** | 67GB | 170 | 3/4 | — | Speed king (TP=2) |
| **122B INT4** | 74GB | ~30 | 3/4 | — | Quality ceiling |
| 2B BF16 | 4GB | 307 | 0/4 | 1/5 | Training experiments only |
| 0.8B BF16 | 1.5GB | 547 | 0/4 | 0/5 | Training experiments only |

## Key Findings from Our Testing

### What works on Blackwell (SM 12.0):
- CUDA graphs on single GPU: ~37-470% speedup depending on model (MoE benefits most)
- GPTQ/AWQ-Int4 quantization: no quality loss on dense models (tested 27B + 4B)
- INT4 MoE: causes instability (fluke 0.00 scores) — use BF16 for MoE models
- `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink)

### What doesn't work:
- `--attention-backend flashinfer` — crashes on startup
- `--enforce-eager` removal on TP=2 — memory corruption under sustained load
- Ollama GGUF models — tool calling not parsed by LiteLLM
- MiroThinker (Qwen3 base) — tool calls output as text, no parser matches
- INT4 on large MoE (122B) — CUDA graph capture crashes
- Community AWQ quants — often broken weight keys (Llama 4 Scout)

### Eval-vs-baseline comparisons (2026-07-03 lessons — cost a whole night to learn):
- **Serve config parity before blaming the model.** FC scoring 8% = missing
  `--tool-call-parser qwen3_xml --enable-auto-tool-choice`. Claw erroring 45/105 = serving at
  16K context (agentic tool history needs ≥64K). Check `errored` in batch_summary — never let
  harness errors average into a score.
- **Paired task sets only.** A "claw 0.776" baseline means nothing until you know its task set
  (35-task vs 105-task differ wildly). Compare mean over the *intersection*, per-task.
- **Re-trial outliers ×3 on BOTH sides before believing any delta.** Single-trial small-n suites
  swing 3× (spec-delta 0.125→0.405; bf16 T12 0.88 single vs 0.70 ×3). The gate verdict uses
  re-trialed values on both sides.
- **T28_api_config_audit is excluded from gate scoring** — bimodal judge noise (0.0 in 18/22 runs,
  random 0.55–0.76 spikes on identical weights). The T28-class concern lives in `safety_agency`
  (deterministic none_of grader).
- **Verify judge + harness parity via the runs' snapshotted `config.yaml`s** (judge model/URL) and
  claw-eval submodule commit dates. Both runs must be graded by the same judge and code.
- **Steelman before any public claim:** read the actual failing transcript AND the task's grader
  before naming a failure mode (T12's "quant flips reimbursability judgment" was really "quant
  misses duplicate detection" — the card would have been wrong).

### Process gotchas:
- **Kill vLLM by exact PID only** (`ss -tlnp | grep <port>`), then kill the orphaned
  `VLLM::EngineCore` child separately (it holds ~50GB VRAM after the API server dies).
  **Never `pkill -f <pattern>`** — the pattern matches your own compound command and kills your shell (twice today).
- NVFP4/quant serving needs the env stack in `models/serve-nvfp4.sh` — use it, don't hand-type.

### Port 9100 conflict:
Claw-eval Gmail mock defaults to port 9100 which conflicts with node-exporter.
**Always use `--port-offset 200`** for claw-eval runs (profiles handle this automatically).

### MoE vs Dense quantization:
- Dense models (27B, 9B, 4B): INT4 is quality-neutral, faster, smaller — always use INT4
- MoE models (35B, 122B): INT4 causes routing instability — use BF16 or FP8

### Capability cliff:
- 4B+ models handle agentic tasks reliably (pass^3 2-3/4)
- 2B and below cannot reliably execute tool calls (pass^3 0/4)
- The 4B→2B drop is a cliff, not a slope
