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

1. Download with: `source ~/dev/vllm-env/bin/activate && HF_HOME=/mnt/models/huggingface huggingface-cli download <model-id>`
2. Add a swap config to `~/dev/experiments/vllm-swap.sh`:
   - Use `--served-model-name local` (required for gateway routing)
   - Pick the right tool-call parser:
     - Qwen3/3.5 models: `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml`
     - Llama 3.x: `--enable-auto-tool-choice --tool-call-parser llama3_json`
     - Llama 4: `--enable-auto-tool-choice --tool-call-parser llama4_json`
     - Hermes-format: `--enable-auto-tool-choice --tool-call-parser hermes`
   - Single GPU: `CUDA_VISIBLE_DEVICES=0`, no `--enforce-eager` (CUDA graphs work on Blackwell SM 12.0)
   - TP=2: `--tensor-parallel-size 2 --disable-custom-all-reduce --enforce-eager`
   - Don't use `--attention-backend flashinfer` (crashes on SM 12.0)

## Step 3: Swap & Verify

1. Run `./vllm-swap.sh <config-name>` and verify it starts
2. Check VRAM: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`
3. Quick smoke test: `curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"local","messages":[{"role":"user","content":"hello"}],"max_tokens":50}'`
4. Verify tool calling works with a tool-bearing request

## Step 4: Claw-Eval Benchmark (Agent Tasks)

Run the standard 4-task benchmark with port offset (port 9100 conflicts with node-exporter):

```bash
cd ~/dev/evals && ./run.sh claw --model local --tasks T02,T04,T06,T08 --port-offset 200
```

Record: task_score per trial, pass^3, wall time per trial.

**Baseline comparison (corrected, with CUDA graphs):**

| Model | T02 | T04 | T06 | T08 | Avg | pass^3 | tok/s |
|-------|-----|-----|-----|-----|-----|--------|-------|
| Qwen 35B MoE BF16 TP=2 | 0.87 | 0.53 | 0.85 | 0.86 | 0.78 | 3/4 | 170 |
| Qwen 27B INT4 | 0.87 | 0.53 | 0.86 | 0.86 | 0.78 | 3/4 | 44 |
| Qwen 122B INT4 1GPU | 0.87 | 0.52 | 0.85 | 0.88 | 0.78 | 3/4 | ~30 |
| OmniCoder 9B | 0.79 | 0.54 | 0.86 | 0.85 | 0.76 | 2/4 | 92 |
| Llama 70B AWQ | 0.71 | 0.52 | 0.58 | 0.79 | 0.65 | 1/4 | 38 |

## Step 5: Custom Suite Benchmarks

Run creative writing, roleplay, research, function calling:

```bash
cd ~/dev/evals
./run.sh custom --suite creative_writing --model local --trials 1
./run.sh custom --suite roleplay --model local --trials 1
./run.sh custom --suite research --model local --trials 1
./run.sh custom --suite coding --model local --trials 1
./run.sh custom --suite svg_generation --model local --trials 1
./run.sh function-call --model local --all-suites
```

## Step 6: Speed Benchmark

```bash
API_KEY="${GATEWAY_API_KEY}"
PROMPT='{"model":"local","messages":[{"role":"user","content":"Write a 500-word essay about computing history."}],"max_tokens":800}'
# Warmup, then 3 timed runs
```

## Step 7: Report & Decision

Compile results into a comparison table. Decide:
- **Keep**: Add to vllm-swap.sh permanently, update CLAUDE.md
- **Nuke**: `rm -rf /mnt/models/huggingface/hub/models--<org>--<model>/`

## Key Findings from Our Testing

### What works on Blackwell (SM 12.0):
- CUDA graphs on single GPU: ~37-470% speedup depending on model (MoE benefits most)
- GPTQ-Int4 quantization: no quality loss on dense models (Qwen 27B)
- INT4 MoE: causes instability (fluke 0.00 scores) — use BF16 for MoE models
- `--disable-custom-all-reduce` always needed for TP=2 (PCIe, not NVLink)

### What doesn't work:
- `--attention-backend flashinfer` — crashes on startup
- `--enforce-eager` removal on TP=2 — memory corruption under sustained load
- Ollama GGUF models — tool calling not parsed by LiteLLM
- MiroThinker (Qwen3 base) — tool calls output as text, no parser matches
- INT4 on large MoE (122B) — CUDA graph capture crashes
- Community AWQ quants — often broken weight keys (Llama 4 Scout)

### Port 9100 conflict:
Claw-eval Gmail mock defaults to port 9100 which conflicts with node-exporter.
**Always use `--port-offset 200`** for claw-eval runs.

### MoE vs Dense quantization:
- Dense models (27B): INT4 is quality-neutral, faster, smaller — always use INT4
- MoE models (35B, 122B): INT4 causes routing instability — use BF16 or FP8

### Power management:
- Inference draws 300-340W per GPU regardless of power limit (up to 500W)
- Safe to run TP=2 at 300W per card on current 1000W UPS
- 1600W UPS coming — can set both to 600W and forget
