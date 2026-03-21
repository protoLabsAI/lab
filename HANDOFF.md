# Session Handoff — 2026-03-21

## What We Built Today

### 1. Eval Laboratory (`protoLabsAI/lab`)
Built a comprehensive LLM evaluation monorepo from scratch:
- **uv workspaces** with 5 packages: `lab-core`, `evals`, `models`, `training`, `experiments`
- **Pydantic models** for ModelSpec, BenchmarkResult, EvalResult, TrainingConfig
- **8 eval suites**: claw-eval (104 agent tasks), function-call, RAG, creative writing, roleplay/GM, research synthesis, coding, SVG generation
- **Grading framework**: outcome (deterministic), LLM-as-judge (model-based), function-call, RAG (4 dimensions), creative (narrative, character voice, world building, engagement)
- **Parallel model comparison** with `--port-offset` to avoid mock service port conflicts

### 2. Gateway Expansion (`infra/gateway/`)
- Added 14+ new models: GPT-5.4 family, Gemini 3.x, DeepSeek V3.2, GLM 5 Turbo, Kimi K2.5, MiMo V2 Flash, GPT-OSS, Grok (via OpenRouter)
- Fixed Gemini model IDs (preview → GA)
- Fixed vLLM routing (`openai/auto` → `openai/local` with `--served-model-name local`)
- Removed Ollama (tool calling doesn't work through LiteLLM)
- All Grok models moved to OpenRouter (no xAI API key)

### 3. protoClaw Observability
- **Langfuse tracing**: every LLM call + tool execution traced with session grouping
- **Prometheus metrics**: 6 collectors (`protoclaw_llm_calls_total`, latency, tokens, tool calls, etc.)
- **`/metrics` endpoint** on :7865 for Grafana scraping
- **Enhanced audit logging**: trace_id cross-reference, session stats

### 4. CUDA Graphs Discovery
Removing `--enforce-eager` on Blackwell SM 12.0 single-GPU configs:
- **Qwen 35B MoE**: 30 → **170 tok/s** (5.7x)
- **OmniCoder 9B**: 30 → **92 tok/s** (3.1x)
- **Qwen 27B INT4**: 30 → **44 tok/s** (1.5x)
- TP=2 still needs enforce-eager (memory corruption under sustained load)
- 122B MoE crashes with CUDA graphs (too many experts)

### 5. Model Benchmarking
Tested 20+ models across claw-eval agent tasks. Key findings:
- **INT4 quantization**: quality-neutral on dense models, unstable on MoE (use BF16)
- **Port 9100 conflict**: node-exporter occupies port, caused false failures on earlier runs
- **Opus distillation hurts agentic tasks**: sends emails instead of drafting, overthinks tool use
- **Community AWQ quants often broken**: Llama 4 Scout, MiniMax REAP both had weight mapping errors

### 6. Consolidated Repos
- Moved gateway into `lab/infra/gateway/` (archive standalone repo)
- Moved evals into `lab/evals/`
- Nuked: `~/dev/experiments/`, `~/dev/gateway/`, `~/dev/nanobot/`, `~/dev/protomaker/` (lowercase dup)

---

## Current Model Inventory (`/mnt/models` — 357GB free)

| Model | Size | tok/s | pass^3 | Role |
|-------|------|:-----:|:------:|------|
| **Qwen 27B INT4** | 14GB | 44 | 3/4 | Daily driver, all-rounder |
| **Qwen 35B MoE BF16** | 67GB | 170 | 3/4 (TP=2) | Speed king |
| **Qwen 122B INT4** | 74GB | ~30 | 3/4 | Quality ceiling |
| **Llama 70B AWQ** | 38GB | 38 | 1/4 | Creative/roleplay |
| **OmniCoder 9B** | 18GB | 92 | 2/4 | Fine-tune candidate |
| Qwen 27B BF16 | 52GB | 44 | 3/4 | Baseline (can nuke to save 52GB) |

---

## What YOU Need To Do

### Immediate
- [ ] **Restart Claude from `~/dev/lab`** (experiments dir was nuked)
  ```bash
  alias sam='cd ~/dev/lab && claude --dangerously-skip-permissions'
  ```
- [ ] **Archive `protoLabsAI/gateway`** on GitHub (Settings → Archive) — it's in lab now
- [ ] **Archive `protoLabsAI/evals`** on GitHub — it's in lab now
- [ ] **Rebuild protoClaw container** to activate Langfuse tracing + Prometheus metrics
  ```bash
  cd ~/dev/protoClaw && docker compose build && docker compose up -d
  ```
- [ ] **Add Langfuse keys to protoClaw** env (already in Infisical, just needs compose restart with `infisical run`)

### When UPS Arrives (1600W)
- [ ] Set GPUs to 600W: `sudo nvidia-smi -i 0,1 -pl 600`
- [ ] Test `qwen-35b-tp2` at 250K context (best config: 170 tok/s, 3/4 pass^3)
- [ ] Test `qwen-27b-int4-tp2` at 256K context (massive concurrency)
- [ ] Test `qwen-122b-int4` at TP=2 128K

### Eval Follow-ups
- [ ] Re-run **Grok 4.1 Fast** via OpenRouter (failed due to gateway restart mid-eval)
- [ ] Test **base Qwen3.5-9B** and compare against OmniCoder 9B
- [ ] Run full 104-task claw-eval sweep on top 3 local models
- [ ] Configure **Langfuse online evaluators** in UI (helpfulness, hallucination, tool use quality)
- [ ] Fix: evals `run.sh` needs update — venv path points to symlink, should use `uv run` instead

### Training Setup
- [ ] Install LLaMA-Factory in training workspace
- [ ] First fine-tune: OmniCoder 9B LoRA on protoClaw tool-use traces from Langfuse
- [ ] Create HuggingFace dataset (`ArtificialCitizens/protoclaw-agent-v1`)

### Infrastructure
- [ ] Create Grafana dashboard for protoClaw metrics (pve01)
- [ ] Set up Langfuse online evaluators for production traffic scoring
- [ ] Install Inspect AI benchmarks: `uv run --package protolabs-evals pip install inspect-ai`

---

## Ecosystem Overview

| Repo | Purpose | Status |
|------|---------|--------|
| **protoLabsAI/lab** | Monorepo: evals, models, training, infra | Active, public |
| **protoLabsAI/protoClaw** | Sandboxed AI agent | Active, Langfuse+Prometheus added |
| **protoLabsAI/protoMaker** | AI dev studio (agent Kanban) | Active |
| **protoLabsAI/mythxengine** | AI RPG engine | Active |
| **protoLabsAI/rabbit-hole.io** | Knowledge graph search | Active |
| **protoLabsAI/svgval** | SVG generation benchmark | Active |
| **protoLabsAI/homelab-iac** | Infrastructure as code | Active |
| **protoLabsAI/gateway** | LiteLLM proxy | → Archive (moved to lab) |
| **protoLabsAI/evals** | Eval suite | → Archive (moved to lab) |

---

## Key Technical Notes for Next Session

1. **vllm-swap.sh is at `~/dev/lab/models/vllm-swap.sh`** — symlinked from old experiments location but that symlink is broken now. Use the lab path directly or create a new symlink.

2. **Port 9100 conflict**: Always use `--port-offset 200` for claw-eval runs. node-exporter Docker container occupies port 9100 which is claw-eval's default Gmail mock port.

3. **MoE quantization rule**: Dense models → use INT4 (no quality loss). MoE models → keep BF16 (INT4 causes fluke 0.00 scores from routing corruption).

4. **CUDA graphs rule**: Single GPU → no enforce-eager (huge speedup). TP=2 → keep enforce-eager. 122B MoE → always enforce-eager (crashes without it).

5. **Gateway config lives in two places**: `~/dev/lab/infra/gateway/config.yaml` (git-tracked) and `~/dev/gateway/config.yaml` (live, read by Docker). After editing lab copy, sync: `cp ~/dev/lab/infra/gateway/config.yaml ~/dev/gateway/`

6. **Eval results directory**: Changed from `~/dev/evals/results/` to `~/dev/lab/evals/results/` (gitignored). Previous results are gone with the nuke.
