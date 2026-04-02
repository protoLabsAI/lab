# Agent Lightning APO — System Prompt Optimization

Automatic Prompt Optimization (APO) experiment for protoResearcher's SOUL.md system prompt, inspired by [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning).

## Background

### What is Agent Lightning?

[Agent Lightning](https://github.com/microsoft/agent-lightning) (16.3k stars, MIT) is Microsoft's framework for training AI agents with reinforcement learning. It provides three optimization paths:

| Path | What it does | GPU needed | Best for |
|------|-------------|:----------:|----------|
| **APO** | Beam search + textual gradients over prompts | No | Prompt iteration, fast wins |
| **VERL** | PPO/GRPO via veRL + vLLM + FSDP | Yes (40GB+) | Weight-level RL on agentic tasks |
| **SFT** | Supervised fine-tuning via Unsloth/TRL | Yes (16GB+) | Distillation, LoRA tuning |

The framework's core insight: decouple agent execution from training by capturing OpenTelemetry spans (LLM calls, tool calls, rewards) and converting them into training data.

### Why we built our own APO loop

Agent Lightning's plumbing is designed for its internal architecture: LLM proxy → trace capture → span-to-reward attribution → training backend. Our use case (optimizing an external agent's system prompt via API) fought the framework at every turn:

- `DummyTracer` doesn't implement `trace_context` (crashes the runner)
- `OtelTracer` + `ClientServerExecutionStrategy` times out on internal store startup
- `SharedMemoryExecutionStrategy` with `prompt_rollout` still routes through the trace pipeline
- APO requires a `TraceToMessages` adapter even when we have no traces to adapt

Rather than patching around the framework, we implemented the APO algorithm directly. It's ~200 lines of Python — beam search with LLM-generated textual gradients. The algorithm is identical to Agent Lightning's `APO` class; only the execution plumbing differs.

## Technique

### APO Algorithm (Textual Gradient Beam Search)

```
1. Evaluate seed prompt on validation tasks → baseline score
2. For each round (beam search):
   a. For each prompt in the beam:
      - Run training tasks → collect (task, score, response) tuples
      - Generate critique via critic model ("textual gradient")
      - Apply edit via critic model → produce candidate prompts
   b. Evaluate all candidates on validation tasks
   c. Keep top-k (beam width) → new beam
3. Best prompt across all rounds wins
```

### Architecture

```
                    ┌─────────────────────┐
                    │   Critic Model      │
                    │  (Claude Sonnet)    │
                    │   via ava gateway   │
                    └────────┬────────────┘
                             │ critique / edit
                             ▼
┌──────────┐    ┌─────────────────────┐    ┌──────────────────┐
│ Eval     │───▶│   APO Loop          │───▶│  Best SOUL.md    │
│ Tasks    │    │   (beam search)     │    │  (saved to disk) │
└──────────┘    └────────┬────────────┘    └──────────────────┘
                         │ run task
                         ▼
                ┌─────────────────────┐
                │  protoResearcher    │
                │  (Docker, :7872)    │
                │  Agent: Qwen 27B   │
                │  FP8 TP=2 on vLLM  │
                └─────────────────────┘
```

**Key design choice**: The critic model (Claude Sonnet via gateway) is different from the agent model (Qwen 27B via vLLM). This prevents the circular problem of a model critiquing its own blind spots. Our first run used the 27B as its own critic — it produced cosmetic changes (bold formatting, gentle rewording). Sonnet produced structural changes (mandatory protocols, output schemas, fallback chains).

### Reward Signal

#### v1: Deterministic Pattern Matching (original)

| Component | Weight | Metric |
|-----------|:------:|--------|
| Has content | 0.2 | Response > 20 chars |
| Has structure | 0.2 | Contains markdown markers (`**`, `##`, `- `, etc.) |
| Pattern match | 0.6 | Expected keywords found in response |

Pass threshold: score >= 0.75. Intentionally simple — tested the optimization loop, not comprehensive eval.

#### v2: LLM Judge (current)

Replaced pattern matching with Claude Sonnet 4.6 as a multi-dimensional judge:

| Dimension | Weight | What it measures |
|-----------|:------:|-----------------|
| Relevance | 0.25 | Does the response address the task? |
| Completeness | 0.25 | Does it cover the full scope? |
| Structure | 0.15 | Well-organized with clear formatting? |
| Accuracy | 0.20 | Factually correct and well-grounded? |
| Actionability | 0.15 | Practical value a researcher can act on? |

The LLM judge revealed that pattern matching was hiding real quality differences — both variants scored 1.0 on patterns, but the judge found the optimized prompt **regressed on complex tasks**.

## Results

### Run 1: 27B Self-Critique (pattern matching scorer)

| Metric | Value |
|--------|-------|
| Seed score | 0.500 |
| Best score | **1.000** |
| Delta | +0.500 |
| Best variant | r1-p0-b1 (round 1, branch 1) |

**Changes made**: Cosmetic formatting (bold bullets), "Concise but Complete" rewording, table-to-list conversion example, generic self-check step.

### Run 2: Claude Sonnet Critique (pattern matching scorer)

| Metric | Value |
|--------|-------|
| Seed score | 0.500 |
| Best score | **1.000** |
| Delta | +0.500 |
| Best variant | r1-p0-b0 (round 1, branch 0) |

**Changes made** (substantially different from 27B):
- **Mandatory Search Log** — every tool-driven response starts with an audit trail
- **Output Schema** — structured template for model/repo discovery tasks
- **Tool Failure Fallback Protocol** — 5-step escalation chain (primary → rephrase → alternate tool → web_search → failure report)
- **"Protocol violation"** language — zero-tolerance for empty responses without exhausting fallbacks
- Completeness constraint on response endings

### A/B Eval: Original vs Optimized SOUL.md (LLM Judge)

**Date**: 2026-04-01 | **Judge**: Claude Sonnet 4.6 | **Agent**: Qwen 27B FP8 TP=2

| Metric | Original | Optimized | Delta |
|--------|:--------:|:---------:|:-----:|
| **Avg composite** | **0.510** | 0.444 | **-0.066** |
| Min composite | 0.000 | 0.000 | +0.000 |
| Avg latency (ms) | 45,374 | 37,631 | -7,743 |

**Winner: Original** (composite delta: -0.066)

#### By Category

| Category | Original | Optimized | Delta |
|----------|:--------:|:---------:|:-----:|
| Simple | 0.921 | **0.952** | +0.031 |
| Medium | 0.422 | **0.440** | +0.018 |
| Complex | **0.353** | 0.110 | **-0.243** |

#### Per-Task Breakdown

| Task | Cat | Original | Optimized | Delta | Winner |
|------|-----|:--------:|:---------:|:-----:|:------:|
| simple_question | simple | 0.870 | 0.940 | +0.070 | **opt** |
| discord_scan | medium | 0.275 | 0.355 | +0.080 | **opt** |
| hf_model_search | medium | 0.575 | 0.000 | -0.575 | **orig** |
| github_trending | medium | 0.000 | 0.568 | +0.568 | **opt** |
| knowledge_topics | simple | 0.973 | 0.965 | -0.008 | — |
| paper_analysis | medium | 0.840 | 0.838 | -0.002 | — |
| multi_step_scan_analyze | complex | 0.455 | 0.330 | -0.125 | **orig** |
| digest_generation | complex | 0.605 | 0.000 | -0.605 | **orig** |
| complex_multi_tool | complex | 0.000 | 0.000 | +0.000 | — |

#### Key Insights from A/B Eval

1. **Pattern matching was masking real differences**: Both variants scored 1.000 on the pattern matcher, but the LLM judge reveals -0.066 composite regression overall.
2. **Optimized prompt overfitted to narrow training set**: APO was trained on 6 tasks; it improved on tasks similar to training (simple, discord) but regressed on out-of-distribution complex tasks.
3. **Complex tasks are dominated by tool failures**: Both variants score 0.000 on `complex_multi_tool` (tool timeouts), injecting noise into the comparison.
4. **Latency improved**: Optimized prompt is 17% faster (37.6s vs 45.4s avg), possibly because the stricter output schema produces shorter responses.
5. **The real problem is tool reliability, not prompt quality**: 3 of 9 tasks have 0.000 scores due to tool failures, not prompt issues. Fixing tools would be higher-leverage than further prompt optimization.

### Subagent Baseline Scores (LLM Judge)

| Subagent | Task | Composite | Notes |
|----------|------|:---------:|-------|
| **Explorer** | github_trending | 0.000 | Tool failure |
| **Analyst** | paper_analysis | 0.865 | Strong |
| **Analyst** | multi_step_scan_analyze | 0.445 | Weak (multi-step coordination) |
| **Writer** | digest_generation | 0.718 | Good |
| **Writer** | complex_multi_tool | 0.000 | Tool failure / timeout |

**Analyst** is the best APO target: strongest signal (no tool failures), clear room to improve on multi-step tasks.

## Multi-Prompt Optimization

### Optimizable Prompt Surfaces

Beyond SOUL.md, three additional prompt surfaces were identified:

1. **Subagent system prompts** (`graph/subagents/config.py`) — Explorer, Analyst, Writer role definitions with workflows and rules. These directly control agent behavior for discovery, analysis, and synthesis.
2. **Memory consolidation prompt** (`nanobot/agent/memory.py`) — Controls what gets extracted from conversations into persistent memory. Optimization could improve knowledge retention.
3. **Tool descriptions** (individual tool modules) — How tools are described to the LLM affects tool selection and usage patterns.

### Analyst Subagent APO

Target: `analyst` subagent prompt | 2 rounds, beam width 2, branch factor 2

| Metric | Value |
|--------|-------|
| Seed score | 0.649 |
| Best score | **0.715** |
| Delta | **+0.066** (+10.2%) |
| Best variant | r2-p1-b0 |

**What Sonnet changed** (original → optimized):

| Aspect | Original (20 lines) | Optimized (91 lines) |
|--------|---------------------|----------------------|
| Source types | Papers only | Papers, web, Discord, RSS, raw text |
| Tool inventory | Implicit | **Step 0: Tool Inventory Check** — verify available tools before starting |
| Significance rating | 4 tiers, no criteria | **Significance Rating Criteria table** — each tier has explicit evidence requirements |
| Output format | Unstructured | **Mandatory structured template** — success and failure formats |
| Failure handling | None | **Failure output template** — tools attempted, blocker, partial findings, recommended next step |
| Fallback strategy | None | **"Fallback before failing"** — try alternatives before reporting blockers |
| Silent stall prevention | None | **"Never stall silently"** — always return structured output, even on failure |

The optimized analyst prompt addresses the exact weakness seen in multi-step tasks (0.425 → 0.590 on `multi_step_scan_analyze`): the original prompt had no failure handling or fallback strategy, causing the agent to stall when tools didn't work as expected. The new prompt forces explicit fallback chains and structured failure reports.

## VERL Path — Feasibility Assessment

### Summary

VERL (Volcano Engine RL) is **feasible but complex** on our 2x RTX PRO 6000 Blackwell setup (192GB VRAM total).

### Compatibility

| Component | Status | Notes |
|-----------|:------:|-------|
| vLLM 0.10.x | ✅ | VERL supports vLLM >= 0.8.0 |
| Qwen 3.5 9B | ✅ | Supported; GRPO demonstrated on Qwen 2.5+ |
| 2-GPU FSDP | ✅ | Demonstrated on 2x H100 with 7B models |
| Agent Lightning integration | ✅ | Supports external agents via OpenAI-compatible proxy |

### Memory Budget (9B Actor-Critic-Reference, FSDP on 2 GPUs)

| Component | Per-GPU (FSDP sharded) |
|-----------|:---------------------:|
| Actor model (9B BF16) | ~11 GB |
| Critic model (9B BF16) | ~11 GB |
| Reference model (read-only) | ~18 GB (or CPU-offload) |
| Optimizer states (Adam) | ~22 GB |
| Activations + KV cache | ~15-20 GB |
| **Total per GPU** | **~77-82 GB** |
| **Available per GPU** | **96 GB** |

**Verdict**: Feasible with FSDP + gradient checkpointing. Tight but workable with ~14-19 GB headroom per GPU.

### Recommended Approach: TRL GRPOTrainer (Simpler Alternative)

Given our constraints, **TRL GRPOTrainer is recommended over full VERL** for the initial RL experiment:

| Factor | VERL | TRL GRPO |
|--------|:----:|:--------:|
| Setup complexity | High (FSDP, collective RPC) | Low (HF ecosystem) |
| 2-GPU support | Yes (FSDP/TP) | Yes (accelerate) |
| Documentation | Good but evolving | Excellent |
| Integration effort | Agent Lightning adapter | Direct rollout → reward → train |
| Blackwell compatibility | Untested | PyTorch-native, likely fine |

### Proposed Pipeline

```
protoResearcher tasks → rollout collection → LLM judge rewards → TRL GRPOTrainer
                                                                      ↓
                                                               Qwen 9B fine-tuned
                                                                      ↓
                                                            Evaluate vs baseline
```

1. Collect 100-500 rollouts from protoResearcher tasks
2. Score each with the LLM judge (Sonnet composite score as reward)
3. Train Qwen 9B with GRPO using these rewards
4. Evaluate fine-tuned model vs baseline on the full task suite

This pipeline reuses the LLM judge we already built, avoids VERL's infrastructure complexity, and can be upgraded to full VERL later if results warrant it.

## Configuration

```bash
# === SOUL.md APO ===

# Dry run (evaluate seed only)
PYTHONUNBUFFERED=1 .venv/bin/python experiments/agent-lightning/run_apo.py --dry-run

# Full run with Sonnet critic + LLM judge
export GATEWAY_API_KEY=<from-infisical>
PYTHONUNBUFFERED=1 .venv/bin/python experiments/agent-lightning/run_apo.py \
  --rounds 2 --beam-width 2 --branch-factor 2 \
  --critic-model "claude-sonnet-4-6" \
  --use-llm-judge

# === A/B Eval ===
.venv/bin/python experiments/agent-lightning/run_ab_eval.py
.venv/bin/python experiments/agent-lightning/run_ab_eval.py --tasks simple_question,paper_analysis

# === Multi-Prompt APO (Subagents) ===
.venv/bin/python experiments/agent-lightning/run_multi_apo.py --target analyst
.venv/bin/python experiments/agent-lightning/run_multi_apo.py --target explorer --dry-run
.venv/bin/python experiments/agent-lightning/run_multi_apo.py --target writer --dry-run
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rounds` | 2 | Beam search rounds |
| `--beam-width` | 2 | Top-k prompts kept per round |
| `--branch-factor` | 2 | Candidate prompts generated per parent |
| `--critic-model` | `claude-sonnet-4-6` | Model for critique/edit (via gateway) |
| `--use-llm-judge` | off | Use Sonnet LLM judge instead of pattern matching |
| `--dry-run` | — | Evaluate seed prompt only |
| `--target` | — | (multi-prompt) Subagent to optimize: explorer, analyst, writer |

### Infrastructure

- **Agent model**: Qwen 27B FP8 TP=2 on protolabs vLLM (:8000)
- **Agent endpoint**: protoResearcher Docker container (:7872)
- **Critic model**: Claude Sonnet 4.6 via ava gateway (:4000)
- **Judge model**: Claude Sonnet 4.6 via ava gateway (:4000)
- **Eval tasks**: `~/dev/protoResearcher/evals/tasks.json` (10 tasks)
- **Optimized prompts**: Written back to source files (bind-mounted)

## Files

```
experiments/agent-lightning/
├── README.md                  — this file
├── llm_judge.py               — Sonnet 4.6 LLM judge (5 dimensions)
├── run_apo.py                 — SOUL.md APO (pattern matching + LLM judge)
├── run_ab_eval.py             — A/B eval harness (original vs optimized)
├── run_multi_apo.py           — Multi-prompt APO for subagents
├── collect_rollouts.py        — Rollout collection for GRPO training
├── train_grpo.py              — TRL GRPOTrainer (LoRA on Qwen 9B)
└── results/
    ├── original_soul.md       — seed SOUL.md (pre-optimization)
    ├── best_soul.md           — best SOUL.md (Sonnet-optimized)
    ├── log.json               — beam search scores per round
    ├── ab_eval_*.json         — A/B eval raw results
    ├── ab_eval_*.md           — A/B eval markdown reports
    ├── multi-prompt/          — subagent optimization results
    └── rollouts/              — collected rollouts (JSONL) for GRPO
```

## Tool Reliability Fixes (2026-04-01)

A/B eval revealed 3/9 tasks consistently scoring 0.000 due to tool failures. Root causes and fixes:

| Tool | Issue | Fix |
|------|-------|-----|
| `github_trending` | Empty `GITHUB_TOKEN` → 10 req/min rate limit | Set token in `.env`, added retry with backoff |
| `huggingface` | 15s timeout too low under load | Bumped to 30s, added retry (2 attempts) |
| `knowledge/store.py` | 10s embed timeout, silent failure → empty results | Bumped to 20s, log failures, **fallback to BM25 keyword search** |
| All tools | No retry logic | Added `_api_get()` with 2 retries + exponential backoff |

**Post-fix A/B eval** (run 2): Overall improvement — original avg 0.510→0.560, discord_scan 0.275→0.560. Some tasks remain non-deterministic (agent behavior variance, not tool failures).

## GRPO Training Pipeline

### Overview

Weight-level RL on Qwen 3.5 9B using protoResearcher rollouts as training data, scored by the Sonnet LLM judge.

```
protoResearcher tasks ──→ collect_rollouts.py ──→ rollouts.jsonl
                                                      │
                                      ┌────────────────┘
                                      ▼
                              Sonnet 4.6 judge scores
                                      │
                                      ▼
                              train_grpo.py (TRL GRPOTrainer)
                                      │
                              ┌───────┴───────┐
                              ▼               ▼
                       GPU 0: vLLM      GPU 1: Training
                       (generation)     (LoRA on 9B)
                              │
                              ▼
                       Qwen 9B + LoRA adapter
                              │
                              ▼
                       Evaluate vs baseline
```

### Step 1: Collect Rollouts

```bash
cd ~/dev/lab
export GATEWAY_API_KEY=<from-infisical>  # for Sonnet judge
.venv/bin/python experiments/agent-lightning/collect_rollouts.py --rounds 10
```

Outputs: `results/rollouts/rollouts_<timestamp>.jsonl` — one JSON per line with prompt, response, reward, judge scores.

### Step 2: Train with GRPO

```bash
source ~/dev/vllm-env/bin/activate  # has TRL + PyTorch + vLLM

# Option A: Online (vLLM generates, GRPO trains)
# Terminal 1: vLLM server on GPU 0
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen3.5-9B --port 8001 --gpu-memory-utilization 0.85

# Terminal 2: Training on GPU 1
CUDA_VISIBLE_DEVICES=1 python experiments/agent-lightning/train_grpo.py \
  --rollouts experiments/agent-lightning/results/rollouts/rollouts_*.jsonl

# Option B: Offline (pre-collected completions, no vLLM needed)
CUDA_VISIBLE_DEVICES=0 python experiments/agent-lightning/train_grpo.py \
  --rollouts results/rollouts/rollouts_*.jsonl --offline
```

### GRPO Run 1 Results (Offline — Failed)

| Metric | Value |
|--------|-------|
| Steps | 72 (24 examples x 3 epochs) |
| Step time | 19.6s |
| Total time | 23.5 min |
| Loss | **0.0** (no gradient signal) |
| Reward mean | **0.5** (constant) |
| Reward std | **0.0** |

**Root cause**: GRPO generates NEW completions each step and calls the reward function. Our placeholder `reward_fn` returned constant 0.5. With zero reward variance, GRPO computes zero advantage → zero loss → zero gradient. The adapter is the identity.

**Lesson**: GRPO is fundamentally online — it needs a live reward function. For offline training from pre-scored data, use DPO (preference pairs). For GRPO, the reward function must actually score completions.

### Fix: Online GRPO with LLM Judge

The correct pipeline requires the reward function to call the LLM judge in real-time:

```python
# In train_grpo.py, replace placeholder reward_fn with:
async def _judge_completion(prompt, completion):
    task = {"prompt": prompt, "category": "unknown", "expected_tools": [], "expected_patterns": []}
    score = await llm_judge.judge_response(task, completion, client=judge_client)
    return score.composite
```

This requires:
1. vLLM serving 9B on GPU 0 (for generation)
2. Training on GPU 1
3. Gateway access (for Sonnet judge scoring each generated completion)
4. Significant Sonnet API cost: 72 steps x 4 generations x $0.003/call ≈ $0.86/run

### Alternative: DPO from Rollouts

Convert collected rollouts into preference pairs (chosen = high-reward, rejected = low-reward) and use TRL DPOTrainer instead. This is purely offline and doesn't need a live reward function.

### Step 3: Evaluate

```bash
# Swap to fine-tuned model (base + LoRA adapter)
# Compare eval scores against baseline 9B
```

### Training Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen3.5-9B | Dense, fine-tune base |
| LoRA rank | 16 | Targets q/k/v/o/gate/up/down_proj |
| Learning rate | 1e-6 | Conservative for RL |
| KL penalty (beta) | 0.04 | Prevents distribution collapse |
| Batch size | 2 x 4 grad_accum = 8 effective | Fits in 96GB |
| Generations/prompt | 4 | For online GRPO variance |
| Epochs | 3 | Over collected rollouts |
| Precision | BF16 | Native on Blackwell |

## Few-Shot Bootstrap (DSPy-Inspired)

Inspired by DSPy MIPROv2's bootstrap stage: before running APO's critique-edit loop, inject high-scoring rollout examples directly into the prompt. The hypothesis is that showing the model what a great response looks like is higher-leverage than abstract instruction editing alone.

### How It Works

```
rollouts.jsonl ──→ bootstrap_fewshot.py ──→ select top-K examples
                                                  │
                                          ┌───────┘
                                          ▼
                                  inject into SOUL.md
                                  (before APO starts)
                                          │
                                          ▼
                                  run_apo.py --bootstrap
```

1. **Selection**: Top-K rollouts by reward, with diversity preference (one per category first, then fill)
2. **Formatting**: Examples rendered as markdown reference blocks with task, response, and score
3. **Injection**: Appended to system prompt before any closing section

### Usage

```bash
# Preview bootstrap examples
.venv/bin/python experiments/agent-lightning/bootstrap_fewshot.py \
  --rollouts "experiments/agent-lightning/results/rollouts/rollouts_*.jsonl" \
  --top-k 3 --min-reward 0.5

# APO with bootstrap
.venv/bin/python experiments/agent-lightning/run_apo.py \
  --use-llm-judge --bootstrap --bootstrap-k 3

# Preview augmented prompt
.venv/bin/python experiments/agent-lightning/bootstrap_fewshot.py \
  --rollouts "experiments/agent-lightning/results/rollouts/rollouts_*.jsonl" \
  --show-prompt
```

### Key Design Decisions

- **Diverse selection**: One example per category (simple/medium/complex) prevents bias toward easy tasks
- **Truncation**: Responses capped at 800 chars to avoid bloating the system prompt
- **Minimum reward**: Default 0.5 threshold ensures only genuinely good examples are injected
- **Composable**: Works independently of APO — can also inject into subagent prompts

## DPO Training Pipeline

Direct Preference Optimization from collected rollouts — the offline alternative to GRPO that doesn't need a live reward function.

### Overview

```
rollouts.jsonl ──→ train_dpo.py ──→ preference pairs (chosen/rejected)
                                          │
                                          ▼
                                    DPOTrainer (TRL)
                                    LoRA on Qwen 9B
                                          │
                                          ▼
                                    adapter weights
```

### Pair Construction

Within-task pairs: same task_id, different rounds → pair high-reward response (chosen) with low-reward response (rejected). Minimum reward spread threshold (default 0.1) ensures meaningful preference signal.

Optional cross-task pairs: within same category, top quartile vs bottom quartile — weaker signal but more training data.

From 200 rollouts (25 rounds × 8 tasks):
- **901 within-task pairs** (spread range: 0.100–0.807, avg 0.482)
- **+297 cross-task pairs** (with `--cross-task`)
- Categories: 391 complex, 807 medium (simple tasks all score 0.94 — no spread)

### Usage

```bash
source ~/dev/quant-env/bin/activate  # transformers 5.5, TRL 1.0

# Dry run (validate pairs + config)
python experiments/agent-lightning/train_dpo.py \
  --rollouts "experiments/agent-lightning/results/rollouts/rollouts_*.jsonl" \
  --dry-run

# Train on GPU 1 (GPU 0 serving vLLM)
CUDA_VISIBLE_DEVICES=1 python experiments/agent-lightning/train_dpo.py \
  --rollouts "experiments/agent-lightning/results/rollouts/rollouts_*.jsonl" \
  --cross-task

# With custom hyperparams
CUDA_VISIBLE_DEVICES=1 python experiments/agent-lightning/train_dpo.py \
  --rollouts "experiments/agent-lightning/results/rollouts/rollouts_*.jsonl" \
  --cross-task --beta 0.05 --lr 1e-6 --epochs 5
```

### Training Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen/Qwen3.5-9B | Dense, fine-tune base |
| LoRA rank | 16 | Targets q/k/v/o/gate/up/down_proj |
| Learning rate | 5e-7 | Conservative for preference learning |
| Beta | 0.1 | KL penalty (lower = more aggressive) |
| Batch size | 2 x 4 grad_accum = 8 effective | |
| Max length | 1024 tokens | Prompt + response |
| Loss | Sigmoid | Standard DPO loss |
| Precision | BF16 | Native on Blackwell |
| Epochs | 3 | Over preference pairs |

### GRPO vs DPO

| Aspect | GRPO | DPO |
|--------|------|-----|
| Mode | Online (generates fresh completions) | Offline (uses pre-collected pairs) |
| Reward | Live reward function needed | Pre-scored, no live calls |
| Data | Can generate infinite data | Limited to collected rollouts |
| Cost | vLLM + judge API per step | One-time rollout collection cost |
| Signal | Explores new behaviors | Optimizes between known behaviors |
| Status | Infrastructure ready, needs live judge | **Working end-to-end** |

### DPO Run 1 (36 pairs)

| Metric | Value |
|--------|-------|
| Pairs | 36 (26 within-task, 10 cross-task) |
| Steps | 15 |
| Train loss | 0.693 → 0.510 (epoch 1 low) → 0.66 avg |
| Reward accuracy | Peak 87.5% |
| Reward margins | Peak 0.539 |
| Total time | 2m 11s |

**Key finding**: Real gradient signal — loss moved, gradients flowed, margins showed preference learning. Unlike GRPO's zero-gradient failure.

### DPO Run 2 (1198 pairs)

| Metric | Value |
|--------|-------|
| Pairs | 1198 (901 within-task, 297 cross-task) |
| Steps | 450 |
| Train loss | 0.693 → 0.5959 avg |
| Reward accuracy | Frequently 0.625–0.875 |
| Total time | 74 min |
| Adapter | `/mnt/data/training/researcher/dpo-9b-v2/` (111 MB LoRA) |

Much more stable than Run 1 — less oscillation with 33x more data.

### DPO Serving (Blocked)

vLLM's native LoRA serving fails on Qwen3.5 due to merged QKV projections (`IndexError` in `column_parallel_linear.py`). Merging the adapter into the base model produces a text-only `Qwen3_5ForCausalLM` model, but vLLM only registers the multimodal `Qwen3_5ForConditionalGeneration` wrapper. A merged model with the multimodal config + vision weights was created but produces empty output — the LoRA was trained under transformers 5.5's text-only weight paths which don't align with the multimodal wrapper. Needs proper investigation.

## Bayesian Combination Search (Optuna)

Search over combinations of independently-optimized subagent prompts using Optuna's TPE sampler. Inspired by DSPy MIPROv2.

### Results (4 trials, 2 combinations)

Only the analyst subagent has an optimized variant (explorer and writer only have original).

| Trial | Analyst | Score | Notes |
|-------|---------|:-----:|-------|
| 0 | optimized | 0.329 | github_trending=0.000 |
| 1 | original | 0.352 | multi_step=0.662 |
| 2 | original | 0.336 | |
| 3 | **optimized** | **0.439** | github_trending=0.655 |

**Winner**: analyst=optimized (0.439), but variance is high (0.329–0.439 for same config) driven by flaky external tools (github_trending, hf_model_search score 0.000 intermittently).

### Usage

```bash
# Dry run (list variants)
.venv/bin/python experiments/agent-lightning/search_combos.py --dry-run

# Run search (4 trials)
.venv/bin/python experiments/agent-lightning/search_combos.py --trials 4
```

## Future Work

- **Fix DPO adapter serving**: Debug Qwen3.5 multimodal/text weight prefix mismatch. Options: retrain LoRA targeting `Qwen3_5ForConditionalGeneration` weight paths, or use custom model class with `--trust-remote-code`.
- **Optimize explorer + writer subagents**: Run multi-prompt APO for both. Unlocks meaningful combo search (8 combinations vs current 2).
- **Re-run combo search with all 3 optimized**: 2^3 = 8 combinations, 8–16 trials needed.
- **Minibatch evaluation**: Evaluate on random subset of tasks per APO step instead of all tasks — faster iterations, less overfitting. (From DSPy.)
- **Online GRPO with live judge**: Wire LLM judge as real-time reward function. Requires vLLM 9B on GPU 0, training on GPU 1, Sonnet via gateway (~$1/run).
- **Cross-validation**: Use k-fold task splits to detect overfitting during APO.
- **Gradio workbench**: General-purpose prompt optimization → RL training app (deferred until flow is proven and abstractions are stable).
