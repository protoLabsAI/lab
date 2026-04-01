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

Deterministic pattern matching from protoResearcher's existing eval harness — no LLM judge:

| Component | Weight | Metric |
|-----------|:------:|--------|
| Has content | 0.2 | Response > 20 chars |
| Has structure | 0.2 | Contains markdown markers (`**`, `##`, `- `, etc.) |
| Pattern match | 0.6 | Expected keywords found in response |

Pass threshold: score >= 0.75. This is intentionally simple — the goal is to test the optimization loop, not build a comprehensive eval.

## Results

### Run 1: 27B Self-Critique

| Metric | Value |
|--------|-------|
| Seed score | 0.500 |
| Best score | **1.000** |
| Delta | +0.500 |
| Best variant | r1-p0-b1 (round 1, branch 1) |

**Changes made**: Cosmetic formatting (bold bullets), "Concise but Complete" rewording, table-to-list conversion example, generic self-check step.

### Run 2: Claude Sonnet Critique

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

### Comparison

| Aspect | 27B Self-Critique | Sonnet Critique |
|--------|:-----------------:|:---------------:|
| Same final score | 1.000 | 1.000 |
| Nature of changes | Cosmetic/soft | Structural/protocol |
| Search Log added | No | Yes |
| Output schema added | No | Yes |
| Fallback protocol | No | Yes (5-step) |
| Would generalize to new tasks | Unlikely | Likely |

Both hit 1.000 on the val set, but Sonnet's edits are more robust — they add concrete protocols that should transfer to unseen tasks, not just surface-level formatting tweaks.

## Configuration

```bash
# Dry run (evaluate seed only)
PYTHONUNBUFFERED=1 .venv/bin/python experiments/agent-lightning/run_apo.py --dry-run

# Full run with Sonnet critic
export GATEWAY_API_KEY=<from-infisical>
PYTHONUNBUFFERED=1 .venv/bin/python experiments/agent-lightning/run_apo.py \
  --rounds 2 \
  --beam-width 2 \
  --branch-factor 2 \
  --critic-model "cli/claude-sonnet-4-6"

# Use local 27B as critic (cheaper, weaker)
PYTHONUNBUFFERED=1 .venv/bin/python experiments/agent-lightning/run_apo.py \
  --critic-model "protolabs/local"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rounds` | 2 | Beam search rounds |
| `--beam-width` | 2 | Top-k prompts kept per round |
| `--branch-factor` | 2 | Candidate prompts generated per parent |
| `--critic-model` | `cli/claude-sonnet-4-6` | Model for critique/edit (via gateway) |
| `--dry-run` | — | Evaluate seed prompt only |

### Infrastructure

- **Agent model**: Qwen 27B FP8 TP=2 on protolabs vLLM (:8000)
- **Agent endpoint**: protoResearcher Docker container (:7872)
- **Critic model**: Claude Sonnet via ava gateway (:4000)
- **Eval tasks**: `~/dev/protoResearcher/evals/tasks.json` (6 usable tasks)
- **Optimized prompt**: Written directly to `~/dev/protoResearcher/config/SOUL.md` (bind-mounted)

## Files

```
experiments/agent-lightning/
├── README.md              — this file
├── run_apo.py             — APO implementation (~200 lines)
└── results/
    ├── original_soul.md   — seed prompt (pre-optimization)
    ├── best_soul.md       — best prompt (Sonnet-optimized)
    └── log.json           — beam search scores per round
```

## Future Work

- **More eval tasks**: 6 tasks with pattern matching is a narrow signal. Add LLM-judge scoring and more diverse tasks.
- **VERL path**: RL training on the 9B model using protoResearcher rollouts as training data. Requires vLLM 0.10.2 compatibility check.
- **Multi-prompt optimization**: Optimize not just SOUL.md but also tool descriptions, retrieval prompts, and middleware injection templates.
- **A/B eval**: Run the full protoResearcher eval suite (10 tasks, nanobot vs langgraph) with original vs optimized SOUL.md.
