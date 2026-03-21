# Runbook: Full Eval Sweep

Run all eval suites against a model for a comprehensive quality assessment.

## When to Use

- Evaluating a new model before deployment
- Quarterly model review
- Comparing local vs cloud models for cost/quality tradeoff

## Procedure

### 1. Pick your model

```bash
# Local models — swap first
cd ~/dev/experiments
./vllm-swap.sh qwen-27b-int4   # or qwen-27b, qwen-35b, omnicoder

# Cloud models — no setup needed
# Available: claude-opus-4-6, claude-sonnet-4-6, gpt-5.4, gemini-2.5-pro, etc.
```

### 2. Run all suites

```bash
cd ~/dev/evals

# Agent tasks (claw-eval, ~15 min for local, ~5 min for cloud)
./run.sh claw --model local --tasks T02,T04,T06,T08,T20,T22,T24,T26,T28

# Function calling accuracy (~2 min)
./run.sh function-call --model local --all-suites

# RAG quality (~3 min)
./run.sh rag --model local --judge-model claude-sonnet-4-6

# General LLM quality (needs inspect-ai, ~30 min)
./run.sh general --model local --limit 50
```

### 3. Compare with baseline

```bash
./run.sh compare results/claw/local_* results/claw/claude-opus-4-6_*
```

Or use Langfuse UI at `http://localhost:3001` → Scores → filter by `eval/*`.

### 4. Record findings

Key metrics to capture:
- **pass^3 rate** across claw-eval tasks (agent reliability)
- **Function call accuracy** (tool use correctness)
- **RAG faithfulness** (hallucination rate)
- **Average wall time per trial** (speed)
- **Tokens per task** (cost efficiency)

## Parallel Cloud Sweep

Test multiple cloud models simultaneously:

```bash
./run.sh claw --model claude-opus-4-6 --tasks T02,T04,T06,T08 --port-offset 0 &
./run.sh claw --model gpt-5.4 --tasks T02,T04,T06,T08 --port-offset 200 &
./run.sh claw --model deepseek-v3.2 --tasks T02,T04,T06,T08 --port-offset 400 &
./run.sh claw --model glm-5-turbo --tasks T02,T04,T06,T08 --port-offset 600 &
wait
```
