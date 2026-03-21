# Runbook: Model Comparison

How to evaluate a new model before deploying it to production.

## When to Use

- New model downloaded (e.g., a new fine-tune or quantization)
- vLLM model swap (`vllm-swap.sh`)
- Deciding between local vs cloud model for a use case
- Quarterly model refresh

## Procedure

### 1. Establish baseline

Run the eval suite against the current production model:

```bash
cd ~/dev/evals
source venv/bin/activate

# Record current model
curl -s http://localhost:4000/v1/models | jq '.data[0].id'

# Run eval suite
python -m runners.run_custom --suite tool_use --model local --submit-langfuse
python -m runners.run_claw --model local --tasks T01,T02,T03,T04,T05
```

### 2. Swap model

```bash
cd ~/dev/experiments
./vllm-swap.sh opus-27b   # or whatever model to test
```

### 3. Run eval suite against new model

```bash
cd ~/dev/evals

# Same tasks, same config — only the model changed
python -m runners.run_custom --suite tool_use --model local --submit-langfuse
python -m runners.run_claw --model local --tasks T01,T02,T03,T04,T05
```

### 4. Compare

```bash
# CLI comparison
python -m runners.compare results/local_2026*

# Or use Langfuse UI:
# http://localhost:3001 → Scores → filter by eval/* → compare timestamps
```

### 5. Decision matrix

| Metric | Weight | Notes |
|--------|--------|-------|
| Pass^3 rate | 40% | Consistency is king for agents |
| Average score | 20% | Overall quality |
| Tokens used | 15% | Efficiency (cost for cloud, speed for local) |
| Latency | 15% | User experience |
| Safety score | 10% | Non-negotiable minimum threshold |

### 6. Deploy or rollback

If the new model improves on the baseline:
```bash
# Update CLAUDE.md with new model info
# Update vllm.service if making permanent
sudo systemctl edit vllm  # override ExecStart
```

If not, swap back:
```bash
./vllm-swap.sh qwen-27b  # or whatever was running before
```

## Cloud Model Comparison

For comparing cloud models (no vLLM swap needed):

```bash
# Run same tasks against different cloud models
python -m runners.run_custom --suite tool_use --model claude-sonnet-4-6 --submit-langfuse
python -m runners.run_custom --suite tool_use --model gpt-4o --submit-langfuse
python -m runners.run_custom --suite tool_use --model gemini-2.5-flash --submit-langfuse
```
