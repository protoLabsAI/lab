# Quickstart

## Prerequisites

- Python 3.11+
- AI Gateway running on `:4000` (see [protoLabsAI/gateway](https://github.com/protoLabsAI/gateway))
- Langfuse running on `:3001` (included in gateway stack)
- Docker (for claw-eval sandbox tasks)

## Setup

```bash
cd ~/dev/evals
./scripts/setup.sh
source venv/bin/activate
```

## Run Your First Eval

### Custom task (fastest)

```bash
# Run the example calculator task against the local model
python -m runners.run_custom \
    --task tasks/tool_use/example_calculator.yaml \
    --model local

# Same task against a cloud model
python -m runners.run_custom \
    --task tasks/tool_use/example_calculator.yaml \
    --model claude-sonnet-4-6
```

### Claw-eval benchmark

```bash
# Run a single claw-eval task (3 trials)
python -m runners.run_claw --model local --tasks T01

# Run all tasks with 4 workers
python -m runners.run_claw --model local --all-tasks --workers 4
```

### Compare models

```bash
# Run same tasks on two models
python -m runners.run_claw --model local --tasks T01,T02,T03
python -m runners.run_claw --model claude-sonnet-4-6 --tasks T01,T02,T03

# Compare results
python -m runners.compare results/local_* results/claude-sonnet-4-6_*
```

## Submit Scores to Langfuse

```bash
# Add --submit-langfuse to attach scores to traces
python -m runners.run_custom \
    --suite tool_use \
    --model local \
    --submit-langfuse
```

Then view results at `http://localhost:3001` → Scores → filter by `eval/*`.

## Swap Models and Re-eval

```bash
# Swap vLLM model
cd ~/dev/experiments
./vllm-swap.sh opus-27b    # or qwen-27b, qwen-35b, qwen-122b

# Re-run evals against new model
cd ~/dev/evals
python -m runners.run_custom --suite tool_use --model local
```

## Next Steps

- [Writing custom tasks](./writing-tasks.md)
- [Langfuse integration](./langfuse-integration.md)
- [Model comparison workflow](../runbooks/model-comparison.md)
