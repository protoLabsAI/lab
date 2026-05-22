# Refusal Evaluation Suite

Tests model safety calibration: **over-refusal** (refusing safe prompts) and **under-refusal** (complying with harmful prompts).

## Quick Start

```bash
cd evals

# Quick smoke test (100 prompts, SimpleSafetyTests)
./run.sh refusal --model local --dataset simple_safety

# XSTest over-refusal benchmark (450 prompts)
./run.sh refusal --model local --dataset xstest

# All datasets, 50 samples each (fast sweep)
./run.sh refusal --model local --dataset all --sample 50

# Full run on heretic (with logit_bias for think suppression)
./run.sh refusal --model local-fast --dataset all \
    --gateway-url http://localhost:8002/v1 \
    --extra-body '{"logit_bias":{"248068":-100,"248069":-100}}'

# Use specific judge model
./run.sh refusal --model local --dataset xstest \
    --judge-url http://localhost:8000/v1 --judge-model local

# Use gateway judge (needs GATEWAY_API_KEY)
./run.sh refusal --model local --dataset xstest \
    --judge-url http://ava:4000/v1 --judge-model claude-sonnet-4-6
```

## Datasets

| Dataset | Size | Tests | Source |
|---------|------|-------|--------|
| **xstest** | 450 | Over-refusal (250 safe + 200 unsafe pairs) | [walledai/xstest](https://huggingface.co/datasets/natolambert/xstest-v2-copy) |
| **orbench_hard** | 1,319 | Over-refusal (safe-but-sensitive prompts) | [BENCH-LLM/OR-BENCH](https://huggingface.co/datasets/BENCH-LLM/OR-BENCH) |
| **orbench_toxic** | 655 | Should-refuse baseline (genuinely toxic) | [BENCH-LLM/OR-BENCH](https://huggingface.co/datasets/BENCH-LLM/OR-BENCH) |
| **do_not_answer** | 939 | Under-refusal (12 risk areas) | [LibrAI/do-not-answer](https://huggingface.co/datasets/LibrAI/do-not-answer) |
| **simple_safety** | 100 | Quick safety smoke test | [Bertievidgen/SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) |
| **sorry_bench** | 9,450 | Under-refusal (21 attack styles) | [sorry-bench/sorry-bench-202406](https://huggingface.co/datasets/sorry-bench/sorry-bench-202406) |
| **harmbench** | 400 | Jailbreak resistance (3 configs) | [walledai/HarmBench](https://huggingface.co/datasets/walledai/HarmBench) |
| **wildguard_test** | 1,725 | Both (gold labels for harmful + safe) | [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix) |
| **Total** | **15,038** | | |

### Dataset Details

**Over-refusal (model refuses when it shouldn't):**
- `xstest` — The gold standard. 10 prompt types: homonyms, figurative language, safe targets, safe contexts, etc. Each has a `contrast_` unsafe counterpart. CC-BY-4.0.
- `orbench_hard` — 1,319 hardest false-refusal triggers from OR-Bench 80K. Prompts that *appear* toxic but are actually safe. CC-BY-4.0.
- `wildguard_test` — 1,725 prompts with gold `prompt_harm_label` (unharmful/harmful). Tests both directions.

**Under-refusal (model complies when it shouldn't):**
- `simple_safety` — 100 clearly harmful prompts. Quick baseline. CC-BY-4.0.
- `do_not_answer` — 939 prompts across 12 risk areas (violence, discrimination, etc.). CC-BY-NC-SA-4.0.
- `sorry_bench` — 9,450 prompts × 21 attack styles (base, slang, role-play, morse code, caesar cipher, etc.). Tests robustness to jailbreak techniques. CC-BY-4.0.
- `harmbench` — 400 harmful behaviors in 3 configs: standard (200), contextual (100), copyright (100). MIT.
- `orbench_toxic` — 655 genuinely toxic prompts (control group for orbench_hard). CC-BY-4.0.

## Metrics

The runner reports:
- **Over-refusal rate**: % of safe prompts the model incorrectly refuses
- **Under-refusal rate**: % of harmful prompts the model incorrectly complies with
- **Per-category breakdown** with error rates and visual bars
- **Combined summary** when running multiple datasets

Each response is classified by an LLM judge as:
- `comply` — model provided the requested information
- `refuse` — model declined or redirected
- `partial` — model partially complied with significant caveats

## Testing a New Model

```bash
# 1. Start the model on vLLM
bash models/vllm-swap.sh <config-name>

# 2. Quick sweep (5 min) — sample 50 from each dataset
./run.sh refusal --model <model-id> --dataset all --sample 50

# 3. Full XSTest (10 min) — the key over-refusal benchmark
./run.sh refusal --model <model-id> --dataset xstest

# 4. Full safety check (30 min) — all under-refusal datasets
./run.sh refusal --model <model-id> --dataset simple_safety,do_not_answer,harmbench

# 5. Jailbreak resistance (60+ min) — sorry-bench attack styles
./run.sh refusal --model <model-id> --dataset sorry_bench --sample 450

# Results saved to evals/results/refusal_<model>_<timestamp>/
```

## Output Format

Each run produces in `evals/results/refusal_<model>_<timestamp>/`:
- `<dataset>_results.json` — per-prompt results (prompt, response, classification, expected)
- `<dataset>_metrics.json` — computed metrics (over/under-refusal rates, per-category)
- `summary.json` — combined summary across all datasets in the run

## Interpreting Results

| Model Type | Expected Over-Refusal | Expected Under-Refusal |
|------------|:--------------------:|:---------------------:|
| Safety-tuned (Qwen, Llama) | 5-15% | 0-5% |
| Uncensored (heretic) | ~0% | 40-80% |
| Base model (no RLHF) | ~0% | 80-100% |

A well-calibrated model should have **low over-refusal** (doesn't refuse safe requests) AND **low under-refusal** (does refuse harmful ones). The tension between these is the core trade-off.

## Judge Configuration

Default judge: `local-fast` (heretic on `:8002`). Fast, no-thinking, good for binary classification.

For higher-quality judging, use an external model:
```bash
--judge-url http://ava:4000/v1 --judge-model claude-sonnet-4-6
```

When using heretic as judge, think tokens are automatically suppressed via `logit_bias`.
