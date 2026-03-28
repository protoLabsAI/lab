---
name: model-card
description: Generate a standardized HuggingFace model card for a quantized model
disable-model-invocation: true
allowed-tools: Read, Bash, Write
---

Generate a HuggingFace model card for a quantized model following the protoLabs standard.

## Arguments

$ARGUMENTS should be the path to the quantized model directory (e.g. `/mnt/models/quantized/Qwen3.5-9B-FP8`).

## Steps

1. Read `config.json` and `quantize_result.json` from the model directory to extract:
   - Base model name and architecture
   - Quantization method and config
   - Output size, quant time, VRAM usage

2. Check if there are speed test results in `experiments/quantize/results/` for this model.

3. Generate a `README.md` model card using this exact template:

```markdown
---
library_name: transformers
base_model:
  - {BASE_MODEL_ID}
base_model_relation: quantized
tags:
  - {QUANT_METHOD_TAG}
  - quantized
  - vllm
  - {MODEL_FAMILY_TAG}
license: apache-2.0
---

# {MODEL_NAME}

{QUANT_METHOD_DESCRIPTION} quantization of [{BASE_MODEL_ID}](https://huggingface.co/{BASE_MODEL_ID}) using [{TOOL_NAME}]({TOOL_URL}).

## Benchmarks

Measured on NVIDIA RTX PRO 6000 Blackwell (96GB), single GPU, vLLM 0.18:

| Metric | Original | Quantized | Change |
|--------|:--------:|:---------:|:------:|
| **Decode tok/s** | {ORIG_TOKS} | **{QUANT_TOKS}** | **{CHANGE}** |
| **TPOT** | {ORIG_TPOT} | **{QUANT_TPOT}** | **{TPOT_CHANGE}** |
| **TTFT** | {ORIG_TTFT} | {QUANT_TTFT} | {TTFT_CHANGE} |
| **Model size** | {ORIG_SIZE} | **{QUANT_SIZE}** | **{SIZE_CHANGE}** |

## Usage

```bash
vllm serve protoLabsAI/{MODEL_NAME} \
  --host 0.0.0.0 --port 8000 \
  --language-model-only \
  --gpu-memory-utilization 0.85
```

## Quantization Details

- **Method:** {METHOD_FULL_NAME}
- **Calibration:** {CALIBRATION_INFO}
- **Tool:** {TOOL_NAME_VERSION}
- **Quantization time:** {QUANT_TIME}
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (96GB)

## Notes

{NOTES}
```

4. Fill in all placeholders from the model metadata and benchmark results.
   - If no benchmark data exists, leave the benchmarks section with "TBD" and note that speed tests should be run.
   - For FP8 models, note that `--language-model-only` is needed for Qwen3.5.
   - Always include the `base_model_relation: quantized` field — this is required to appear in HuggingFace's quantized model listings.

5. Write the README.md to the model directory.

6. Ask if the user wants to upload it:
   ```bash
   huggingface-cli upload protoLabsAI/{MODEL_NAME} {MODEL_DIR}/README.md README.md
   ```

## HuggingFace Org

All quantized models go to the `protoLabsAI` organization on HuggingFace.
