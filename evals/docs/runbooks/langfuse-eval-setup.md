# Runbook: Langfuse Eval Pipeline Setup

One-time setup for online evaluators and the eval flywheel.

## Prerequisites

- Langfuse running at `http://localhost:3001` (part of gateway stack)
- Gateway tracing active (check: Langfuse → Traces should show recent requests)
- A Langfuse project with API keys

## Step 1: Get Langfuse API Keys

1. Open `http://localhost:3001`
2. Go to **Settings** → **API Keys**
3. Create a new key pair
4. Save to Infisical under `secret-management` project:
   - `LANGFUSE_PUBLIC_KEY` = `pk-lf-...`
   - `LANGFUSE_SECRET_KEY` = `sk-lf-...`

## Step 2: Set Up Online Evaluators

In Langfuse UI → **Evaluators** → **New Evaluator**:

### Helpfulness (start here)

- **Type:** Observation-level
- **Template:** Helpfulness (managed)
- **Filter:** All traces
- **Sampling:** 10%
- **Variable mapping:** `{{input}}` → input, `{{output}}` → output

### Hallucination

- **Type:** Observation-level
- **Template:** Hallucination (managed)
- **Filter:** Cloud model traces only
- **Sampling:** 20%
- **Variable mapping:** `{{input}}` → input, `{{output}}` → output

### Custom: Tool Use Quality

- **Type:** Observation-level
- **Template:** Custom
- **Prompt:**
  ```
  You are evaluating whether an AI agent selected and used the correct tools.

  User request: {{input}}
  Agent response: {{output}}

  Score the tool usage:
  - 1.0: Correct tool(s) selected and used properly
  - 0.75: Correct tool but minor usage issues
  - 0.5: Partially correct tool selection
  - 0.25: Wrong tool selected
  - 0.0: No tools used when they should have been

  Respond with just the numeric score.
  ```
- **Filter:** Tag = `protoclaw`
- **Sampling:** 100%

## Step 3: Create Initial Dataset

```python
from graders.langfuse_scorer import LangfuseScorer

scorer = LangfuseScorer()

# Create datasets for each eval category
for name, desc in [
    ("regression-v1", "Core regression tests — must always pass"),
    ("tool-use-v1", "Tool selection and invocation tests"),
    ("capability-v1", "Aspirational capability tests"),
]:
    scorer.create_dataset(name=name, description=desc)
```

## Step 4: Verify Pipeline

```bash
# Send a test request through the gateway
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}'

# Check Langfuse:
# 1. Trace appears in Traces tab
# 2. Online evaluator scores appear within ~30s
# 3. Score visible on trace detail page
```

## Step 5: Add to Infisical

Ensure these env vars are in the `secret-management` Infisical project:

| Key | Purpose |
|-----|---------|
| `LANGFUSE_PUBLIC_KEY` | Eval SDK authentication |
| `LANGFUSE_SECRET_KEY` | Eval SDK authentication |
| `LANGFUSE_HOST` | `http://localhost:3001` |

## Maintenance

- **Weekly:** Review low-scoring traces, add interesting failures to datasets
- **Monthly:** Check evaluator calibration (sample 10 traces, compare human vs LLM judge)
- **Quarterly:** Review and update evaluator prompts, add new dimensions
- **On model swap:** Run full eval suite, compare in experiments UI
