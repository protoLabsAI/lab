# Langfuse Integration

Our Langfuse instance at `:3001` is the backbone of the eval system. All gateway
traffic is already traced automatically — evals build on top of that.

## How Tracing Works (Already Set Up)

```
Agent/User → Gateway (:4000) → LLM Provider
                │
                └─── auto-trace ──→ Langfuse (:3001)
                                      │
                                      ├── Trace (full request lifecycle)
                                      ├── Observations (individual LLM calls)
                                      └── Scores (eval grades, attached post-hoc)
```

Every request through the gateway creates a Langfuse trace with:
- Model name, provider, parameters
- Input/output messages
- Token counts and latency
- Cost (for cloud models)

## The Eval Flywheel

```
 ┌─────────────────────────────────────────────────────┐
 │                                                     │
 │  1. Production Traffic                              │
 │     └── Gateway auto-traces to Langfuse             │
 │                                                     │
 │  2. Online Evaluators (continuous)                  │
 │     └── LLM-as-judge scores sample of traffic       │
 │     └── Flags low-quality responses automatically   │
 │                                                     │
 │  3. Dataset Building                                │
 │     └── Failing traces → dataset items              │
 │     └── Add corrected expected outputs              │
 │                                                     │
 │  4. Experiments (on model/prompt change)             │
 │     └── Run dataset through new model               │
 │     └── Compare scores in Langfuse UI               │
 │                                                     │
 │  5. Deploy with confidence                          │
 │     └── Data-driven model selection                 │
 │     └── New edge cases feed back to step 3          │
 │                                                     │
 └─────────────────────────────────────────────────────┘
```

## Setting Up Online Evaluators

Online evaluators run automatically against production traffic in Langfuse.

### Via Langfuse UI (http://localhost:3001)

1. Go to **Evaluators** → **New Evaluator**
2. Pick type: **Observation-level** (recommended) or Trace-level
3. Choose a managed template or write custom:
   - **Hallucination** — detects fabricated information
   - **Helpfulness** — rates response quality
   - **Toxicity** — flags harmful content
   - **Context Relevance** — checks if context was used properly (RAG)
4. Set filters (trace name, tags, model) and sampling rate
5. Map variables: `{{input}}` → trace input, `{{output}}` → trace output

### Recommended Starter Evaluators

| Evaluator | Target | Sampling | Purpose |
|-----------|--------|----------|---------|
| Helpfulness | All traces | 10% | Baseline quality monitoring |
| Hallucination | Cloud model traces | 20% | Catch fabrications |
| Tool Use Quality | protoClaw traces | 100% | Verify tool selection |

## Programmatic Scoring

```python
from graders.langfuse_scorer import LangfuseScorer

scorer = LangfuseScorer(
    public_key="pk-...",
    secret_key="sk-...",
    host="http://localhost:3001",
)

# Score existing traces
traces = scorer.get_traces_for_scoring(tags=["eval"])
for trace in traces:
    scorer.client.score(
        trace_id=trace.id,
        name="eval/custom_metric",
        value=0.85,
        comment="Scored by custom pipeline",
    )
```

## Building Datasets from Production

```python
scorer = LangfuseScorer()

# Create a dataset
scorer.create_dataset(
    name="tool-use-failures-v1",
    description="Tool use traces that scored below 0.75",
)

# Add failing traces as dataset items
failing_traces = scorer.get_traces_for_scoring(tags=["low-quality"])
for trace in failing_traces:
    scorer.add_dataset_item(
        dataset_name="tool-use-failures-v1",
        input={"messages": trace.input},
        expected_output={"correct_tool": "calculator"},  # manually corrected
        source_trace_id=trace.id,
    )
```

## Running Experiments

```python
scorer = LangfuseScorer()

def my_agent(*, item, **kwargs):
    """Run the agent on a dataset item."""
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:4000/v1", api_key="key")
    response = client.chat.completions.create(
        model="local",
        messages=item.input["messages"],
    )
    return {"output": response.choices[0].message.content}

# Run experiment
scorer.run_experiment(
    dataset_name="tool-use-failures-v1",
    experiment_name="opus-27b-v1",
    task_fn=my_agent,
)
```

Then compare experiments side-by-side in Langfuse UI → Datasets → Select dataset → Experiments tab.

## Langfuse Score Naming Convention

All eval scores use the `eval/` prefix for easy filtering:

| Score Name | Type | Source |
|------------|------|--------|
| `eval/correctness` | NUMERIC | Outcome grader or LLM judge |
| `eval/tool_usage` | NUMERIC | Outcome grader or LLM judge |
| `eval/safety` | NUMERIC | Outcome grader |
| `eval/helpfulness` | NUMERIC | Online evaluator |
| `eval/hallucination` | NUMERIC | Online evaluator |
| `eval/passed` | NUMERIC | Aggregate (1.0 or 0.0) |
