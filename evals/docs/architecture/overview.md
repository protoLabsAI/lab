# Eval Architecture Overview

## System Diagram

```
                  ┌──────────────────────────────────────────────┐
                  │          protoLabs Eval Laboratory            │
                  │                                              │
                  │  ┌────────────────────────────────────────┐  │
                  │  │           Eval Suites                  │  │
                  │  │                                        │  │
                  │  │  claw-eval    104 agent tasks          │  │
                  │  │  inspect-ai   GAIA, SWE-bench, MMLU…  │  │
                  │  │  func-call    tool call accuracy       │  │
                  │  │  rag          retrieval quality         │  │
                  │  │  t-bench      DevOps/sysadmin          │  │
                  │  │  tau2         conversational agents     │  │
                  │  │  custom       YAML task definitions     │  │
                  │  └──────────────────┬─────────────────────┘  │
                  │                     │                        │
                  │  ┌──────────────────┴─────────────────────┐  │
                  │  │           Graders                      │  │
                  │  │  outcome │ llm_judge │ function_call   │  │
                  │  │  rag (4 dimensions) │ langfuse_scorer  │  │
                  │  └──────────────────┬─────────────────────┘  │
                  │                     │                        │
                  │  ┌──────────────────┴─────────────────────┐  │
                  │  │           Runners + CLI                │  │
                  │  │  run.sh → Infisical → proto-eval CLI   │  │
                  │  └──────────────────┬─────────────────────┘  │
                  └─────────────────────┼────────────────────────┘
                                        │
                  ┌─────────────────────┼────────────────────────┐
                  │                     v                        │
                  │  ┌──────────────────────────────────────┐    │
                  │  │     AI Gateway (:4000)               │    │
                  │  │     LiteLLM Proxy — 20+ models       │    │
                  │  └──────────────────┬───────────────────┘    │
                  │       ┌─────────────┼─────────────┐         │
                  │       v             v             v         │
                  │    vLLM          Claude        Gemini       │
                  │    Qwen 27B      GPT-5.4      DeepSeek      │
                  │    OmniCoder     Haiku        GLM/Kimi      │
                  │                                              │
                  │  ┌──────────────────────────────────────┐    │
                  │  │     Langfuse (:3001)                 │    │
                  │  │  traces + scores + datasets           │    │
                  │  │  experiments + comparison UI          │    │
                  │  └──────────────────────────────────────┘    │
                  │              Infrastructure                  │
                  └─────────────────────────────────────────────┘
```

## Eval Suites

### Tier 1 — Ready Now

| Suite | Source | What It Tests | Tasks |
|-------|--------|---------------|-------|
| **Claw-Eval** | Git submodule | Agent tool use across mock services | 104 |
| **Function Call** | Custom (BFCL-inspired) | Structured tool call accuracy | Growing |
| **RAG** | Custom | Groundedness, faithfulness, relevance | Growing |
| **Custom** | YAML definitions | Any task with LLM judge grading | Any |

### Tier 2 — Needs Install

| Suite | Source | What It Tests | Tasks |
|-------|--------|---------------|-------|
| **Inspect AI** | pip install | GAIA, SWE-bench, HumanEval, MMLU, BrowseComp | 1000s |
| **General** | Inspect wrapper | Standard LLM quality battery | ~500 |

### Tier 3 — Scaffolded

| Suite | Source | What It Tests | Tasks |
|-------|--------|---------------|-------|
| **Terminal-Bench** | Harbor framework | DevOps/sysadmin | 89 |
| **tau2-Bench** | pip install | Conversational customer service | ~200 |

## Data Flow

1. **Task Definition** → YAML files, submodule configs, or Inspect benchmarks
2. **Secret Injection** → `run.sh` authenticates with Infisical, injects API keys
3. **Execution** → Runners send tasks to models via the gateway
4. **Tracing** → Gateway auto-traces all LLM calls to Langfuse
5. **Grading** → Suite-specific graders score outputs
6. **Scoring** → Grades submitted to Langfuse as scores attached to traces
7. **Comparison** → CLI compare tool or Langfuse experiment UI

## Key Design Decisions

### Grade outcomes, not paths
Agents find valid approaches that eval designers didn't anticipate.
Checking tool call sequences is brittle. Check the final state instead.

### Pass^k for consistency
pass^3 (default) means the agent must succeed in all 3 independent trials.
This separates reliable capability from lucky runs.

### Multi-dimensional scoring with partial credit
Each task scored across multiple dimensions. An agent that gets the right answer
but uses the wrong tool is meaningfully different from one that fails entirely.

### Gateway as evaluation substrate
All models accessed through the same gateway endpoint. Identical request format,
automatic tracing, fair comparison. Swap `--model local` for `--model gpt-5.4`.

### Parallel cloud evaluation
Port offsets let mock services coexist so multiple models can be tested simultaneously.
Cloud model evals that took 30 min sequential now take 5 min parallel.
