# Agent Eval Framework Landscape (2026)

## Benchmark Tiers

### Tier 1 — Industry Standard
- **SWE-bench Verified** — coding (effectively retired, 59.4% tasks flawed)
- **Terminal-Bench 2.0** — DevOps/sysadmin, 89 tasks, "used by every frontier lab"
- **OSWorld-Verified** — full desktop computer use, gold standard for GUI agents

### Tier 2 — Domain Standards
- **tau2-Bench** — conversational customer service agents
- **BrowseComp** — deep web research (1,266 tasks)
- **Multi-SWE-bench** — multilingual coding (2,132 tasks, 7 languages)

### Tier 3 — Niche/Emerging
- **GAIA** — general assistant (Level 1 near-saturated)
- **Claw-Eval** — personal assistant/productivity (what we use)
- **CORE-Bench** — scientific reproducibility

## Eval Frameworks

| Framework | OSS | Stars | Best For |
|-----------|-----|:-----:|---------|
| **Inspect (UK AISI)** | MIT | 1.8k | Safety evals, benchmark collection |
| **Harbor** | Yes | 1k | Running benchmarks at scale |
| **Langfuse** | MIT | 23k | Observability + eval flywheel |
| **LangSmith** | No | — | LangChain ecosystem |
| **Braintrust** | No | — | CI/CD quality gates |
| **Arize Phoenix** | Yes | 9k | Production monitoring |

## Gaps in Claw-Eval

| Gap | Better Benchmark |
|-----|-----------------|
| Real software engineering | SWE-bench Pro/Live |
| GUI/computer use | OSWorld |
| Deep web browsing | BrowseComp |
| DevOps at scale | Terminal-Bench 2.0 |
| Multi-turn conversation | tau2-Bench |
| Non-Python coding | Multi-SWE-bench |

## Our Approach

We use Claw-Eval for agent tool-use + custom suites for creative writing, roleplay, research, coding, SVG, function-call, and RAG evaluation. Inspect AI is scaffolded for broader benchmarks (HumanEval, MMLU, GAIA). All scores flow to Langfuse for the eval flywheel.
