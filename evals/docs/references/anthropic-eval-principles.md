# Reference: Anthropic's Agent Eval Principles

Source: [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

## Core Mental Models

### Capability vs Regression Evals

| | Capability | Regression |
|--|-----------|------------|
| **Question** | What can this agent do? | Does it still work? |
| **Starting pass rate** | Low (a hill to climb) | ~100% (protect the baseline) |
| **Lifecycle** | Graduate to regression when saturated | Grow as capabilities stabilize |

### Swiss Cheese Model

No single eval layer catches everything. Stack multiple methods:
- Automated evals (this repo)
- Production monitoring (Langfuse online evaluators)
- Human review (Langfuse annotation queues)
- A/B testing (Langfuse experiments)

### pass@k vs pass^k

| Metric | Formula | Use When |
|--------|---------|----------|
| pass@k | P(at least 1 success in k trials) | Tools (one success is enough) |
| pass^k | P(all k trials succeed) | Agents (consistency is essential) |

We **previously** used pass^3 as default, but dropped it 2026-06-29: at our breadth (30 claw tasks +
~16 custom suites + FC) the 3× repetition cost far outweighed the marginal signal — running *more
distinct tasks* separates models better than running the *same* task three times. Default is now
single-trial; reach for pass^k (`--trials N`) only when run-to-run consistency on a specific small
task set is the actual question.

## The 8-Step Roadmap

0. **Start early** — 20-50 tasks from real failures is enough to begin
1. **Start with manual QA** — convert what you already test by hand
2. **Write unambiguous tasks** — two experts must agree on pass/fail
3. **Build balanced sets** — test both positive and negative cases
4. **Stable environments** — each trial starts clean, no shared state
5. **Grade outcomes, not paths** — check final state, allow creative solutions
6. **Read the transcripts** — verify your eval measures what matters
7. **Watch for saturation** — when pass rates hit 100%, build harder evals
8. **Open contribution** — let domain experts contribute tasks

## Anti-Patterns to Avoid

- **Path-checking**: Asserting specific tool call sequences is brittle
- **Grader bugs**: Anthropic's Opus 4.5 went from 42% → 95% on CORE-Bench after fixing grading bugs
- **Goal misalignment**: Grading by "exceeding threshold" when instructions say "optimize to threshold"
- **Shared state**: Claude once gained advantage by reading git history from prior trials
- **Face-value scores**: Always dig into the eval details and read transcripts

## Grader Types

| Type | Speed | Cost | Nuance | Reproducible |
|------|-------|------|--------|-------------|
| Code-based | Fast | Free | Low | Yes |
| Model-based | Medium | $$  | High | No |
| Human | Slow | $$$$ | Highest | Somewhat |

Use all three. Code-based for verifiable facts, model-based for quality,
human for calibration and edge cases.
