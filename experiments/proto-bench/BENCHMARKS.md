# Coding Benchmark Landscape — protoLabs Reference

Research digest for selecting and running coding benchmarks against protoCLI and local models.

## The Reality Check

There's a ~53-point collapse from headline benchmark scores to production performance. Models scoring 80% on SWE-bench Verified drop to ~17% on genuinely private/unseen codebases. SWE-bench Verified was officially retired by OpenAI in February 2026 as contaminated.

**Implication for us:** Public benchmark scores are marketing. Internal eval on our own codebases is the real signal.

---

## Tier 1 — High Trust (what we should run)

### SWE-bench Pro (Scale AI / SEAL)

**The current community consensus for agent eval.**

| Property | Value |
|----------|-------|
| Tasks | 1,865 |
| Repos | 41 |
| Languages | Python, Go, JS, TS |
| Eval method | Pass/fail via existing test suites |
| Contamination defense | Private commercial split (GPL + proprietary) |
| Key caveat | Scaffolding variance is huge — same model scores 45.9% (SEAL standardized) vs 57.0% (best agent system) |

**How to run:**
- SEAL leaderboard standardized scaffolding available
- Docker-per-task isolation
- Must report which scaffolding produced the score

**Relevance to us:** Direct eval for protoCLI as a coding agent. Run with standardized scaffolding for apples-to-apples comparison.

### LiveCodeBench v6

**Best for contamination-free raw model eval.**

| Property | Value |
|----------|-------|
| Source | LeetCode, CodeForces, AtCoder (post-publication) |
| Eval method | Temporal gating — only problems after model training cutoff |
| Frontier scores | 55–75% (vs 90%+ on dead HumanEval) |
| Weakness | Competitive programming only — zero signal on multi-file SE work |

**Relevance to us:** Good for comparing raw model capability (Gemma 4 vs Qwen 3.5) without agent scaffolding noise.

### Aider Polyglot

**Fast, reproducible base model selection.**

| Property | Value |
|----------|-------|
| Tasks | 225 hard Exercism problems |
| Languages | 6 |
| Design | Two-attempt: attempt → unit test feedback → retry |
| Eval method | Unit test pass/fail |

**Relevance to us:** Quick smoke test for new models. 225 problems runs fast on local vLLM.

### Terminal-Bench 2.0 (Harbor)

**Only serious harness for shell/DevOps agents.**

| Property | Value |
|----------|-------|
| Tasks | 89 manually verified |
| Environment | Real container environments (compiling, training, server config) |
| Supports | RL rollouts |
| Top scores | ~50% (still differentiating) |
| Weakness | Small task count |

**Relevance to us:** Direct target for protoCLI APO optimization. We want to get on this leaderboard.

---

## Tier 2 — Useful with Caveats

| Benchmark | Tasks | Notes |
|-----------|:-----:|-------|
| **SWE-bench Live** (Microsoft) | 1,319 | Rolling monthly updates, Docker-based, newer/less tested |
| **SWE-rebench** | 21k+ | Automated, temporal contamination flags. Best for RL training data |
| **BigCodeBench** | 1,140 | Function-level, 139 libraries, 99% branch coverage |
| **SWE-Lancer** | $1M Upwork tasks | Economically grounded but JS/TS only, private eval |
| **BFCL v4** | 4,441 | Function calling (we already run this — Gemma 4 MoE at 94.4%) |

---

## What We Run Today

| Benchmark | Status | Models Tested |
|-----------|--------|---------------|
| **BFCL v4** (4,706 tests) | ✅ Running | Full harness integrated — see detailed results below |
| **Function Calling** (8 tests) | ✅ Running | Gemma 4 MoE (8/8), E4B (8/8), E2B (8/8) — all 100% |
| **Aider Python Hard** (34 exercises) | ✅ Running | Qwen 27B (32%), Qwen 35B MoE (29%), Gemma 4 31B (18%), Gemma 4 MoE (15%) |
| **Claw-Eval** (104 tasks) | ✅ Running | Qwen 122B (86%), Qwen 27B (83%), Qwen 35B MoE (45%), Gemma 4 MoE (32%), Gemma 4 31B (30%) |
| **Custom suites** (59 tasks) | ✅ Running | Gemma 4 MoE + thinking: 51/59 (86%) |
| **WildBench** (1,024 tasks) | ✅ Available | GPT-5.4, Sonnet, Haiku, Qwen 9B variants |
| **Proto-bench coding** (5 tasks) | ✅ Prototype | Qwen 122B (0.800 baseline) |
| **Terminal-Bench 2.0** | ❌ Not yet | Target for protoCLI |
| **SWE-bench Pro** | ❌ Not yet | Need Docker setup |
| **LiveCodeBench** | ❌ Not yet | Need temporal gating |

### Aider Polyglot Results (Gemma 4 26B MoE FP8, 2026-04-04)

176/225 exercises completed (C++/JS errored due to test runner config).

| Language | Total | Pass1 | Pass2 | Rate |
|----------|:-----:|:-----:|:-----:|:----:|
| Rust | 25 | 8 | 12 | **48%** |
| Python | 25 | 2 | 11 | **44%** |
| Go | 32 | 5 | 11 | 34% |
| Java | 37 | 4 | 10 | 27% |
| C++ | 21 | — | — | errored (cmake) |
| JavaScript | 36 | — | — | errored (jest) |
| **Valid total** | **119** | **19** | **44** | **37.0%** |

Leaderboard context (pass_rate_2):
- Claude Sonnet 4.5: ~55-60%
- GPT-5.4: ~50-55%
- Qwen 2.5 Coder 32B: ~40-45%
- **Gemma 4 MoE (local, 175 tok/s): 37%**

Note: Gemma 4 is a general-purpose model, not coding-specialized. Strong self-correction (11.8% → 37% with retry).

### Claw-Eval Full Results (2026-04-05/07)

| Model | Size | tok/s | Pass | Rate | Avg Score | Notes |
|-------|------|:-----:|:----:|:----:|:---------:|-------|
| **Qwen 122B INT4** | 74GB | 122 | 89/103 | **86.4%** | 0.860 | Quality ceiling, TP=2 |
| **Qwen 27B INT4** | 29GB | 53 | 86/103 | **83.5%** | 0.830 | Daily driver, best quality/speed |
| **Qwen 35B MoE FP8** | 35GB | 180 | 41/92 | **44.6%** | 0.604 | Speed king, 3B active params |
| Gemma 4 26B MoE | 35GB | ~175 | 30/94 | 31.9% | 0.584 | 10B active params |
| Gemma 4 31B dense | 59GB | ~45 | 26/87 | 29.9% | 0.566 | bf16, single GPU |
| Gemma 4 E2B | 10GB | 220 | 16/98 | 16.3% | 0.501 | Ollama bf16 |
| Gemma 4 E4B | 16GB | 126 | 14/99 | 14.1% | 0.472 | Ollama bf16 |

Category breakdown:

| Category | Q122B | Q27B | Q35B MoE | G4 MoE | G4 31B | G4 E2B | G4 E4B |
|----------|:-----:|:----:|:--------:|:------:|:------:|:------:|:------:|
| **PinBench** | — | — | 83% | 32% | 67% | 42% | 33% |
| **Research** | — | — | 71% | 50% | 57% | 12% | 0% |
| **Security** | — | — | 67% | 33% | 100% | 100% | 67% |
| **Agentic (Core)** | — | — | 44% | 32% | 26% | 16% | 19% |
| **Finance/Office** | — | — | 29% | 18% | 0% | 0% | 0% |
| **Coding (T100+)** | — | — | 20% | 17% | 17% | 0% | 0% |
| **Vision/Multimodal** | — | — | 0% | 0% | 0% | 0% | 0% |

### Aider Python Hard (34 Exercism exercises, whole edit format)

| Model | Pass2 | Rate | Notes |
|-------|:-----:|:----:|-------|
| **Qwen 27B INT4** | 11/34 | **32.4%** | Best coding accuracy |
| **Qwen 35B MoE FP8** | 10/34 | **29.4%** | Nearly matches 27B, 3.4x faster |
| Gemma 4 31B dense | 6/34 | 17.6% | +thinking, whole format |
| Gemma 4 26B MoE | 5/34 | 14.7% | +thinking, whole format |
| Gemma 4 26B MoE (diff) | 3/34 | 8.8% | No thinking, diff format (worst) |

### Custom Eval Suite (59 tasks, LLM-judged, 2026-04-06)

Gemma 4 MoE with thinking enabled:

| Suite | Pass | Rate |
|-------|:----:|:----:|
| Coding | 9/10 | 90% |
| Reasoning | 4/5 | 80% |
| Structured Output | 5/5 | 100% |
| Instruction Following | 3/5 | 60% |
| Summarization | 22/25 | 88% |
| Safety | 5/5 | 100% |
| Research | 3/4 | 75% |
| **Total** | **51/59** | **86.4%** |

### BFCL v4 Full Results (2026-04-07)

Gorilla Berkeley Function Calling Leaderboard — full 4,706 test harness.
Harness at `~/dev/gorilla-bfcl/`, wrapper at `evals/run-bfcl.sh`.

**Single-turn (non-live, curated):**

| Category | Gemma 4 MoE | Qwen 27B INT4 |
|----------|:-----------:|:-------------:|
| simple_python (400) | **96.75%** | 56.50% |
| parallel (200) | **95.50%** | 54.00% |
| parallel_multiple (200) | **91.50%** | 20.00% |
| irrelevance (240) | **92.92%** | 78.75% |
| simple_javascript (50) | 76.00% | 70.00% |
| simple_java (100) | **66.00%** | 3.00% |

**Live (real-world user-contributed):**

| Category | Gemma 4 MoE | Qwen 27B INT4 |
|----------|:-----------:|:-------------:|
| live_simple (258) | **87.98%** | 62.40% |
| live_parallel (16) | 87.50% | 87.50% |
| live_multiple (1053) | 66.67% | **74.26%** |
| live_parallel_multiple (24) | 58.33% | 58.33% |
| live_irrelevance (884) | — | 71.49% |
| live_relevance (16) | — | **93.75%** |

**Multi-turn (conversational tool use):**

| Category | Gemma 4 MoE | Qwen 27B INT4 |
|----------|:-----------:|:-------------:|
| multi_turn_base (200) | 6.00% | **74.50%** |
| multi_turn_miss_param (200) | 2.00% | **16.00%** |
| multi_turn_miss_func (200) | 5.00% | 0.00% |
| multi_turn_long_context (200) | 4.50% | **15.50%** |

**Root cause of claw-eval gap identified:** Gemma 4 has best-in-class single-turn function calling (92-97%) but collapses on multi-turn (2-6%). Qwen 27B is mediocre at single-turn (20-57%) but strong at multi-turn base (74.5%). Claw-eval tasks are multi-turn, explaining why Qwen scores 83% vs Gemma's 32%.

Note: Qwen 27B single-turn scores may be depressed by QwenFCHandler format mismatch with Qwen 3.5 (handler built for Qwen 3). Qwen's native tool calling via vLLM scores ~89% on our 8-test FC suite.

### Key Findings

- **Qwen 27B INT4 is the best all-rounder**: 83% claw-eval, 32% Aider, 53 tok/s — best quality-per-speed ratio
- **Qwen 35B MoE FP8 is the speed king**: 180 tok/s, 45% claw-eval — trades quality for 3.4x throughput
- **Qwen 122B INT4 is the quality ceiling**: 86% claw-eval, needs TP=2 (both GPUs)
- **Gemma 4 MoE vs 31B dense are nearly identical**: ~30% claw-eval, ~16% Aider — MoE is 4x faster
- **Gemma 4 excels at single-turn function calling** (97% BFCL simple_python) but collapses on multi-turn (2-6%). This is the root cause of its poor claw-eval showing — not format issues
- **Qwen 27B is the opposite**: mediocre single-turn FC (57% simple_python via BFCL handler) but strong multi-turn (74.5% base). Multi-turn ability drives agentic benchmark success
- **Thinking mode helps Gemma 4**: +3-6% on Aider, enables reasoning extraction, but doesn't move claw-eval scores
- **Edit format matters for Aider**: whole >> diff for all models tested
- **Vision tasks are 0%** across the board (no multimodal serving configured)
- Qwen2.5-Coder is obsoleted by Qwen3.5 — general models beat the code-specialized variant
- **BFCL v4 full harness integrated** (`evals/run-bfcl.sh`) — 4,706 tests across single-turn, live, multi-turn, memory, and web search categories

---

## Implementation Priorities

### P0: Expand proto-bench (this week)

Our `experiments/proto-bench/` harness needs real tasks. Current 5 coding tasks are too thin.

**Action items:**
1. Pull Aider Polyglot test set (225 problems, fast to run)
2. Set up Docker-per-task isolation for SWE-bench format
3. Add temporal gating (only use problems after model training cutoff)
4. Multi-trial runs (3 minimum) with variance reporting

### P1: Terminal-Bench 2.0 (next sprint)

Get protoCLI on the Terminal-Bench leaderboard.

**Action items:**
1. Install Harbor CLI
2. Run protoCLI + Gemma 4 MoE through `harbor run -d terminal-bench@2.0`
3. APO-optimize protoCLI's system prompt on terminal-bench tasks
4. Submit to leaderboard

### P2: SWE-bench Pro (following sprint)

Full agent eval on real-world software engineering tasks.

**Action items:**
1. Set up Docker image pipeline (base → env → instance)
2. Adapt `proto_agent.py` for SWE-bench Pro format
3. Run protoCLI + Gemma 4 MoE vs cloud models
4. Report scaffolding configuration alongside scores

### P3: Internal Private Benchmark

The gold standard — eval on our own codebases.

**Action items:**
1. Curate 50-200 historical tickets from protoMaker/protoResearcher/lab repos
2. Use existing test suites as pass/fail oracle
3. Only issues from past 90 days (temporal gating)
4. 3-5 trial average with variance tracking
5. Track cost-per-resolved-task alongside resolution rate

---

## Harness Design Principles

Based on community research:

### Do

- **Docker-per-task** — isolated containers, no shared state between tasks
- **Temporal gating** — only problems after model training cutoff
- **Multi-trial runs** — 3-5 trials, report variance
- **Full test suite** — run ALL existing tests after patches, not just new ones
- **Trajectory capture** — log all tool calls, not just final patch
- **Report scaffolding** — always specify agent framework alongside model scores
- **Cost tracking** — cost-per-resolved-task as primary metric

### Don't

- Public repos without temporal gating
- LLM-as-judge as primary signal (use test suites)
- Let agent see test cases during execution
- Single-trial scoring with no variance report
- Run only new tests after patches
- Share Docker environments across tasks
- Mix "model score" with "model + scaffolding score"
- Use benchmark problems that appear in RL training data

---

## Isolation Options

| Method | Boot Time | Overhead | Security | Best For |
|--------|:---------:|:--------:|:--------:|----------|
| **Docker** | ~1s | Moderate | Container escapes possible | Standard eval |
| **gVisor** | ~1s | 10-20% | Syscall filtering | Untrusted agents |
| **Firecracker** | ~125ms | <5 MiB | Full VM isolation | Production eval, RL rollouts |

Recommendation: Docker for development, Firecracker for production eval with untrusted agent code.

---

## Parallelism

```
workers = min(0.75 * cpu_count, 24)
max_concurrent_containers = 32  # SWE-bench recommendation

# Read-only tools: thread pools (up to 5 concurrent)
# Write tools: sequential within a task
# Task design: stateless, no shared mutable state
```

---

## References

- [SWE-bench Pro / SEAL Leaderboard](https://scale.com/leaderboard/swe-bench-pro)
- [LiveCodeBench](https://livecodebench.github.io/)
- [Aider Polyglot](https://aider.chat/docs/leaderboards/)
- [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0)
- [BFCL v4](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [BigCodeBench](https://bigcode-bench.github.io/)
- [SWE-rebench](https://arxiv.org/abs/2501.05728)
- [SWE-Lancer](https://swe-lancer.com/)
