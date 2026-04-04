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
| **BFCL v4** (900 tests) | ✅ Running | Gemma 4 MoE (94.4%), Qwen 35B (90.9%), Qwen 27B (89.2%), E4B (87.5%) |
| **Claw-Eval** (30-52 tasks) | ✅ Running | Qwen 27B (86/103), Qwen 122B (89/103), Gemma 4 MoE (0.634), MiniMax (0.635) |
| **Custom suites** (10 suites) | ✅ Running | All models pass |
| **WildBench** (1,024 tasks) | ✅ Available | GPT-5.4, Sonnet, Haiku, Qwen 9B variants |
| **Proto-bench coding** (5 tasks) | ✅ Prototype | Qwen 122B (0.800 baseline) |
| **Terminal-Bench 2.0** | ❌ Not yet | Target for protoCLI |
| **SWE-bench Pro** | ❌ Not yet | Need Docker setup |
| **LiveCodeBench** | ❌ Not yet | Need temporal gating |
| **Aider Polyglot** | ❌ Not yet | Easy to add |

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
