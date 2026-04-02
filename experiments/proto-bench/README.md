# proto-bench — protoCLI Benchmark Harness

Harness for evaluating and optimizing protoCLI on coding benchmarks (SWE-bench, Terminal Bench 2.0).

## Goal

Get protoCLI on the [Terminal Bench 2.0 leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0). Current top: ForgeCode + Opus 4.6 at 81.8%.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                proto_agent.py                    │
│   (task loading, prompt formatting, scoring)     │
└────────────────────┬────────────────────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
     ┌─────────────┐   ┌──────────────┐
     │  protoCLI   │   │  APO Loop    │
     │  (headless) │   │  (critique →  │
     │  proto -p   │   │   edit →      │
     │             │   │   evaluate)   │
     └──────┬──────┘   └──────────────┘
            │
     ┌──────┴──────┐
     │ LLM Backend │
     │ (vLLM/      │
     │  gateway)   │
     └─────────────┘
```

Modeled after Microsoft Agent Lightning's `examples/claude_code/` harness.
protoCLI mirrors Claude Code's architecture, so the adapter pattern translates directly.

## Usage

```bash
# SWE-bench tasks
python experiments/proto-bench/proto_agent.py swebench \
  --dataset-path experiments/proto-bench/tasks/swebench_samples.jsonl \
  --model protolabs/local \
  --max-turns 25

# APO system prompt optimization
python experiments/proto-bench/proto_agent.py apo \
  --dataset-path experiments/proto-bench/tasks/swebench_samples.jsonl \
  --prompt-path ~/.proto/AGENTS.md \
  --rounds 2

# Terminal-bench (via Harbor)
harbor run -d terminal-bench@2.0
```

## Strategy

1. **Baseline** — Run protoCLI + cloud model (Opus/GPT-5.4) to establish score
2. **APO** — Optimize system prompt using binary pass/fail as reward signal (cleaner than LLM judge)
3. **Local model** — Run with Qwen 27B/122B to measure gap, verify APO gains transfer
4. **DPO** — If APO isn't enough, collect rollouts for preference training (pipeline proven in agent-lightning experiment)

## Key Differences from Agent Lightning Experiment

| Aspect | agent-lightning (research) | proto-bench (coding) |
|--------|---------------------------|---------------------|
| Agent | protoResearcher | protoCLI |
| Tasks | Research queries | SWE-bench / terminal-bench |
| Reward | LLM judge (5 dimensions) | Binary pass/fail (ground truth) |
| Prompt | SOUL.md | AGENTS.md + system prompt |
| Execution | HTTP API | CLI headless mode |

## Files

- `proto_agent.py` — Main harness (SWE-bench, terminal-bench, APO modes)
- `tasks/swebench_samples.jsonl` — 5 SWE-bench smoke test instances
- `results/` — Benchmark outputs

## Dependencies

- protoCLI installed and on PATH (`proto` command)
- For SWE-bench: Docker (instances run in containers), `swebench` Python package
- For terminal-bench: Harbor CLI
- Agent Lightning reference: `~/dev/agent-lightning-ref/examples/claude_code/`

## Reference

- [Terminal Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0) — 122 entries, top 81.8%
- [Agent Lightning claude_code example](https://github.com/microsoft/agent-lightning/tree/main/examples/claude_code) — SWE-bench harness for Claude Code
- [protoCLI](https://github.com/protoLabsAI/protoCLI) — Model-agnostic terminal agent (Claude Code fork)
