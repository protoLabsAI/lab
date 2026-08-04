# AI Agent Guidelines - Kaggriculture

This file contains instructions for AI coding assistants (Gemini, Codex, Claude, ChatGPT, Cursor, Antigravity, etc.) working on the **Kaggriculture** repository.

> [!IMPORTANT]  
> **MANDATORY CONTEXT READ BEFORE PROCEEDING**  
> Before making any code changes, creating implementation plans, or debugging issues, you **MUST** read the following documentation files to understand the project domain, rules, state schema, and architecture:

## Mandatory Files to Read

1. **[`overview.md`](file:///Volumes/Important/Office/White%20Way%20Web/Github/kaggriculture/overview.md)**
   * **What it contains:** The complete rules, mechanics, action formats, price curves, unit yields, town demand schedules, and observation JSON schemas for the Kaggle Kaggriculture simulation competition.
   * **Why read it:** All bot logic, action choices, price estimations, and simulation handling MUST adhere strictly to the mechanics detailed in this document.

2. **[`python_bot/README.md`](file:///Volumes/Important/Office/White%20Way%20Web/Github/kaggriculture/python_bot/README.md)**
   * **What it contains:** Architecture and test instructions for the Python submission agent (`agent.py`, `strategy_rules.py`, `test_agent.py`).

3. **[`web/README.md`](file:///Volumes/Important/Office/White%20Way%20Web/Github/kaggriculture/web/README.md)**
   * **What it contains:** Setup and running instructions for the web frontend / visualizer app.

4. **[`implementation_plan.md`](file:///Volumes/Important/Office/White%20Way%20Web/Github/kaggriculture/implementation_plan.md)** & **[`walkthrough.md`](file:///Volumes/Important/Office/White%20Way%20Web/Github/kaggriculture/walkthrough.md)**
   * **What it contains:** Current implementation state, goals, and test verifications.

---

## Workspace Structure

```
kaggriculture/
├── AGENTS.md                  # AI agent instructions (this file)
├── overview.md                # Full competition specification & rules
├── Kaggriculture-Kaggle-*.pdf # Original Kaggle competition PDF
├── python_bot/                # Submission bot codebase
│   ├── agent.py               # Main agent entry point (kaggle_environments compatible)
│   ├── strategy_rules.py      # Rule engine for farm management & market trading
│   ├── test_agent.py          # Unit tests & local simulation runner
│   └── README.md
└── web/                       # Web frontend / visualization interface
    ├── src/                   # React / Vite source code
    └── README.md
```

---

## Core Operational Rules for AI Agents

1. **Check Domain Rules First:** Always consult [`overview.md`](file:///Volumes/Important/Office/White%20Way%20Web/Github/kaggriculture/overview.md) whenever altering bot behavior, price logic, crop selection, feeding, watering, or market order formatting.
2. **Preserve API & Submission Compatibility:** `python_bot/agent.py` must expose a valid `agent(obs)` entry point compatible with `kaggle_environments` and Kaggle submission constraints (<=100 MiB tar.gz).
3. **Run Tests After Changes:** Always verify changes by running unit tests (e.g. `python3 -m unittest python_bot/test_agent.py` or `pytest`) and checking web build (`npm run build` in `web/`).
4. **Benchmark Every Strategy Change:** Any change to `python_bot/agent.py` that can alter game decisions must be run through `python_bot/run_official_tournament.py` before it is described as verified, packaged for release, or submitted. Run the configured benchmark suite (candidate vs `pass`, `random`, `starter`, and the previous approved artifact); if the official engine is unavailable, report the benchmark as blocked and do not claim a performance improvement.
