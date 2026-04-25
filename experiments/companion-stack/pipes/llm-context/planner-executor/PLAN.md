# planner-executor — dual-model agentic loop for ORBIS

**Blog headline:** "Two models, one brain: why we split planning from execution in our voice agent"

**One-liner:** Route every agentic turn through a fast no-thinking executor (35B MoE), with a thinking planner (27B dense) that writes the plan, reviews execution, and corrects failures — without the user waiting for CoT on every tool call.

---

## The hypothesis

A single LLM doing planning + execution + review on every turn is wasteful:
- Thinking is expensive (~200–500ms extra TTFT per turn)
- Most turns are *execution* — follow a plan, call a tool, return a result
- Planning and reviewing only matter at the *boundaries* of a task

Split the work:

```
User turn
   │
   ▼
[Planner — Qwen3.6-27B-FP8, :8000, thinking ON]
   │  "Here's the plan: step 1, step 2, step 3"
   │  Produces: structured task plan (JSON)
   ▼
[Executor — Qwen3.6-35B-A3B-FP8, :8002, thinking OFF]
   │  Executes each step: tool calls, retrieval, side-effects
   │  231 tok/s, 23ms TTFT, 100% tool accuracy
   │  Produces: step results + accumulated context
   ▼
[Planner — review pass]
   │  "Is the result correct? Is the plan complete?"
   │  Produces: pass | retry(step, reason) | replan
   ▼
[Response to user]
```

The planner thinks *once* at the start and *once* at the end (or on failure).
The executor never thinks — it just does.

**Expected wins:**
- Reduced per-turn latency (no thinking tokens on tool calls)
- Better plan quality (27B thinking > 35B no-thinking for decomposition)
- Natural failure recovery (planner reviews, not executor)
- Voice-friendly: executor's 23ms TTFT keeps the pipeline responsive

---

## ORBIS integration target

```
Browser ──WebRTC──▶ STT ──▶ [Planner] ──plan──▶ [Executor loop] ──▶ [Planner review] ──▶ TTS ──▶ Browser
                                │                                            │
                                └─────── replan on failure ─────────────────┘
```

The planner/executor boundary maps cleanly onto ORBIS's existing tool-call loop.
Planner uses the gateway as `local` (port 8000).
Executor uses the gateway as `local-voice` (port 8002).

---

## Scope

### In scope
- Python harness: `planner_executor.py` — orchestrates the two-model loop
- Structured plan format (Pydantic): `PlanStep`, `Plan`, `ExecutionResult`, `ReviewDecision`
- Retry logic: max 2 replans before falling back to planner-only answer
- Benchmark suite: 20 multi-step agentic tasks (tool-calling, retrieval, multi-turn)
- Latency measurements: planner TTFT, executor TTFT, total loop time vs single-model baseline
- Quality measurements: task completion rate, tool accuracy, retry rate

### Out of scope (for this experiment)
- Streaming responses to the user mid-execution (Phase 2)
- Persistent plan state across turns (memory pipe handles that)
- Fine-tuning either model on this loop's data (future experiment)

---

## Tier-0 baselines (mandatory)

Before claiming the dual-model loop wins, measure:

1. **Single model (27B thinking)** — planner alone, no executor split, on the same 20 tasks
2. **Single model (35B no-thinking)** — executor alone, no planning pass, on the same 20 tasks
3. **Planner/executor loop** — this experiment

Metrics per baseline: task completion rate, tool accuracy, total latency (wall clock, not tok/s).

---

## Task suite

20 multi-step agentic tasks covering:
- 5 × multi-tool chaining (e.g. "find my next meeting, check the weather there, draft a travel note")
- 5 × conditional branches (e.g. "if X then do Y else do Z" with tool-state dependency)
- 5 × error recovery (tasks where step 1 deliberately returns an error, model must adapt)
- 5 × long-horizon (4+ step plans, testing plan coherence across steps)

Tasks are synthetic but ORBIS-shaped (calendar, email, facts lookup, web search stubs).

---

## Plan format

```python
class PlanStep(BaseModel):
    step_id: int
    description: str
    tool: str | None          # None = LLM-only step
    args: dict
    depends_on: list[int]     # step_ids this step waits for

class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    success_criteria: str

class ExecutionResult(BaseModel):
    step_id: int
    status: Literal["ok", "error", "skipped"]
    output: str
    tool_calls: list[dict]

class ReviewDecision(BaseModel):
    verdict: Literal["pass", "retry", "replan", "abort"]
    failed_step: int | None
    reason: str | None
    revised_steps: list[PlanStep] | None   # populated on "replan"
```

---

## Files

```
planner-executor/
├── PLAN.md               ← this file
├── RESULTS.md            ← populated after experiment
├── BLOG.md               ← draft after results
├── planner_executor.py   ← core harness
├── models.py             ← Pydantic schemas
├── tasks/
│   ├── multi_tool.py     ← 5 multi-tool tasks
│   ├── conditional.py    ← 5 conditional tasks
│   ├── error_recovery.py ← 5 error-recovery tasks
│   └── long_horizon.py   ← 5 long-horizon tasks
├── eval/
│   ├── run_eval.py       ← runs all 20 tasks × 3 baselines
│   └── metrics.py        ← latency + accuracy scoring
└── results/              ← raw JSON outputs per run
```

---

## Exit criteria

- All 3 baselines measured on all 20 tasks
- Planner/executor shows ≥ 10% latency reduction vs 27B-alone on execution-heavy tasks
- Planner/executor shows ≥ equal or better task completion vs 35B-alone
- RESULTS.md written with honest numbers, failure analysis, and confusion table
- BLOG.md drafted

---

## Schedule

| Phase | What |
|-------|------|
| Day 1 | Implement `models.py`, `planner_executor.py`, stub tasks |
| Day 2 | Complete all 20 tasks, run baseline evals |
| Day 3 | Full eval run × 3 baselines, write RESULTS.md |
| Day 4 | BLOG.md draft |
