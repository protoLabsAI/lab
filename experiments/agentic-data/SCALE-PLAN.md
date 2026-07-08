# Scale-to-lift plan — 5–10k diverse verified Ornith trajectories (2026-07-06)

**Goal:** push the Ornith verified corpus from 1,314 → **5–10k DIVERSE** trajectories and retrain
scaled-arm-B-v2, testing whether teacher-consistent verified SFT crosses base (claw **0.642**) into
a **net lift**. Tonight's trend says it should: 435→1,314 (3×) moved claw 0.618→0.638 (parity),
monotonic, zero degradation ([[RESULTS.md]]).

**The real constraint is task diversity, not GPU time.** Retail-train = 500 unique tasks; we've
already 3× passed them (greedy + 2× temp-0.7). More passes = repetitive echoes. The lift needs
*new task distributions*, so this plan is mostly about task sources.

## Phase 1 — τ²/τ³-bench (low friction, high diversity) ← do this first

Same lineage as our working τ-bench: `git clone sierra-research/tau2-bench && uv sync`. Deterministic
action-check reward (`evaluation_criteria.actions`, `reward_basis`), litellm agent+user, trajectories
saved to `data/simulations/`, train/test splits. **New domains vs what we used: telecom, banking_knowledge**
(+ retail-v2, airline-v2). CLI: `tau2 run --domain X --agent-llm A --user-llm U`.

Steps:
1. Install in an isolated venv (like our tau-bench). **Verify at impl:** does `--agent-llm`/`--user-llm`
   take separate `api_base`? If yes → dual-endpoint is native (no patch). If it shares `OPENAI_API_BASE`
   → reuse tonight's `user.py` api_base patch pattern.
2. Confirm reward is deterministic + trajectories carry full `messages` (adapt `filter_tau.py` to the
   `data/simulations/` schema → canonical Trajectory, `verified=True, reward=1.0`).
3. Generate Ornith rollouts on **train splits** of: telecom, banking_knowledge, retail-v2, airline-v2
   (skip `mock`). Greedy + 1–2× temp-0.7 per domain for diversity.
4. **Yield budget:** retail/airline ~65–85%, telecom/banking harder (~50–60%, like airline tonight).
   Rough: 4 domains × ~200–500 tasks × ~2 trials × ~65% → **~2–4k new verified.**

## Phase 2 — one AgentGym env (different domain, medium friction) ← if Phase 1 short of 5k

`WooooDyy/AgentGym-RL` (MIT, vLLM-native). Pick **WebShop** (retail-adjacent, deterministic reward)
or **ALFWorld** (embodied). NOT τ-bench-compatible → needs a small rollout adapter (its own agent
loop + reward). Adds genuinely different domain (web-shopping / embodied). ~1–2k more verified.
Treat as stretch; Phase 1 may reach 5k alone.

## Generation setup (dedicated, fast)
- Serve a dedicated generation Ornith OR round-robin both prod replicas (`:8000`+`:8003`).
- Concurrency 12–16 (tonight's 6, contended, was the slow part). **~1–1.5h / 1,000 runs dedicated;
  ~2–2.5× that at high conc + both replicas.** So ~4–8k runs = **~3–6h dedicated.**
- Prod: either let the gateway absorb it, or pause `vllm-replica-b` for a dedicated generation card.

## Corpus → train → gate
- Merge all verified (tonight's 1,314 + Phase-1/2) → `build.py` dedup (content-hash) → 5–10k unique.
- Train scaled-arm-B-v2 (same masked recipe, r32, lr 5e-5, 2–3 epochs) → merge → NVFP4 → `run.sh` gate.
- **Success = claw > 0.642 (net lift over base).** Also watch FC↑ / custom-held (graceful scaling).

## Eval integrity (contamination)
- Primary eval unchanged: **claw + FC + custom** (uncontaminated — different task set from τ-bench).
- Add τ²/τ³ **test** splits to the held-out anchor; `build.py` contamination filter drops any train
  row whose prompt_hash collides. Generate from **train** only.

## Time estimate (one focused day)
    Phase-1 setup (install, verify reward/traj, adapt filter, 2-task smoke)   ~2–4h
    Phase-1 generation (dedicated, high conc, ~3–4k verified)                 ~3–6h
    build + train scaled-arm-B-v2 + quant + gate                              ~1h
    ────────────────────────────────────────────────────────────────────────────
    → 5k+ diverse verified + the lift test in ~1 day. Phase-2 adds ~half a day.

## Risks
- τ² domains harder → lower yield (budget 60–70%). - Dual-endpoint: verify τ²'s user-llm base_url.
- Diversity is the point: telecom/banking/web are OOD-adjacent to claw, so they should *broaden*
  agentic competence — but watch custom/FC don't drift (arm-D's over-tool-calling lesson: keep the
  mix balanced, and τ-bench's natural tool-use-AND-respond structure already does).
