# Training-side contribution to the Agent-Zoo direction (2026-07-07)

Our (lab) response to the mythxengine-sdk agent-zoo / RLVR-worlds direction. **Division of labor:
they own the engine + SDK (the env factory, worlds, packs, harness); we own the ML + data +
training (R2 BC / R3 RLVR / R4 transfer, the reward + distillation science).** They build the
worlds; we say which worlds are worth building and prove the training loop.

## Where they are (current, verified from code — `b0992e5`)
- **Settled:** RLVR env-factory thesis · verifiable reward (deterministic replay + leaderboard,
  never an LLM judge) · semantic tool surface ≈ protoAgent · the goal-time/window-time split.
- **Proven:** COURIER v1 has WIDE learnable headroom — **greedy 136 (37%) / Ornith-35B 152 (41%)
  / oracle 370 (100%)**; "59% unclaimed, a clear RL gradient, no saturation."
- **Not started:** R2 (BC), R3 (RLVR), R4 (transfer eval) — unassigned. Env side produced a
  measured gradient + ~302 SFT rows and is waiting for a consumer. **That consumer is us.**

## What we bring
1. **Independent validation of their thesis.** We proved the *data-ceiling* from the training side
   (SFT scales to +0.028 then plateaus; white-box KD on the same corpus is flat → the corpus is the
   limit, not the method — [[project_distill_base_decision]]). Their "environments are the scarce
   input" bet is confirmed by an orthogonal method.
2. **The R2/R3/R4 loop, already run end-to-end.** `train_lora.py`/`train_lora_kd.py` →
   `merge_lora` → NVFP4 requant → serve → claw gate + the ×3 methodology. Their SFT rows are
   OpenAI chat-format w/ tool schemas — directly consumable.
3. **The R4 transfer harness — "the bet's judgment day," which they can't build alone.** We own the
   held-out real-tool agentic benchmark: **claw + FC + ×3 + local-judge discipline**, plus the
   protoAgent bench axis ([[project_protoagent_evals]]).

## Three inputs to their live decisions (from the training side)
1. **Surface debate (flat-ReAct vs commander) → NOT either/or.** protoAgent (verified) is a
   flat-ReAct tool-loop **core** + delegation-as-a-tool (`task()` subagents, `delegate_to`,
   `set_goal/update_goal_plan`). So `scenario_survive` trains protoAgent's **core loop**;
   `commander_band` trains its **delegation layer**. Keep both; they target different real
   capabilities. Settle *depth/priority* empirically via R4, not by argument.
2. **Don't block R2 on the story_survive oracle — start R2 on COURIER v1 now.** We learned the hard
   way (COURIER v0, our τ-bench ceiling): without a headroom ceiling you can't tell "good model"
   from "shallow world." COURIER v1 has the proven gap *today*; story_survive is half-instrumented
   (frontier band, no oracle). Train on the proven world, run R4, resolve the surface debate with
   that data.
3. **Their terminal-scalar reward will hit the exact plateau we just hit — and we built the fix.**
   Reward is episode-level `banked/alive/rank`; the review's own finding #5 wants dense subgoal
   events but they're unimplemented. Tonight's data-ceiling was partly a *reward* problem (τ²'s
   AND-collapse of decomposed checks). We have the tooling: **`reward_dense.py`** (fraction-of-checks
   + potential-based shaping + monotone-prefix, from the long-horizon DD). Their deposit-events ×
   value-catalog *is* the subgoal signal — aggregated to a scalar before the trace, same AND-collapse,
   same fix. Dense reward is not optional for R3.

## Milestone 1 — the cheapest test of the entire bet
```
COURIER v1 (proven headroom) → Ornith rollouts (commander_band, several seeds)
  → reward-filter/label (zoo_traces) → our masked-BC (2B) → NVFP4
  → in-sim eval on HELD-OUT COURIER seeds (byte-replay = contamination-clean)
  → R4: claw before/after (out-of-sim)      ← fires the transfer test in the same motion
```
Closes R2 *and* the first R4 shot, on the world with a proven gradient + a pipeline we just proved.
**claw moves after sim-BC → transfer bet validated. Flat → we learned it in a day.** Highest-leverage
next experiment in the program.

## World-design framework — what worlds are worth building (the forward ask)
**A world earns its build cost when it maximizes:**
`(learnable headroom) × (a skill our small models measurably LACK) × (transfer to protoAgent's real
surface) × (verifiable-by-construction reward)`. Skills are the unit (their rule); we pick skills by
our *empirical failure modes*, not intuition.

Prioritized skill → world wishlist (training-side, tied to findings):

    rank  skill (why it's ours to ask for)                      world archetype              status
    1     RESTRAINT / when-NOT-to-act                            acting costs a resource,     PROPOSE
          arm-D: small models CRATER hallucinating tool calls    or "wait for the signal";    (new)
          (0.335). Our #1 empirical failure. protoAgent needs    reward penalizes over-action
          tool discipline.
    2     RECOVERY / replan-on-failure                           plans break mid-execution    PROPOSE
          KD/exposure-bias: the plateau IS the student can't     (route closes, resource      (new)
          reproduce teacher trajectories. Recovery traces are    spoils, delegate fails) →
          exactly what we lack; a world that forces detect+       detect + replan
          recover generates them on-policy.
    3     DEPENDENCY-CHAIN / long-horizon depth                  FOREMAN: gather→refine→       PROPOSE
          current worlds are ~40-window shallow. Prereqs force   deliver→build; ordered       (candidate)
          real horizon + planning. Maps to goal-plan.            prerequisites
    4     DELEGATION-under-budget                                heterogeneous delegates      EXTEND
          protoAgent DOES delegation (task/delegate_to). Route   with costs/latencies;        courier→
          sub-tasks well under a budget.                          reward = good routing        commander
    5     DEDUCTION / evidence→inference                         SLEUTH: certified unique-    IN PROGRESS
          protoAgent's researcher subagent shape (plan→gather→   solution puzzle              (they're adding)
          synthesize). Verifiable by unique solution.
    -     PROVISION-&-SURVIVE / resource allocation over time    story_survive                CANONICAL
          allocate scarce food belly-vs-larder over horizon.     (needs oracle)               (in progress)

**The two highest-value NEW asks are RESTRAINT and RECOVERY** — they target our two proven small-model
failures (over-tool-calling, exposure-bias) that no current world exercises, and both transfer directly
to protoAgent's tool-discipline + replanning skills. FOREMAN adds the genuine long-horizon depth the
current ~40-window worlds lack.

## Coordination flags
- **Doc/code divergence:** `agent-zoo.md` stops at Rev 4; the scenario reframe lives only in commits
  (`748f246`+). Worth them updating so "settled" means the same thing on both sides.
- **Trace-schema stability** is our data-integrity prerequisite: pin the harness view (compact obs +
  tool schemas) in the pack so BC/RL datasets don't silently invalidate across harness versions
  (their open question #1 — we care about it most).
