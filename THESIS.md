# protoLabs — THE NORTH STAR: the self-improving agent-fleet flywheel

*This is the thesis every experiment reports up to. If a piece of work doesn't turn a gear in this
flywheel, question why we're doing it.* (2026-07-08)

## The one idea
A **self-improving agent-fleet flywheel**: production agents emit traces → we mine the traces for
where they fail → we build deterministic verifiable **environments** that isolate those failures →
teacher + fleet agents play them → verified-reward data → we distill/RL **cheap small agents** →
deploy them back to the fleet → more traces. The loop tightens on every turn.

```
        ProtoAgent Fleet (production — real agents, real work)
                          │  traces (successes + failures, real distribution)
                          ▼
   OBSERVE ── mine fleet traces: where do agents fail? what skill is missing?
   ORIENT  ── gap analysis: this failure = a missing skill = a missing world
   DECIDE  ── which environment to build / improve / invent
                          ▼
   ENV FACTORY (MythXEngine) ── deterministic verifiable worlds isolating that skill
                          ▼
   ROLLOUTS ── teacher + fleet agents play → verified-reward traces (BC + RL data)
                          ▼
   TRAIN ── distill / RL small agents   (quant + serving = the forge)
                          ▼
   DEPLOY ── better, cheaper agents → back into the Fleet
                          └────────────► more traces → the loop tightens
```

**The recursion:** the fleet's agents run OODA/ReAct loops; **the lab runs OODA on the fleet.** Fleet
traces = our Observe, gap-analysis = Orient, world-selection = Decide, build→train→deploy = Act. Same
loop, one level up. That is why OODA is both the agent representation *and* the lab process.

## The agent representation — OODA & ReAct as trace shapes
- **ReAct** (`Thought→Act→Observe`) — the loop for *static* task execution.
- **OODA** (`Observe→Orient→Decide→Act`) — the loop for *dynamic/adversarial/uncertain* worlds; the
  extra step is **Orient** (updating your world-model as it changes under you).
- **The world's dynamics choose the loop.** Static world → ReAct trace; a world that shifts under the
  agent → OODA trace. Design lever: `world dynamics → loop shape → skill in the trace`.

**Strong AND diverse data** = `LOOP (invariant → transfers) × DOMAIN (variant → diversifies)`. The
loop is the same whether surviving, routing, investigating, negotiating, or commanding a robot arm.
Train the loop across many domains → the skill generalizes while domain-spread beats the data-ceiling
(the empirically-proven plateau of fixed single-domain data — see `experiments/agentic-data/RESULTS.md`).

**World-design taxonomy — the OODA × domain matrix** (fill it over time):

    OODA phase stressed     world archetype          domains (columns = diversity)
    OBSERVE→ORIENT          SLEUTH (evidence→picture) investigation, diagnosis, forensics
    ORIENT (re-model)       STATE-TRACK / RECOVERY    survival, logistics-disruption, ROBOTICS-perception
    DECIDE (commit; when-NOT) RESTRAINT, COURIER       trading, security, triage, resource-allocation
    ACT (execute+verify)    FOREMAN (dependency chain) manufacturing, build-pipelines, ROBOTICS-manip
    CLOSE-LOOP (replan)     RECOVERY                   any dynamic domain
    ORCHESTRATE (A2A)       COMMANDER, delegation      multi-agent teams, sub-agent routing
    PERSIST (multi-day)     seasons, standing tasks    long-horizon multi-agent operations

Robotics is a first-class column: MythXEngine's **goal-time/window-time split IS the robotics
high-level-planner / low-level-controller hierarchy** (SayCan/RT-style). A robotics world = semantic
manipulation/navigation goals + a physics-y execution layer with real failure modes.

## Why every lab thread is a gear (the cohesion)
    Quant + serving (substrate #2)  → the FORGE: cheap fast agents + cheap RL rollouts (NVFP4 speed)
    Evals (claw, proto-bench, ×3)   → MEASUREMENT + REWARD: R4 transfer test, fleet-trace analysis, graders
    Distillation / tiny-models      → the AGENTS that power the fleet cheaply
    RLVR / reward science           → the TRAINING SIGNAL (reward_dense, PBRS, verifiable-by-construction)
    MythXEngine                     → the ENV FACTORY + persistent multi-agent substrate
    ProtoAgent Fleet                → PRODUCTION + the Observe (real traces)
    reward-trust discipline         → trustworthy reward at scale (immutable boundary, never an LLM judge)
None were detours. Tonight's data-ceiling finding was the last *conceptual* gap — proof that fixed data
runs out and you need the factory.

## The three-repo contract (this is multi-repo by nature)
    protoLabsAI/mythxengine-sdk   OWNS the engine + SDK — the env factory, worlds, packs, harness, determinism
    protoLabsAI/protoAgent        OWNS the fleet — production agents + the trace SOURCE (the Observe)
    protoLabsAI/protoLab (lab)    OWNS ML + data + training — R2 BC / R3 RLVR / R4 transfer, reward + distill science
The flywheel only turns if **traces flow across the seams**: fleet→lab (Observe), engine→lab (rollout
data), lab→fleet (deployed agents). Each seam is a schema contract, not a handoff.

## Honest state — we have the gears; they aren't connected yet
Every gear is proven *individually* (serving, distill run-to-conclusion, reward science, a
proven-headroom world, a fleet in production). What's missing is the *connections*. The plan is to
close them one at a time:

1. **First quarter-turn (Milestone 1):** COURIER v1 → Ornith rollouts → our masked-BC (2B) → NVFP4 →
   in-sim held-out-seed eval → **R4: claw before/after**. One full env→train→measure pass on the
   world with proven headroom. If claw moves, the transfer bet is validated — cheaply.
2. **Wire the Observe:** protoAgent production traces → lab ingestion → gap-analysis *drives* world
   selection (the loop's steering).
3. **Grow environments up the ladder:** single-agent → A2A orchestration → long multi-day multi-agent,
   as the fleet's real tasks demand — with the OODA×domain matrix as the map.

## The data-quality asks (what makes trace data worth training on)
- **Surface the Orient step** in traces (scratchpad-as-world-model) — turns ReAct data into OODA data.
- **Dense subgoal reward** (their deposit-events × value-catalog = the signal; reuse `reward_dense.py`).
- **Pin the harness view** (compact obs + tool schemas) in the pack so datasets don't silently drift.
- **Verifiable-by-construction reward only** — deterministic replay, never an LLM judge, on every seam.
