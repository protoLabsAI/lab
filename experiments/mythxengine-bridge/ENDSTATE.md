# The End-Game — the bottomless well (2026-07-08)

*The 1-year target world the engine team works back from. Everything below rung 0 is a build
target; rung 0 is done. Reports up to `THESIS.md` (the flywheel).*

## The one idea
A **bottomless well is not a big set of tasks — it is a task GENERATOR.** Hand-authored tasks are
finite (that's the τ-bench diversity ceiling we proved empirically). The end-state is a **living,
procedurally-generated, multi-agent WORLD where tasks EMERGE** from the interaction of mechanics,
scarcity, a living economy, and other agents — and every outcome is deterministically verifiable.
Diversity comes from seeds, depth from persistence, difficulty from other agents, and *supply* from
emergence. That combination is effectively infinite.

## The end-state, concretely — "The Persistent Frontier"
A seeded, persistent, MMO-shaped simulation an agent enters through a semantic tool surface
(isomorphic to protoAgent). It is populated by **many agents at once** — the fleet agents we train,
strong teacher agents, adversarial opponents, and NPC actors. There are **no authored tasks**;
challenges *arise*: scarcity → provisioning; opponents → defense/competition; contracts & factions →
negotiation/collaboration; discoveries → exploration; a persistent economy → long-horizon
accumulation. The agent must **Observe** the world/economy/other agents, **Orient** (build a
world-model, read threats and opportunities), **Decide** under uncertainty *and* other agents' moves,
**Act** over windows, and **Recover** as the world shifts — across long horizons, in **collaboration**
(delegate, form coalitions) and **competition** (out-maneuver opponents). Every outcome is
byte-replay-verifiable; no LLM judge, ever.

## The seven pillars (each a build target)
1. **Emergence over authoring.** Tasks arise from mechanics + world state, not a task list. This is
   the property that makes the well bottomless and beats the data-ceiling.
2. **Procedural + persistent.** Seeded worlds (infinite fresh instances = diversity, contamination-
   free held-outs) + multi-day persistent economies (depth, genuine long-horizon).
3. **A composable mechanic library.** Survival, trade, construction, investigation, defense, research,
   movement/manipulation (robotics) — packs that *combine* into rich worlds. The OODA×domain matrix
   realized as a library, not a fixed set.
4. **A multi-agent society.** Fleet + teacher + adversarial + NPC agents co-inhabit one world;
   delegation, coalitions, markets, and conflict → **orchestration and collaboration emerge** (the
   exact skills protoAgent's `task()`/`delegate_to` layer needs).
5. **Verifiable-by-construction reward, everywhere.** Replay determinism + world-state assertions +
   economic value + dense subgoal events, at every timescale. Never a judge. (The reward-trust moat.)
6. **Self-calibrating headroom.** Adversarial world-synthesis (generator/discriminator, CUA-Gym-style)
   + auto-computed greedy/frontier/oracle bands keep every world at the *learnable frontier* — no
   shallow worlds (the COURIER-v0 lesson), no saturated ones.
7. **The transfer instrument, built in.** Every world exposes a held-out-seed protocol + a mapping
   from its skills to a real-tool benchmark (R4), so its training value is *measurable* the moment
   it's built.

## Work-back build ladder (the engine sequences from the end-state; builds forward)
    RUNG 0  now      single-skill · single-agent · terminal reward · hand-picked worlds     [DONE — pipeline proven, M1 parity]
    RUNG 1  ~Q1      MECHANIC DEPTH + DENSE REWARD — dependency chains (FOREMAN), dynamic
                     events (spoilage/blocked routes → RECOVERY), restraint (when-not-to-act).
                     + dense subgoal events (#325 ckbox-2). Richer SINGLE-agent worlds.
                     → first place we expect R4 to twitch off parity.
    RUNG 2  ~Q2-3    THE MULTI-AGENT SOCIETY — A2A in-world: teammates (delegation/collab) +
                     opponents (competition → TRUE OODA, the world moves under you). Orchestration
                     worlds. This is where protoAgent's real (delegation) skills train.
    RUNG 3  ~Q3-4    PERSISTENCE + ECONOMY — multi-day worlds, seasons, resource economies,
                     contracts, factions. Long-horizon; tasks start to EMERGE, not be picked.
    RUNG 4  ~Q4+     EMERGENCE + SELF-CALIBRATION — procedural task-emergence, adversarial world-
                     synthesis, auto-oracle. The bottomless well fully realized: worlds generate
                     their own diverse, hard, long-horizon tasks at the learnable frontier, forever.

## How we know it's working — the transfer curve
At each rung we run the *same* proven pipeline and watch **R4 (claw + the protoAgent bench)**. The
thesis: transfer stays **flat on shallow worlds** (rung 0-1), **lifts as complexity crosses a
threshold** (rung 2-3), and **compounds at rung 4** (bottomless diverse verified data). We are
instrumented to catch that inflection — *"transfer emerges at world-complexity X"* is the research
story **and** the go/no-go signal at every rung. We never bet blind: each rung is gated on R4 moving.

## Division of labor at end-state
- **Engine (mythxengine-sdk):** the world — mechanics, the society, emergence, determinism, the
  headroom instruments. Builds up the rungs.
- **Lab (us / protoResearcher):** the machine — BC/RL/measure — + the transfer instrument (R4) + the
  steering: fleet-trace gap-analysis tells the engine *which mechanics to prioritize next*.
- **Fleet (protoAgent):** production reality — the traces that steer which worlds get built, and the
  deployment target the trained agents return to.
All three turn one flywheel. The engine makes the well; the lab draws from it; the fleet says where to dig.

## The honest frame
Rung 0 gave parity — expected, and it proved the machine. The value is the *ladder*, and the ladder
is the engine team's to climb. This doc is the top of it: build toward a world where interesting,
diverse, complex, long-horizon, collaborative, adversarial tasks are not authored but **generated,
verified, and bottomless.** Work back from here, one rung at a time, R4 as the compass.
