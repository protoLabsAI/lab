# Milestone 1 — the flywheel's first quarter-turn (2026-07-08)

**First full env→train→measure loop, end-to-end.** Proved the pipeline; the payoff is on the
complexity ladder (see ENDSTATE.md).

## What ran
    generate   scenario_survive story_courier × 30 seeds (24 train / 6 held-out) → OODA-grade traces
               (287 world-model orients; the #326 Orient step, genuine strategic assessments)
    convert    zoo_traces --survive → 24 canonical SFT rows (system+tools+reward+orient)
    BC         masked-SFT Qwen3.5-2B (24/24 masked, loss 1.15→0.88) → merge → NVFP4
    R4         claw (30 tasks, local Ornith-35B judge) — courier-BC-2B vs base-2B

## Result — parity (honest)
    courier-BC-2B   claw 0.642  (n=30, single-trial)
    base-2B         claw 0.604  (×3)
    → +0.038, but single-vs-×3 AND within noise (~0.045 SE). 0.642 == base single-trial baseline.
    VERDICT: parity — no clear transfer, no harm. Exactly as predicted for 24 toy OOD trajectories.

## The signal that IS there (weak, not overclaimed)
BC on a completely different domain (survival/logistics) left claw **intact/nominally-up, not
degraded** — no game-specific junk learned (the arm-D failure mode would drop claw; it didn't).
Weakly consistent with "OODA harness-level skills are domain-neutral" (the transfer hypothesis).
A whisper, not a signal — needs complexity + volume.

## The deliverable
The machine is **world-agnostic**: every future world plugs into
`scenario_survive → zoo_traces → train_lora → requant → claw R4` with zero re-plumbing. We now
have an instrument to catch the moment transfer emerges as worlds get harder — the
**transfer-emerges-at-complexity-X curve** is measurable from here. Payoff = the complexity ladder.
