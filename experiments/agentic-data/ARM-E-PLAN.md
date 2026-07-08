# arm-E — parametric-skills lite: modular vs monolithic (2026-07-06)

Tests the "Parametric Skills" (arXiv 2606.30015) thesis on our stack: do skill-SPECIALIZED
adapters beat the MONOLITHIC arm-B on their own skill? (Lite: a skill-oracle picks the adapter
by the task's known action — the simple stand-in for their text→LoRA hypernetwork.)

- **Generalist:** arm-B (all 435 Ornith τ-bench trajectories, one LoRA).
- **Specialists:** 4 LoRAs — cancel/return/exchange/modify (~100–120 traj each), 4 epochs.
- **Eval (held-out, deterministic):** τ-bench retail **TEST** split, filtered per skill by the
  task's primary action. For each skill's test tasks: reward(monolithic arm-B) vs reward(matched
  specialist). Needs Ornith on :8000 as the user-sim → run AFTER the more-Ornith generation frees :8000.
- **Verdict:** if per-skill specialist > monolithic on its own skill → modular/parametric skills
  beat monolithic distillation on our stack (validates the paper vs tonight's saturation wall).
- Follow-up if it wins: vLLM multi-LoRA serving (base + skill library, route per request) → the
  real "parametric skill library"; then the hypernetwork is worth chasing.
