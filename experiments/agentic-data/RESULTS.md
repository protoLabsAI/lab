# agentic-distill pilot — results (2026-07-06)

Distilling Ornith-1.0-35B → Qwen3.5-2B via LoRA-SFT. Question: what training data actually
*improves* a strong small agent? Every arm: same base, same LoRA (r32, MLP+attn targets,
assistant-masked loss, lr 5e-5), quant to NVFP4, gated on the full profile (judge = local
Ornith :8000, thinking-on). Baseline = base Qwen3.5-2B-NVFP4 (no training).

## Headline

**Teacher-consistent, deterministically-verified, in-domain data is the only data that helps —
and it SCALES: 3× data closed the gap to base parity (0.618→0.638), monotonically, no degradation.
Public data degrades regardless of teacher strength; composition/balance dominates. A net lift is
in reach with more verified data.**

**SCALING ARC — RESOLVED (2026-07-07).** More verified data DID cross into a lift: arm-B v2 (3,192
τ+τ² verified) beats base **×3-confirmed (0.632 vs 0.604, +0.028)** — small but consistent across
single-trial (0.646>0.642), ×3, and the monotonic curve. **Then the plateau's cause was isolated:
white-box logit-KD on the SAME 3,192 (skew-fwd-KL top-16 + CE, β0.5, r64) landed FLAT — 0.627 ≈ SFT
0.632.** Wringing the corpus two ways (hard-token SFT + full-distribution KD) hit the same ceiling →
**the plateau is the DATA (diversity), not the method.** Exactly the "bigger same-pipeline teacher =
limited-gain" regime (Rethinking-OPD 2604.13016). **Only lever left = new verified data → the env
factory ([[project_mythxengine_bridge]]).** Off-policy GKD / higher LoRA rank remain untried levers
but the data-ceiling makes new-diverse-data the higher-EV pivot.

    arm            claw    FC     custom   N      data
    baseline       0.642   87%    44%      —      (control)
    arm-B          0.618   78%    41%      435    Ornith τ-bench verified (reward=1.0)          ← BEST
    arm-A'         0.577   76%    36%      435    mixed public (matched to arm-B N)
    clean          0.562   74%    31%      10k    public, curated to strong agentic sources
    arm-A          0.557   76%    34%      10k    public kitchen-sink (mixed, incl. abstention+instruct)
    arm-B (scaled) 0.638   82%    42%      1314   Ornith τ-bench verified, 3× data (435+853 temp0.7+26 airline)  ← SCALE CLOSES GAP TO PARITY
    arm-B v2       0.632*  —      —        3192   +tau2 telecom/retail (2 epochs)  ← ×3-CONFIRMED LIFT: v2 0.632 vs base 0.604 (+0.028)
    arm-B v2 KD    0.627   —      —        3192   SAME data + white-box logit-KL (skew-fwd-KL top16, β0.5, r64)  ← FLAT vs SFT = DATA-CEILING
    arm-D          0.335   24%    6%       10k    STRONG-teacher PURE tool-use (smolagents+xlam Qwen3-32B + OS-Genesis GUI)  ← WORST

## Findings (all controlled)

1. **Public data DEGRADES the strong base**, and curating sources doesn't fix it: kitchen-sink
   (0.557) ≈ curated (0.562). It's public data *as a class* vs this base's own tuning, not the
   source mix. More public didn't help either — arm-A' 435 (0.577) ≈ arm-A 10k (0.557).
2. **Teacher-consistent + verified wins at fixed N**: arm-B 0.618 vs arm-A' 0.577 = **+0.041 claw**
   (and +2 FC, +4.6 custom) on the *same* 435 examples, same recipe. Provenance + deterministic
   verification is the lever — exactly the thesis. 435 Ornith trajectories beat a 10k public mix.
3. **SCALE closes the gap — the base is NOT hard-saturated.** Ornith 435→1314 (3×) moved claw
   0.618→0.638 (gap to base −0.024→−0.004, parity), monotonically, with FC 78→82% and custom held
   — zero degradation. The earlier "nothing lifts it" was a data-*scale* limit, not a ceiling. Best
   data + balance scales gracefully; a net LIFT is in reach with more verified data.
4. **Composition/distribution-match DOMINATES teacher quality.** arm-D used a *stronger* teacher
   (Qwen3-32B) than arm-A's weak mix — and cratered WORST (0.335 / FC 24% / custom 6%). Cause:
   the data was **pure tool-use** (smolagents ReAct + xlam FC) + off-distribution GUI actions, with
   **no "when-NOT-to-call" signal** → the model became pathologically tool-call-happy. Smoking gun:
   on a plain "summarize this sentence" prompt with **no tools**, arm-D emitted
   `<tool_call>{"name":"summarize",...}</tool_call>` instead of an answer. So general tasks (custom)
   and complex FC collapse. **Lesson: a strong teacher can't rescue bad composition; balance +
   distribution-match (Ornith's natural tool-use-AND-direct-response mix) is what preserves the base.**

## Recipe finding (load-bearing, separate from data)

Full-sequence SFT on multi-turn agentic data **actively degrades** the base (arm-A v1: claw 0.419).
Cause: the model learns to imitate the *user/tool* turns too. Fix = **assistant-masked loss**
(train only on assistant tokens) → recovered to 0.557. Qwen3.5's chat template has **no
`{% generation %}` block**, so TRL's `assistant_only_loss` silently masks *everything*; we do
**explicit delimiter-based masking** (`train_lora.py`) — label only tokens between each
`<|im_start|>assistant\n` header and its `<|im_end|>`. Verified: ~32% of tokens get loss, decoded
spans are exactly assistant content. **The recipe must be right before any data comparison means
anything** — otherwise every arm regresses identically.

## Pipeline (reusable, all validated end-to-end)

- **Data factory** (`dataset/`): canonical Trajectory schema, per-source adapters, dedup
  (content-hash, not first-turn — env-traj datasets share boilerplate openers), contamination
  filter, versioned builds. HF cache → /mnt/scratch (models drive is 99% full).
- **Verified-rollout harness**: τ-bench (MIT, deterministic DB-state reward) driven by local
  Ornith via `OPENAI_API_BASE`; `filter_tau.py` keeps reward=1.0 → canonical. Ornith scores
  **~87% on τ-bench retail** (strong agent). 500 retail-train tasks → 435 verified trajectories.
- **Trainer** (`train_lora.py`): TRL/peft, explicit assistant-masking, FLA-accelerated GDN
  (`flash-linear-attention`; ~8s/step; causal-conv1d skipped — marginal + sm120 build risk).
- **Quant+gate**: `qwen35_2b_requant.py` (NVFP4, agentically lossless) → `run.sh` profile.

## Next
- **more local Ornith data** (retail ×2 @ temp0.7 + airline) → grow arm-B from 435 → ~1.5k;
  test whether scale tips near-harmless into a genuine lift.
- **arm-D** (strong-teacher public: smoltalk2 smolagents/xlam @ Qwen3-32B + OS-Genesis claw-ops,
  16.9k) — does *strong-teacher* public behave differently from arm-A's weak-teacher public?
- **arm-C** (blend Ornith-core + best public) once the above land → the mixing ratio.

Reusable for the 80B coder path ([[project_agentic_coding_rl]]): swap sources → SWE/coding,
swap τ-bench → code-exec sandbox, swap 2B trainer → QLoRA/FSDP 80B. ~70% of this carries over.

## arm-E — parametric-skills lite (modular vs monolithic, 2026-07-06)

Tested arXiv 2606.30015's thesis: 4 skill-specialist LoRAs (cancel/return/exchange/modify, ~100
Ornith τ-bench traj each) vs the monolithic arm-B, on held-out τ-bench retail-TEST (skill-oracle
routing; agent :8010, Ornith user-sim :8000 via a user.py api_base patch).

    skill      n   monolith  specialist
    cancel     6    0.333     0.500
    return    10    0.500     0.600
    exchange  13    0.462     0.308
    modify    16    0.250     0.188
    weighted  45    0.378     0.356     → MONOLITH edges it (within noise)

**Isolated per-skill adapters do NOT beat the monolith** (small samples, ~100 traj/specialist).
Doesn't refute the paper — isolates *why* it works: the paper's gain is the **hypernetwork's
cross-skill parameter sharing**, not modularity per se; our isolated specialists lose the shared
knowledge the monolith gets from all 435 traj together. Given monolithic already hits base parity
(scaled arm-B 0.638), **monolithic distillation is the pragmatic winner** vs building a hypernetwork.
