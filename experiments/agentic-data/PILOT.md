# Unfiltered best-data pilot — find the mixing ratio (spec, 2026-07-06)

**Question the pilot answers (the one the census can't):** for distilling Ornith-1.0-35B →
Qwen3.5-2B, does the **Ornith-generated verified core** beat the **best public agentic data**,
and does a **blend** beat both — at fixed data volume? That decides how we spend the budget.

Not a legal exercise (license is off the table — see `DATASETS.md`). Pure quality: foreign-model
trajectories dilute the teacher signal, so "best data" ≠ "most data" → measure the blend.

## Baseline to beat
Qwen3.5-2B-NVFP4, no distillation: **claw 0.642 / FC 87.0% / custom 44%** (`evals/baselines/README.md`).
Teacher ceiling: Ornith-35B **claw ~0.74**. Goal = move the 2B toward the ceiling; report the lift.

## Arms (fixed N = ~8k trajectories each — fair blend test, NOT a volume test)

    arm            data                                                         build.py --mix
    A public       APIGen-MT + orca + ToolACE + hermes + When2Call + AgentTraj  public
    B ornith-core  τ-bench env → Ornith rollouts → reward-filter (verified only) ornith
    C blend        50/50 A:B by trajectory count                                blend
    D +reason      C + OpenThoughts3/SYNTHETIC-1 reasoning fuel (~15%)           blend (openthoughts3 enabled)

Each arm: `build.py --mix <x> --cap-per-source <n>` → LoRA-SFT → NVFP4 → gate. Same N, same
LoRA config, same seed → the only variable is composition.

## Per-arm procedure
1. **Build** the arm's JSONL (`dataset/build.py`), normalized to canonical `Trajectory`.
2. **LoRA-SFT** Qwen3.5-2B (bf16 base) — high-rank LoRA (r=32–64), 2–3 epochs, ~8k traj.
   LLaMA-Factory config in `training/` or TRL SFTTrainer. Train the text backbone; freeze vision.
3. **Quantize** the merged model → NVFP4 (`experiments/quantize/qwen35_2b_requant.py`, ~15 min).
4. **Gate** (`evals/run.sh --local profile`, judge pinned `local`, 40960 ctx): claw / FC / custom
   vs the 0.642 baseline. Δclaw is the headline.

## Ornith τ-bench shard (feeds arms B, C, D) — the one GPU-serving step
- Install τ-bench (MIT env; retail/airline/telecom, deterministic DB-state verifier).
- Point its agent at a served Ornith replica (or a freed card); run its task set, capture full
  multi-turn traces.
- **Reward-filter:** keep only trajectories the env's DB-state check marks success (`verified=true`,
  `reward=1.0`); keep a sample of failures as negatives (tagged, not trained-on in v0.1).
- Write canonical rows to `/mnt/data/datasets/agentic-distill/_raw/ornith_tau.jsonl` → `build.py`
  ingests via the `ornith_tau` adapter. Reuse the agentic-coding-rl reward plumbing.

## GPU budget
- Ornith τ-bench gen: ~1–2 h (serving; can piggyback the live replicas at low concurrency).
- LoRA-SFT a 2B on ~8k traj: ~1–2 h/arm (est., unmeasured). 4 arms ≈ half a day sequential.
- Quant+gate: ~1 h/arm. **Whole pilot ≈ one GPU day**, prod single-replica during SFT windows.

## Decision rule
- **B ≥ A** → Ornith-core is the spine; scale τ-bench + ASTRA/AgentGym generation, public data is garnish.
- **C > max(A,B)** → blend wins; the C ratio (start 50/50, then sweep 70/30, 30/70) sets the recipe.
- **D > C** → reasoning fuel earns a standing ~15% slot.
- Flat/negative on all → the task set is too narrow; widen environments before spending a full run.

## Held-out eval (never trained on)
τ²-bench (deterministic DB-state) is the honesty gate + the contamination anchor: `build.py` drops
any train row whose `prompt_hash` collides with a τ²-bench prompt. Report the distilled 2B on
τ²-bench alongside claw so the number is a real generalization signal, not a train-set echo.

## Storage
Corpus primary: `/mnt/data/datasets/agentic-distill/<version>/` (datasets drive). Optional private
mirror: `hf upload protoLabsAI/agentic-distill --repo-type dataset --private` (WAN ~100 Mbps, so
local is the source of truth; HF is backup/portability). The corpus is student-agnostic — reusable
for the 0.8B, future 9B distills, or any student.
