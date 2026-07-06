# creative-writer — non-thinking house-voice content model

**Status:** scoping (2026-07-06). Research thread running externally (Josh); this
doc is the lab-side experiment scope. Nothing trained yet.

## Goal

A **non-thinking** creative writer for fast internal content — blog drafts, model/
dataset card copy, announcements — in the protoLabs house voice. Optimize for:
latency (no CoT), brand-voice fidelity, low slop, factual grounding on a supplied
brief. NOT a chatbot, NOT a fiction/RP model.

## Non-goals

- Fiction / roleplay / character-voice quality (the usual creative-finetune target).
- Thinking / reasoning traces. We never train a `<think>` block — keeping it
  genuinely non-thinking is the whole speed thesis. `enable_thinking:false` end to end.
- A general instruct regression. It must still follow a brief; guard with a merge-back.

## Base model

**Gemma-4-26B-A4B-it** is the pick, on our own evidence:
- `experiments/diffusion-creative-eval/` found AR Gemma 4 beats DiffusionGemma
  **64/36** on creative quality, with lower MMD + less slop.
- 3.8B active → fast; already queued for NVFP4 serving (see
  [[project_gemma4_nvfp4_mtp]] / `models/serve-gemma4-nvfp4.sh`). Train bf16,
  serve NVFP4 non-thinking.
- Fallback base if we want edgier latitude: an abliterated 9B (heretic/Qwythos
  lineage) — but for professional blog copy, stock Gemma is correct.

Open question to settle in Phase 0: 26B-A4B vs a 9B for the speed/quality knee.
Blog copy may not need 26B; a 9B non-thinking writer at 2-3x the throughput could
win on "fast content" if quality holds. Bake off both.

### Strong contender: gemma-4-12B-it (DENSE) — may be the better base
Added 2026-07-06 after finding DeepSeek's **DeepSpec** ([[reference_deepspec_dspark]]).
Three converging reasons the 12B dense may beat the 26B-A4B MoE as the *fine-tune* base:
1. **Dense = far easier/cheaper to fine-tune** (clean LoRA/full-FT; no MoE
   expert-routing/calibration headache). Big practical win for the training run.
2. **Off-the-shelf non-thinking spec-decode drafts** — DeepSeek released Eagle3 /
   DFlash / DSpark drafts for `google/gemma-4-12B-it`, trained in non-thinking mode
   on open-perfectblend. vLLM 0.22.1 serves eagle3 + dflash today (DSpark TBD).
   Fast single-stream decode with zero draft-training effort.
3. **Single-stream is our regime** → DFlash-class drafts are single-stream kings
   (their non-scaling under concurrency is irrelevant for content drafting).
Tradeoff to MEASURE: 12B active vs 26B-A4B's 3.8B active — raw the MoE decodes
faster, but a good draft may close/beat it single-stream. Base + all 3 drafts
staged to cache (2026-07-06). **Decisive bench:** serve gemma-4-12B-it + released
eagle3 (or dflash) draft, measure single-stream tok/s + accept, vs the validated
26B-A4B NVFP4+MTP **257 tok/s** ([[project_gemma4_nvfp4_mtp]]).

### 26B-A4B NVFP4 baseline is validated (the speed anchor)
185 tok/s no-MTP, **257 tok/s with MTP @ SPEC_TOK=2 (+39%)**, non-thinking clean.
Fast, but MoE = hard to fine-tune. It's the "keep as-is / prompt-only" option if
the 12B fine-tune doesn't clearly win on house-voice quality.

### Abliterated base? — refusals are a *separate axis* from prose quality
Abliteration buys **latitude (fewer refusals), not better writing**. Our own data
says so: Qwythos-9B (abliterated) is FC 94% but **NOT a creative standout**, and
carries a Safety 0.25 ([[project_lowbit_35b_vs_bf16_9b]]); heretic was un-retired
only for *uncensored prose* latitude, not quality ([[project_brand_pivot]] memory).

Decision: **don't default to abliterated.** For professional team blog copy the
false-refusal rate is likely near-zero, and abliteration risks a coherence/quality
hit plus a safety liability on a model whose output we *publish under the brand*.
Instead:
- **Measure it.** Add a *false-refusal rate on real briefs* metric to the Phase-0
  bake-off (stock Gemma vs abliterated Gemma vs 9B). Decide on data, not vibes.
- SFT+ORPO on non-refusing creative/house data **already suppresses refusals** as a
  side effect — often enough on its own.
- If measured false-refusals are still annoying, the cleaner lever than a fully
  abliterated base is **targeted refusal-direction ablation or a small "comply on
  benign creative" DPO set** on the stock base — removes the specific false-refusals
  without gutting safety wholesale.
- Keep an abliterated variant as a fallback base only if a genuine class of briefs
  (edgy marketing, dark themes) reliably trips stock Gemma.

## Data plan (three tiers)

### Tier 1 — house corpus (the differentiator, high-EV)
- Every published protolabs.studio post + our model/dataset cards + the
  `studio-brand` voice contract (`docs/reference/foundation.md` §3 filter).
- Owner: protoContent (coordinate export; see protoContent#368 schemas). Format as
  `(brief|outline|topic) → (finished post)` SFT pairs. Even 100-300 real posts is
  a usable style-transfer signal with augmentation.

### Tier 2 — anti-slop preference (public, proven)
- `jondurbin/gutenberg-dpo-v0.1` — chosen=human Gutenberg prose, rejected=model
  rewrite. The backbone anti-slop set.
- `nbeerbower/gutenberg2-dpo` (+ related Gutenberg variants) — more passages.
- `HuggingFaceH4/no_robots` (creative/writing slice) — clean human SFT.
- **Build our own slop-rejected pairs**: mine an over-represented phrase profile
  ("a testament to", "tapestry", "barely above a whisper", em-dash overuse, …)
  from model output, make DPO pairs where rejected = slop-laden. Cheap, high-impact.
- **[VERIFY LIVE]** 2026-specific creative/anti-slop sets that may supersede the
  above (search backend down at scope time — see Research-to-verify).

### Tier 3 — synthetic house-voice augmentation (bridge thin data)
- Generate `brief → post` pairs with the 122B (or frontier) **conditioned on** the
  brand contract + few-shot real posts; distill into the small writer.
- **HARD GATE:** our DiffusionGemma finding is that synthetic prose *launders slop
  and hallucinations*. Every synthetic sample passes (a) anti-slop DPO and (b) a
  fact-check pass before it enters SFT. Never straight in.

## Training recipe

1. **Sampler-side calibration first (zero training).** Sweep XTC / min-p / DRY /
   temp on the stock base against the eval harness. Establishes how much of the
   quality gap is *sampling* vs *weights* before spending a run. Likely a big free win.
2. **Style-transfer SFT** — Tier 1 + Tier 2 no_robots + fact-checked Tier 3.
   Completions only, thinking disabled. LoRA first (fits our cards; matches
   `experiments/agentic-data/train_lora.py` pattern), full-FT only if LoRA underfits voice.
3. **ORPO** with Gutenberg-DPO + our slop-rejected pairs. Folds preference into
   one pass, no separate reference model — the "sounds human, not assistant" step.
   (KTO as fallback if pair construction is a bottleneck — binary good/bad is cheaper.)
4. **Merge-back insurance** — mergekit SLERP/DARE-TIES of the finetune toward
   Gemma-4-26B-A4B-it to recover any instruction-following lost to aggressive
   style training. Standard creative-community move.

## Eval gate (reuse what we built)

- **Primary:** `experiments/diffusion-creative-eval/` harness — MMD / Token-L2 /
  slop-score + pairwise judge vs human refs (50 prompts). This is the training gate.
- **House-voice gate:** pairwise judge of `model post` vs a held-out *real team
  post* on the same brief; report win-rate + slop delta.
- **External comparability:** EQ-Bench Creative Writing (community standard).
- **Regression guard:** `evals/` instruction_following + a brief-adherence check —
  must not regress vs base (this is why the merge-back exists).
- Judge isolation: point the judge at a `local` replica, not the round-robin
  gateway ([[project_agents_a1_eval]] .env-clobber gotcha; [[feedback_llm_judge_silent_fallback]]).

## Serving

Non-thinking on NVFP4 via `models/serve-gemma4-nvfp4.sh` (MTP off for creative —
spec-decode gives no quality benefit and creative is latency-tolerant single-stream;
revisit if content volume makes it batch-bound). `enable_thinking:false`,
`--reasoning-parser gemma4` still set as belt-and-suspenders.

## Phases (research cycle)

| Phase | Exit criterion |
|---|---|
| 0 experiment | base bake-off — {Gemma-4-26B-A4B (MoE, NVFP4+MTP validated), **gemma-4-12B-it (dense, + DeepSpec draft)**, a 9B} × {stock, abliterated} scored on creative harness + false-refusal rate + single-stream speed (with draft) + sampler sweep; eval harness wired; data manifest exported from protoContent. Decide base on quality × trainability × single-stream speed |
| 1 train | SFT+ORPO run beats stock base on creative harness AND ties it on brief-adherence |
| 2 report | RESULTS.md: honest slop delta, house-voice win-rate, what didn't work |
| 3 engineering | wired into the content workflow (a `protolabs/writer` route or CLI) |
| 4 test | real briefs from the team, one round of live signal |
| 5 content | BLOG.md draft + (if we publish) HF model + dataset cards, Josh-approved |

## Risks / open questions

- **Thin house corpus.** May be too small for voice transfer without heavy Tier-3
  synthetic — which risks slop laundering. Mitigation: sampler-side + Gutenberg-DPO
  do a lot without needing much house data; synthetic gated hard.
- **Slop vs coherence tradeoff.** Aggressive anti-slop can hurt instruction-following.
  Merge-back + brief-adherence gate.
- **Non-thinking ceiling.** A no-CoT model may plan long-form structure worse. Test
  explicitly; if structure suffers, a *brief→outline→post* prompt scaffold (still
  no `<think>`) recovers it without reintroducing thinking latency.
- **26B may be overkill** for blog copy; 9B could be the better speed/quality point.

## Research-to-verify (search backend down at scope time — run when up)

- 2026 creative-writing / anti-slop datasets that supersede Gutenberg-DPO.
- Current EQ-Bench Creative Writing leaderboard — which open finetunes lead and their recipes.
- State of XTC / anti-slop samplers in our vLLM 0.22.1 (native support? flags?).
- ORPO vs KTO vs SimPO current consensus for style/creative.
- Whether any 2026 method beats the SFT→ORPO→merge pipeline for voice transfer.
