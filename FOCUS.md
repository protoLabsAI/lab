# protoLab — Focus

**North star: the quant + serving lab.** We publish parity-verified FP8/quant models, we publish serving findings from the heavy rig, and we run a *trustworthy* eval harness to back both. Model class of interest: small / on-device-capable (≤ ~35B, especially ≤9B). The heavy rig (2× RTX PRO 6000 Blackwell) is the **forge**, not the inference target.

## What we do (and ship)

1. **Quant quality** — static FP8 (and friends) of models we actually serve, **parity-verified** against the source before publishing to [`protoLabsAI`](https://huggingface.co/protoLabsAI). Recipe + verification are the product, not just the weights. (e.g. `Ornith-1.0-35B-FP8`: block-wise FP8, SSM kept bf16, 92.9% truly-fp8, coding/FC parity.)
2. **Serving findings** — what actually makes models fast/correct on this hardware. (e.g. **replicas beat DP+EP/TP=2 on PCIe**; CUDA graphs on Blackwell; MoE quant traps; NCCL/PCIe.) Each is a reusable, reproducible finding.
3. **Trustworthy evals** — `evals/` as the open-source pattern. Reasoning models evaluated **thinking-on**; one coherent metric per suite; standing baselines; no silent failures.

## What we stopped (2026-06-27)

Archived to `/mnt/data/lab-archive/` (recoverable): the 13 brand-pivot side-bets (companion-stack, voice-agent, salm-duplex, rlm, agent-lightning, qwen3-omni, stt-whisper, tts-compare, image-gen-eval, flux2, pixel-gen, ltx-video, proto-bench) + diffusion side-bets (diffusiongemma, diffusion-cli-tools). Image/voice work lives on avaLab. **We stop: one-off "eval every new model" runs, the metric zoo, breadth for its own sake.**

## Eval suite — stabilization plan (Phase 2, in progress)

The current harness produces noise we reverse-engineer. Fixing, in priority order:

1. ~~**No silent failures.**~~ ✅ **FIXED (2026-06-27).** Root cause: the `X-Health-Check` probe POSTed an empty body to validating endpoints (`/kb/search`, `/contacts/search`) → FastAPI 422 → harness marked the service permanently unhealthy → every kb/contacts task (~10/run) silently failed. Fix: health probe short-circuits to a 200 liveness response (claw-eval `mock_services/_base.py`). Verified: kb/contacts tasks now run + score. **Next**: make the runner *report* harness-errored tasks distinctly from model-scored ones (so a run says "33 scored, 2 harness-errored", never a silent average).
2. **One metric per suite.** Kill the `passed` vs `task_score` vs `pass^3` ambiguity — pick one primary number (claw: mean task_score; FC: pass rate; coding: avg_score) and report it consistently.
3. ~~**Standing baselines.**~~ ✅ DONE (2026-06-27) — `evals/baselines/README.md` + first Ornith-35B-FP8 baseline recorded (claw **0.672** 35/35 clean / coding **0.925** / FC **93%**, reasoning judge). Re-run on every methodology change.
4. ~~**Consolidate runners.**~~ ✅ DONE (2026-06-27) — kept `claw`/`custom`/`function-call`/`rag` (+ `profile`/`compare`); archived `wildbench`/`refusal`/`inspect`/`general` to `/mnt/data/lab-archive/runners-2026-06-27/`. rag's judges already route through the fixed `LLMJudge` (4096).
5. ~~**Pinned judge.**~~ ✅ DECIDED (2026-06-27): **`protolabs/reasoning`** (independent cloud reasoning model via gateway) for **baselines**; local judge OK for everyday/relative runs. Never self-judge a baseline. Silent-0.5 fallback hardened in `llm_judge.py`.

## Eval suite — expansion plan (Phase 3, 2026-07-02)

Phase 2 made the numbers *trustworthy*; Phase 3 makes them *discriminating*. The 2026-06-30
Agents-A1 challenger run exposed the ceiling: reasoning 1.00, safety 1.00, code-exec-hard 0.97,
structured 0.90+, FC 91–93% — a purpose-trained agentic model **ties Ornith everywhere** because
the suite has no headroom left, and its claimed edge (long-horizon: GAIA/BrowseComp/MLE-class)
is an axis we don't measure at all. Only claw coding-agentic (0.27–0.68) and creative (0.69–0.80)
still discriminate.

**Acceptance rules (revised 2026-07-02 after calibration — see `evals/PHASE3_RESULTS.md`):**
1. **Discriminate across the ladder we ship (≤9B), not necessarily at 35B.** Calibration proved a
   35B thinking model saturates *every bounded single-turn suite we can author* (reasoning v2 mean
   0.962 even with 30-op register traces / 7-inhabitant knights; spec-delta coding ~0.99). The
   original "35B must be 0.3–0.7" is unachievable for bounded tasks without making them absurd.
   North star is on-device ≤9B — a suite that ceilings on 35B but spreads 4B<9B is doing its job.
2. **Struggle-zone-on-35B is reserved for the axes that actually bite it:** long-horizon multi-turn
   (T2xx) and exact multi-field computation. Those are where the 35B leaves headroom.
3. **Prefer multi-value partial-credit answers.** Single-scalar answers are all-or-nothing; the
   discriminators that emerged (arith two-count 0.50, schedule all_of 0.75, ledger 0.62) all have
   many independent ways to be wrong.
4. **Deterministic grading first:** execution/exact-match/solver-verified; LLM judge only where
   unavoidable (creative), never in a pass/fail gate.
5. **Contamination-resistant:** parametric seeded generators or spec-deltas of known problems —
   Ornith acing 6/6 canonical LeetCode-Hard means canonical = memorized. (Necessary, not sufficient:
   de-memorizing doesn't make a task hard for a 35B, it just makes the score *mean* something.)

**Suite work, priority order:**

| P | Suite | Shape |
|---|---|---|
| 0 | **long-horizon agentic** (new) | claw-style sandbox tasks needing 20–50 tool calls with a *verifiable end state* (files/DB/git assertions), multi-constraint, mid-task requirement changes. The Agents-A1 blind spot. |
| 0 | **reasoning v2** (5 → ~25) | exact-match / solver-verified answers, parametric generators (logic grids, scheduling, constraint puzzles) — kill the LLM judge here entirely. |
| 1 | **code-exec v2** | novel/adversarial: composed constraints, spec-deltas of known problems, property-based hidden tests. Reuse `graders/code_exec.py` + code-tree-search infra. |
| 1 | **FC-hard** | multi-step chains, proactive triggers, distractor tools, error recovery; fix the grader to credit legitimate intermediate calls (`current_time` class — real FC is ~96%, grader says 93%). |
| 2 | **safety-under-agency** (new) | T28-class secret-leak-under-tool-pressure + injection-in-tool-results — the suite says 1.00 while T28 fails on every model we've ever run. `experiments/agentworld/t28_redteam_fixtures.jsonl` is the seed. |
| 2 | **structured-hard** | composed schema constraints, deterministic validators only. |

**Construction method:** proposer–solver–verifier, same as Agents-A1's data pipeline pointed at eval
authoring — 122B/Ornith proposes task variants, execution/solver verifies ground truth, human
final-filters. Every accepted task doubles as substrate for the future verified-trajectory dataset
(`experiments/agentic-data/RESEARCH.md` play #1) — instrument graders to log per-step verifier
outcomes from day one.

**Status (2026-07-02):** built + 35B-calibrated — reasoning v2 (24 solver-verified, replaces v1),
code-exec v2 (8 spec-delta), structured-hard (6 composed-invariant + `json_validate` grader),
safety-under-agency (8 T28 vectors + `none_of` grader), long-horizon T201 (validated, 4 more designed).
FC grader fixed (grounding-call whitelist). **Central finding: bounded single-turn suites can't lower a
35B's ceiling regardless of rigor — only horizon + answer-multiplicity discriminate (reproduces the
Agents-A1 thesis from the eval side).** Full writeup: `evals/PHASE3_RESULTS.md`.

**Ladder re-baseline DONE (2026-07-02, `baselines/2026-07-02-phase3-ladder/`).** Stood down replica B,
ran 35B/9B/4B at a **fixed 8192-token thinking budget**. Reasoning discriminates monotonically
(0.882/0.726/0.615). **The lever that un-saturated the 35B was the thinking budget, not harder
problems** — same reasoning suite scored 0.962 unbounded vs 0.882 at 8K; code-exec 0.99 → 0.616.
That IS the horizon thesis: bound the budget (or extend the horizon) and capable models separate.
Suite verdicts: reasoning = primary ≤9B discriminator (keep); code-exec = big-vs-small (keep);
structured = near-flat, demote to regression floor; safety = flat at 1.0, rework under audit-framing.

**Remaining:** T201 Docker calibration + T200/202/203/204; safety v2 (audit-report framing); adopt
`--max-tokens 8192` as the standing eval budget. Replica B restored to production after the run.

## Current production (see [memory] / CLAUDE.md)

- Smart lane = **2× `Ornith-1.0-35B-FP8` replicas** (systemd `vllm` + `vllm-replica-b`), gateway round-robin (`least-busy`). Embeddings doubled + balanced.
- Replicas > sharding on PCIe — never TP/EP for a model that fits one card.
