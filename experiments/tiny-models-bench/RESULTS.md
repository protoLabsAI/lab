# tiny-models-bench — RESULTS

As of 2026-05-22. Tier-2 scope: refusal (XSTest 450 + simple_safety 100) + 6 custom suites (coding, reasoning, structured_output, summarization, instruction_following, factual consistency). Single trial per custom task; refusal at full scale. Local judge throughout (Gemma 4 26B-A4B MoE FP8 on `protolabs/fast`). Zero cloud spend.

## Headline — refusal calibration (canonical data)

Refusal data is the cleanest read on what a tiny model has actually been trained to do. Two axes:
- **Under-refusal:** harmful prompts it should refuse but complies with
- **Over-refusal:** safe prompts it shouldn't refuse but does

| Model | Params | xstest refuse harmful | xstest comply harmful | xstest over-refuse safe | simple_safety refuse | simple_safety comply |
|---|---|---|---|---|---|---|
| **smollm2-135m** | 135M | 32.0% | 61.0% | 17.6% | 31.0% | 64.0% |
| **smollm2-360m** | 360M | 46.0% | 47.0% | 22.4% | 30.0% | 64.0% |
| **qwen-0.8b-base** | 0.8B (Qwen3.5) | 86.5% | 7.5% | 16.0% | 80.0% | 18.0% |
| **gemma-3-1b** | 1B | 89.5% | 7.0% | 10.8% | 90.0% | 8.0% |
| **llama-3.2-1b** | 1.2B | 73.5% | 18.0% | 6.8% | 78.0% | 17.0% |
| **smollm2-1.7b** | 1.7B | 68.5% | 22.0% | 11.6% | 62.0% | 33.0% |
| **qwen-2b-base** | 2B (Qwen3.5) | 92.5% | 5.5% | 8.8% | 87.0% | 10.0% |
| **gemma-4-e2b** | 2.3B eff | 88.5% | 10.0% | 20.0% | 95.0% | 5.0% |
| **llama-3.2-3b** | 3.2B | 89.5% | 9.0% | **4.4%** | 92.0% | 7.0% |
| **phi-4-mini** | 3.8B | 96.5% | 2.0% | 14.0% | 89.0% | 8.0% |
| **gemma-3-4b** | 4B | 85.0% | 6.5% | **4.8%** | 86.0% | 11.0% |
| **qwen-4b-fp8** | 4B FP8 (Qwen3.5) | 99.5% | 0.5% | **78.8%** | 98.0% | 1.0% |
| **gemma-4-e4b-fp8** | 4.5B eff (FP8) | 92.5% | 6.0% | 12.8% | 92.0% | 8.0% |
| **granite-4.1-8b** | 8B (FP8) | 97.5% | 1.0% | 9.2% | 88.0% | 8.0% |

Three findings stand out:

1. **The safety cliff lives between ~360M and ~1B.** SmolLM2-135M complies with 61% of harmful prompts. SmolLM2-360M with 47%. SmolLM2-1.7B drops to 22%. Llama-3.2-1B at the same scale drops to 18%. Below ~1B, "safety" effectively doesn't exist.
2. **Llama-3.2-3B is the calibration leader at this size class.** 4.4% over-refusal *and* 89.5% refuse-harmful — the only model that comes close to a well-calibrated frontier-model posture, at 3.2B params.
3. **Qwen3.5-4B-FP8 is broken on the over-refusal axis.** 78.8% — it refuses 4 out of 5 *safe* prompts. The other Qwen3.5 sizes are calibrated normally; the 4B FP8 variant alone is in this regime. Either the FP8 quantization tipped it, or the 4B chat-tune is specifically over-conservative. Worth a v2 investigation: re-run with bf16 vs FP8 side-by-side.

The over-refusal column maps cleanly to "this model knows when to refuse" vs "this model just refuses a lot." A model with 89% refuse-harmful + 4% over-refuse is calibrated; 89% refuse-harmful + 78% over-refuse is just paranoid.

## Custom-suite data — uncalibrated, do not cite

Every custom suite score in this run is exactly **0.500**, across all 14 models, all 9 sub-suites, all ~50 tasks per model — ~6,300 task scores with zero variance. That is **not** a real measurement. It is a silent failure mode in the LLM judge.

Root cause: `evals/graders/llm_judge.py:111-113` catches every exception during judge JSON parse and returns `score=0.5` with `reasoning="Judge error: ..."`. A manual probe of the judge with a realistic grading prompt works — returns proper JSON. The production trigger is suspected to be `max_tokens=500` truncating verbose Gemma 4 26B responses mid-JSON, but the silent fallback hides every failure.

Action: see [llm-judge silent-fallback memory](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/feedback_llm_judge_silent_fallback.md). Fix the judge to raise on parse failure (or persist the error reasoning so we see it), bump max_tokens, re-run.

## Failed loads (real findings, not bugs)

- **functiongemma-ft** (`litert-community/functiongemma-270m-ft-mobile-actions`) — gated HF repo. Needs one-click terms-accept on `artificial-citizen`. Worth running specifically: the fine-tuned-on-mobile-actions variant against the base `google/functiongemma-270m-it` is the talk's whole methodology demonstration.
- **olmoe** (`allenai/OLMoE-1B-7B-0125-Instruct`) — vLLM's OLMoE kernel rejects CUDA capability major=12: `RuntimeError: No supported CUDA architectures found for major versions [12]`. Blackwell sm_120 not supported in this vLLM version. Blocked upstream. AllenAI's tiny MoE is unreachable from this rig until vLLM's olmoe.py patches the architecture check.
- **qwen-9b-fp8** — mamba cache budget too small at util 0.18 (247 blocks vs `max_num_seqs=256`). One-line fix: drop max_num_seqs to 128 or bump util to 0.22. Worth re-running — Qwen3.5-9B is the upper bound of "tiny" and the headline-table-completeness model.

## Source of truth

- Aggregated machine-readable: `results.json` next to this file
- Per-model raw: `evals/results/tiny-bench/<model>/refusal_local-bench_*/summary.json` + `<model>/custom/custom_results.json`
- Raw run logs: `/mnt/scratch/logs/night1+2-tier2.log`, `/mnt/scratch/logs/night-followup-v3-tier2.log`
