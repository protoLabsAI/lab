# What 14 tiny LLMs actually know about being safe

*Draft — protolabs.studio. As of 2026-05-22.*

We took every tiny LLM we could load and threw 550 safety prompts at each of them. The capability cliff lives somewhere between 360M and 1B parameters. The 1B that beats the 1.7B that beats the 2B isn't who you'd expect. And the 4B that refuses four out of five *safe* prompts is the model the search results call "leading at this size."

## Setup

192 GB of Blackwell on a 14-model sweep. No cloud. Judge is Gemma 4 26B-A4B MoE FP8, running on the other GPU. All 14 models are tiny LLMs in the 2026 sense — sub-9B parameters, the band Google AI Edge calls "tiny" (TLM) below 1B and "small" (SLM) above. Reproduction recipe in [`tiny-models-bench/`](https://github.com/protoLabsAI/protoLab/tree/main/experiments/tiny-models-bench).

Two safety datasets:
- **XSTest** — 250 safe prompts that *sound* dangerous + 200 actually-harmful prompts. Both axes in one set.
- **simple_safety** — 100 prompts that should clearly be refused.

The judge classifies each response as comply / refuse / partial. The numbers below are the real responses across 7,700 model-prompt pairs.

## The safety cliff

```
            Model       Params  Refuses harmful  Complies harmful  Over-refuses safe
            ─────       ──────  ───────────────  ────────────────  ─────────────────
    smollm2-135m         135M             32%              61%              18%
    smollm2-360m         360M             46%              47%              22%
    qwen-3.5-0.8B        0.8B             87%               8%              16%
    gemma-3-1b             1B             90%               7%              11%
    llama-3.2-1B         1.2B             74%              18%               7%
    smollm2-1.7b         1.7B             69%              22%              12%
    qwen-3.5-2B            2B             93%               6%               9%
    gemma-4-E2B         2.3B*             89%              10%              20%
    llama-3.2-3B         3.2B             90%               9%               4%
    phi-4-mini           3.8B             97%               2%              14%
    gemma-3-4b             4B             85%               7%               5%
    qwen-3.5-4B-fp8        4B             99%               0%              79%   ← look at this
    gemma-4-E4B (fp8)   4.5B*             93%               6%              13%
    granite-4.1-8b (fp8)   8B             98%               1%               9%
```

\* effective params; loaded weights are higher per Google's E-series naming

Three things jump out.

### One — there is no safety below 360M

SmolLM2-135M complies with **61%** of harmful prompts. Specifically: "Illegal and Highly Regulated items", "Physical Harm and Violence", "Suicide, Self-Harm, and Eating Disorders" — name the category, the 135M cooperates. Not because it's malicious. Because it has no idea what "refuse" means. Refusing is a learned behavior. The training pass at 135M doesn't teach it.

360M halves the compliance rate to 47%. Still bad. The first model in the lineup that actually refuses a majority of harmful prompts is **Qwen3.5-0.8B at 87%**. Going from 360M to 0.8B is the cliff.

The audio-tags lesson applies here: at this scale, *training matters more than parameters*. A 360M model with safety training would beat any 1B without. Below 1B is where you stop getting safety for free.

### Two — Llama-3.2-3B is the calibration leader

Refuse-harmful is half the picture. The other half: **how often does the model refuse safe prompts?** Because "I refuse everything" gets you 100% on harmful-prompts but it's useless.

Llama-3.2-3B threads it: 90% refuse-harmful, **4% over-refusal**. The only model in the set that comes close to a frontier-model posture, at 3.2B parameters. Gemma-3-4B is right behind at 85% refuse / 5% over-refusal. Phi-4-Mini (the search-result darling for "best reasoning at this scale") is at 97/14% — better refusal than Llama 3.2 3B, but three times the false-positive rate on safe prompts. If you're building anything that has to *not* refuse user requests gratuitously, the Llama is the safer pick. Phi-4-Mini is more conservative.

The 1B-class data tells the same story. Llama-3.2-1B at 74/7% beats SmolLM2-1.7B at 69/12% on both axes, despite 0.5B fewer parameters. The Llama family's safety post-training is doing more work than the extra 0.5B.

### Three — Qwen3.5-4B-FP8 is the broken one

99.5% refuse-harmful is theoretically perfect. 78.8% **over-refusal** means it refuses 4 out of 5 *safe* prompts. The other Qwen3.5 sizes — 0.8B at 87/16, 2B at 93/9 — are normal. The 4B FP8 alone is in some entirely different regime.

Possibilities:
1. The on-the-fly FP8 quantization tipped the model into "refuse everything"
2. The 4B chat-tune specifically over-corrected on safety
3. Some interaction between the hybrid Mamba/attention architecture and FP8

We can disambiguate with a v2 run: same 4B in bf16, compare. Worth doing before anyone deploys this model expecting it to be useful.

The interaction matters because **on-the-fly FP8 is the deployment path most people will use**. If FP8 reliably breaks calibration at 4B, that's a constraint everyone needs to know about.

## What this *doesn't* tell us

We benched a six-suite custom capability set alongside the refusal data — coding, reasoning, structured_output, summarization, instruction_following, factual consistency. Every score came back as exactly 0.500. That's not real data. It's a [silent failure mode](../experiments/tiny-models-bench/RESULTS.md#custom-suite-data) in our LLM-judge implementation where any JSON-parse exception during scoring falls through to a 0.5 default — and the result schema didn't persist the error reasoning, so 6,300 task scores went uncalibrated without anyone noticing. The judge works fine in isolation. Under production load it appears to be truncating verbose responses mid-JSON.

We're publishing the refusal headline because refusal data is canonical (uses a different code path). The custom-capability cliff story has to wait for the judge fix and a re-run. The pattern that produces this kind of silent failure — exception-catching with a sane-looking default — is itself a finding worth sharing: **if your judge defaults to "I don't know" on failure, you cannot tell when it has failed.** Failure modes should raise, not default. We're patching it.

## Reproduction

All commands live in `experiments/tiny-models-bench/`. The 14-model loop:

```bash
for m in smollm2-135m smollm2-360m qwen-0.8b-base gemma-3-1b llama-3.2-1b \
         smollm2-1.7b qwen-2b-base gemma-4-e2b llama-3.2-3b phi-4-mini \
         gemma-3-4b qwen-4b-fp8 gemma-4-e4b-fp8 granite-4.1-8b; do
  bash run-bench.sh "$m" tier2
done
```

That's two scripts (`serve.sh` brings up a model on GPU 1 alongside the judge; `run-bench.sh` runs the suites). 14 hours of wall time across the cycle. Raw outputs land in `evals/results/tiny-bench/`.

Three failures worth naming, because they're real-world friction:

- **functiongemma-ft** — `litert-community/functiongemma-270m-ft-mobile-actions` is gated on HF. Needs a one-click accept that the org's six other Gemma repos didn't auto-extend to. The fine-tuned-vs-base comparison this would enable is the entire point of Google's FunctionGemma talk; we'll re-run when accepted.
- **olmoe** — AllenAI's tiny MoE (`OLMoE-1B-7B-0125-Instruct`) doesn't load on Blackwell. vLLM's olmoe.py rejects CUDA capability major=12: *"No supported CUDA architectures found for major versions [12]."* Blocked on upstream.
- **qwen-9b-fp8** — Mamba cache blocks (247) < `max_num_seqs` (256). One-line fix; would have unblocked the 9B headline row. Caught too late to retry tonight.

## What we're doing about it

Two things, neither of them "publish prematurely":

1. **Fix the judge.** Stop the silent 0.5 fallback. Bump max_tokens, surface parse errors, persist reasoning to disk. Re-run the 14-model loop to get the capability data. Update this post when we have it.
2. **The fine-tune is the next experiment.** Tiny on-device function calling is bottlenecked by training, not parameters. Google's [FunctionGemma 270M](https://huggingface.co/google/functiongemma-270m-it) is fine-tuned for function-calling from a 270M base and outperforms larger general-purpose models at the specific task. We can do this on the same rig — heavy iron forges, prosumer hardware deploys. That's the v2 post.

The brand commitment: *patterns to study and steal, not products to be used.* If you've got a tiny-model question that wasn't answered above, the raw data is on HF. If you ship one of these in production and find something we missed, [tell us](https://protolabs.studio).

## Credits

- claw-eval and the refusal pipeline live in [`protoLabsAI/protoLab/evals/`](https://github.com/protoLabsAI/protoLab/tree/main/evals)
- Model catalog and Blackwell findings: [`models/RESULTS.md`](https://github.com/protoLabsAI/protoLab/blob/main/models/RESULTS.md)
- HuggingFace datasets, model cards, raw eval JSONL: [`protoLabsAI`](https://huggingface.co/protoLabsAI)

---

*This post is v0.1 — refusal data only. v0.2 lands when the judge is fixed and the capability suite re-runs.*
