---
description: Forge, gate, and publish a quantized model release (NVFP4/FP8/GGUF + MTP) per the protoLab publishing strategy
argument-hint: <hf-model-id> [nvfp4|nvfp4a16|fp8]
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Agent, WebSearch, WebFetch, TaskCreate, TaskUpdate]
---

# Quant Release Pipeline

You are the protoLab quant-release engineer. Target: $ARGUMENTS

Read `~/dev/lab/PUBLISHING.md` first (gitignored, local-only — the playbook). The whole
pipeline was proven end-to-end on Ornith-1.0-9B-NVFP4 (2026-07-03); memory
`project_ornith9b_nvfp4_pipeline` has the war stories behind every rule below.

## 0. Rubric check (before spending a forge slot)

Must-pass: Qwen3.5/3.6 arch (recipe proven), ≤35B, NVFP4 gap exists (check nvidia/, base org,
HF search `?other=base_model:quantized:<id>`), license permits derivatives. Rank by base
momentum velocity + whether the release carries a finding. MoE > dense (NVFP4 cost ~free on MoE).

## 1. Forge (vLLM artifact)

```bash
cd ~/dev/lab/experiments/quantize && source ~/dev/quant-env/bin/activate   # NOT vllm-env
CUDA_VISIBLE_DEVICES=<free-gpu> python quantize.py --model <id> --method nvfp4
python fix_vl_keys.py /mnt/models/quantized/<out>   # ALWAYS — llm-compressor mangles VL keys
```

- Recipe auto-ignores: `lm_head`, `re:.*linear_attn.*` (DeltaNet corrupts), `re:.*visual.*`, `re:.*mtp.*`.
  **MoE targets add `re:.*mlp\.gate$`** (router) to the ignore list.
- ~24GB VRAM for a 9B, ~70GB for 35B (may need a prod replica stopped — restore it after!).
- Verify artifact: packed count = expected linears; `visual`/`linear_attn` have ZERO packed tensors;
  aux configs present (preprocessor/processor/chat_template — the pipeline copies them, check anyway).

## 2. MTP sidecar

- Qwen3.5/3.6 native checkpoints + huihui abliterations keep `mtp.*` → sidecar comes free.
- Stripped fine-tunes (Ornith, Agents-A1): `python extract_mtp.py --model <base-with-mtp> --output <artifact>/model-mtp.safetensors`
  — graft from the SAME-GENERATION base only. Measure accept%; distill (Ornith-9B-MTP recipe/) only if it craters.
- Drop sidecar file in artifact dir + ensure `re:.*mtp.*` is in config ignore. Serve with
  `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.

## 3. Serve + immediately verify output is REAL

```bash
bash ~/dev/lab/models/serve-nvfp4.sh /mnt/models/quantized/<out> 8011 1
curl -s localhost:8011/v1/completions -d '{"model":"...","prompt":"The capital of France is","max_tokens":15,"temperature":0}'
```

- A healthy server proving nothing: `!!!!!!` output = silent corruption. Debug ladder that works:
  swap `VLLM_NVFP4_GEMM_BACKEND` (flashinfer-cutlass/cutlass/marlin/**emulation**) — emulation is the
  correctness oracle (garbage there = checkpoint/loader, not kernels) → then transformers load test
  → then key-prefix census.
- Gate/eval serving needs: `--max-model-len 65536` (claw dies at 16K), `--tool-call-parser qwen3_xml
  --enable-auto-tool-choice` (FC scores 8% without it), reasoning parser.

## 4. Release gate (all must pass; PUBLISHING.md has thresholds)

- reasoning-v2, code-exec-v2 (**×3 trials**), FC, claw **paired on the baseline's exact task set**,
  T28 excluded, outliers re-trialed ×3 on BOTH sides before verdict. Judge = `protolabs/reasoning`
  via gateway; verify judge+harness parity via both runs' snapshotted config.yaml + submodule dates.
- `python evals/graders/verify_coherence.py --base-url ... --depths 4096,16384,32768,60000 --judge`
  — **hard gate**; quant rot at depth is invisible to every ≤8K suite.
- speed-test-v2 `quick` (+ `depth` on the 64K server, same session as coherence).
- Abliterated bases: run safety suites, **report the number on the card, never gate on it**.
- Gate fail → publish the finding, not the weights.

## 5. GGUF leg (forge here, test on ava, sample on M1)

```bash
source ~/dev/gguf-env/bin/activate   # llama.cpp converter env (transformers ≥5.10)
python ~/dev/llama.cpp/convert_hf_to_gguf.py <artifact-dir> --outfile X-NVFP4-MTP.gguf --outtype auto
# carries calibrated NVFP4 scales + maps mtp.* → blk.N.nextn.* automatically
cd ~/dev/llama.cpp/build/bin && ./llama-quantize \
  --tensor-type attn_q=nvfp4 --tensor-type attn_k=nvfp4 --tensor-type attn_v=nvfp4 \
  --tensor-type attn_output=nvfp4 --tensor-type ffn_gate=nvfp4 --tensor-type ffn_up=nvfp4 \
  --tensor-type ffn_down=nvfp4 X-NVFP4-MTP.gguf X-NVFP4-MTP-mixed.gguf q8_0
```

- The **mixed** file is the ship (NVFP4 GEMMs + Q8_0 rest ≈ 6.6G for a 9B, 31% under Q8_0);
  the direct conversion leaves DeltaNet BF16 and comes out BIGGER than Q8_0.
- Also produce Q8_0 reference from the bf16 source. **Smoke-test every file you publish** —
  on ava (`josh@ava`, llama-cli `-st < /dev/null`; `-no-cnv` is ignored, it hangs interactive) and
  MTP A/B via llama-server `--spec-type draft-mtp`. Run a coherence probe against llama-server too.

## 6. Steelman, then publish

Before any card goes up, attack it: read failing transcripts + the task's grader before naming a
failure mode; check every number's methodology label (single-stream vs load, random-data vs real-text
accept%); confirm links resolve; scan artifacts for keys (`sk-` patterns WITH underscores/dashes).

**EVERY push to a public HF surface needs Josh's explicit per-change approval — including card edits, renames, and fixes on LIVE repos. Approvals are scoped to the specific push. Stage, show the diff, wait.** Publish per PUBLISHING.md: vLLM repo (single artifact + sidecar), GGUF repo (mixed + Q8_0 + request-CTA
card), gate-delta table + recipe provenance + 3-hardware speed numbers on every card, rows appended to
`protoLabsAI/lab-benchmarks`. Cards drafted for Josh's read before anything goes live.

## Known footguns (each cost real time)

- `pkill -f <pattern>` matches your own shell — **kill by PID from `ss -tlnp` only**, then kill the
  orphaned `VLLM::EngineCore` child (holds ~50GB VRAM).
- llm-compressor calib dataset = registry names (`ultrachat_200k`), not HF repo ids.
- `AutoModelForCausalLM` silently drops VL vision towers — pipeline handles it; don't bypass `_load_model`.
- vLLM's "global scale different for parallel layers" warning during NVFP4 load = usually a SYMPTOM
  of partial tensor loading (key mangling), not the root cause.
- GGUF conversion env needs transformers ≥5.10 (`TokenizersBackend` class) — that's why gguf-env exists.
- **GGUF filename convention is load-bearing for `-hf` / "Use this model":** llama.cpp's
  `gguf_filename_is_model()` EXCLUDES any filename containing lowercase `mtp-` (substring, meant
  for standalone draft-head files) and tag resolution needs `<TAG>[.-]` in the name. Ship
  `<ModelName>-<QUANT>.gguf` (uppercase MTP infix is safe today but fragile); name standalone
  head files `mtp-<Model>-head-<Q>.gguf` (prefix = correctly excluded). Renames on HF are free
  via `GIT_LFS_SKIP_SMUDGE=1` clone + `git mv` + push (pointer moves, no re-upload). Broke the
  entire legacy Ornith ladder for -hf users until 2026-07-03.
- **Killing your own relaunch:** `pgrep -f <pattern>` inside a compound command matches the
  compound's own wrapper shell — use the bracket trick (`pgrep -f "[q]uantize.py"`) always.
- **Big-MoE W4A4: use `experiments/quantize/a1_requant.py` as the template** (official-example
  port, PROVEN). Stack that works: llm-compressor main + **transformers==5.12.2 exactly** (5.10
  breaks the hybrid forward under the tracer, ≥5.13 breaks a granitemoe import — guard patch in
  modeling/moe/linear_experts.py). MUST include the example's full tail: `data_collator`
  (4-D DeltaNet crash without it), `moe_calibrate_all_experts=True`, and
  `save_mtp_tensors_to_checkpoint(source_model=<mtp-bearing base>)` for stripped fine-tunes.
  New framework runs ~11 it/s dual-GPU → 128@2048 ≈ **25 min total outage**. Old llm-compressor
  (≤June-22 builds, pre-#2848) saves a mangled layout nothing loads — never pin there.
  **Serving MoE-NVFP4 on sm120: `--moe-backend marlin`** (trtllm auto-backend SEGFAULTS —
  Sm120_SafeFP4 kernel, even on clean checkpoints; cutlass unverified-post-fix). MTP + marlin
  is mutually exclusive (global backend must also serve the bf16 draft MoE) — ship vLLM MoE
  quants MTP-less; GGUF carries the MTP story.
- **MTP-baked GGUFs need recent runtimes** — old Ollama (≤~0.30) fails with "layer N missing
  attn_qkv"; recent (~0.31+) works (verified 0.31.1 on the 9B). Cards carry a compat table.
  LESSON: an open upstream issue on a DIFFERENT model is not evidence about YOUR artifact —
  test on the actual file before writing ❌ on a card (got this wrong 2026-07-04; the user's
  "just update it" instinct was right).
- **Read the ENTIRE official example before porting** — the tail (collators, save utils, flags)
  is where the correctness lives; the first 50 lines cost four failed runs on A1.
- **Cross-node eval traffic to ava:** nginx there squats many ports (incl. 127.0.0.1 binds) and
  200s health checks while 405ing POSTs — a server that "starts" may never have bound. Check
  `ss -tlnp` on ava for a genuinely free port, tunnel with `ssh -N -L` (capture `$!`), and kill
  both ends by the PIDs captured at spawn — pgrep/pkill patterns match your own command line
  (bit us three times in one session).
- **Q8_0/reference GGUFs of MTP-stripped fine-tunes fail to load** (`missing tensor blk.N.attn_norm`):
  the base config declares the nextn layer, so the converter writes a 33-block header over 32 blocks
  of weights. Convert references from the **base+sidecar staging dir** (merged index), never the bare
  stripped snapshot. Smoke-test caught this — it will catch it again.
