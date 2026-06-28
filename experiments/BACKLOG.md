# Experiment backlog

Future-experiment ideas, queued but not started. Each entry: what, where it lives, why it might fit, what an experiment would look like. Order is not priority — see the active experiment dir (`tiny-models-bench/` as of 2026-05-22) for what's running now.

Add to this list when you find an idea worth holding; promote to an `experiments/<name>/PROPOSAL.md` when it graduates to next-up.

---

## 1. Lance — unified multimodal 3B (ByteDance)

- **HF:** [`bytedance-research/Lance`](https://huggingface.co/bytedance-research/Lance)
- **What:** Single 3B-active-param transformer that does text-to-image (768p), text-to-video (up to 121 frames @ 480p), image/video editing, and image/video VQA/captioning — all unified into one model. Trained from scratch on 128 A100s.
- **License:** Apache 2.0
- **Why it might fit:** The "one tiny model does everything multimodal" pattern is exactly the brand's substrate-#2-meets-experiences story. Brand piece writes itself: *"3B parameters that read, watch, generate, and edit. Here's where it cliffs."*
- **Substrate caveat:** Image and video generation moved to `avaLab` per the pivot (see [protoBanana handoff memory](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_protobanana_handoff.md)). This experiment likely belongs over there, not here. Worth cross-rig coordination if pursued.
- **Experiment shape (if run here):** capability-map at 3B unified vs single-task specialists at similar size — measure quality + tok/s + VRAM on each axis (gen / edit / understand × image / video). Pair with our existing FLUX / Qwen-Image-Edit numbers on ava for cross-substrate context.

## 2. Dramabox — expressive prompt-driven TTS (Resemble AI)

- **HF:** [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox)
- **What:** IC-LoRA fine-tune of LTX-2.3 3.3B audio-only DiT (flow-matching). Prompt itself controls speaker identity, emotion, delivery, laughs, sighs, breaths, pauses. Optional 10-second clip for voice cloning. ~2.5s per generation on H100.
- **License:** LTX-2 Community License (not Apache; restricted commercial)
- **Why it might fit:** TTS-with-prosody-from-prompt is the missing piece in any voice surface. The Fish S2 stack handles voice cloning but not the laughs/sighs/breath layer. Dramabox shows the IC-LoRA pattern over a flow-matching audio DiT — that's a methodology worth understanding even if the model itself isn't the long-term answer.
- **Substrate caveat:** Voice work was ORBIS-shaped (parked). Resurrecting it needs a new consumer — a game NPC, a tournament narrator, a `mythxengine-sdk` audio pack.
- **VRAM:** 24 GB inference; fits single-GPU on Blackwell easily. Slots into GPU 1 alongside the trimmed Gemma 4 26B judge.
- **Experiment shape:** A/B against Fish S2 Pro on the same script set with prompts-for-affect vs voice-clone-with-no-affect. Measure: human-judged expressiveness, latency, output coherence on multi-sentence affect changes.
- **Related parked work:** [voice-agent PARKED.md](voice-agent/PARKED.md), [fish-s2 tuning memory](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_fish_s2_tuning.md).

## 3. Qwen-AgentWorld — a world model as a synthetic agentic environment

- **HF:** [`Qwen/Qwen-AgentWorld-35B-A3B`](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B) (+ a 397B variant)
- **What:** NOT an agent — a **world model**. Qwen3.5-35B-A3B fine-tune (CPT→SFT→RL/GSPO) trained to **simulate agentic environments**: predict the next *state* of a Terminal, SWE repo, Web page, Android, OS, Search, MCP tool-call. AgentWorldBench scores the *simulation* (Format/Factuality/Consistency/Realism/Quality), not the acting. Same arch/size as our Ornith-1.0-35B daily driver (256 experts/8+1, hybrid DeltaNet+attention, 262K ctx); `--language-model-only` in vLLM.
- **License:** Apache 2.0. 34 community quants already exist.
- **Why it might fit:** It's the **environment**, not the agent — exactly the missing piece for two things we care about. (a) **Agentic eval at scale without real infra:** claw's coding-agentic (T100–104) needs real Docker sandboxes; a faithful simulator could mock Terminal/SWE/Web responses so we scale agentic eval cheaply. (b) **A substrate for agent RL** (the parked `agent-lightning`/DPO direction): RL on agents needs a fast, controllable env to roll out trajectories — this is built for that ("controllable perturbations / fictional worlds / zero-shot OOD"). Our FP8 / serving / MoE-spec-decode findings transfer (same base arch).
- **Not for:** the daily driver (it's a simulator, not an assistant), nor AgentWorldBench-as-our-metric (we care about Ornith *acting*, not world-modeling).
- **Experiment shape (cheap first probe):** point it at one claw task and diff its *simulated* terminal/SWE responses against a real sandbox run — does the simulation match closely enough to trust an eval built on it? That single fidelity check decides whether the synthetic-environment idea is real for our suite before any wiring.
- **Caveat:** specialist tool, real effort to adopt (wire simulator into eval/RL loop + validate fidelity); doesn't change the current stack. File under "next agentic-eval-at-scale or agent-RL push," not now.

---

## Promotion gate

Don't promote to an active experiment unless:
1. The current active experiment is past `engineering` (or parked with a memo)
2. There's a clear brand piece (a one-sentence blog headline that survives the audience filter)
3. The experiment can complete one `report` cycle within the next month of calendar time (no open-ended research)

Backlog is for ideas that pass the audience filter; failures of the filter belong in nobody's notes.
