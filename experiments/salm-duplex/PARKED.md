# PARKED — salm-duplex

Parked 2026-05-22. ORBIS retired; no consumer for a full-duplex voice agent. See [project_brand_pivot.md](file:///home/ava/.claude/projects/-home-ava-dev-lab/memory/project_brand_pivot.md).

## What this was

Full-duplex voice on Qwen3.5-4B via NVIDIA SALM channel fusion. Every 80 ms frame: user audio (FastConformer) + previous agent audio (NanoCodec) → sum → LLM emits text + 4 codec tokens → NanoCodec decodes. No VAD, no turn detection — model learns turn-taking from data.

## Where it stood (2026-04-15)

- Phase 0 ✅ Data prep — 283,861 samples + descriptions across LibriSpeech 960h, manifests at `/mnt/data/salm-duplex/manifests/full-960h-described/`
- Phase 1 ✅ Speech encoder + adapter
- Phase 2 ✅ Conversational speech decoder (DailyTalk, 2,541 dialogues, 21.7h)
- Phase 3 🟡 Conversational SFT — not started
- Phase 4/5 ⬜

## Disk state to preserve / reclaim

- `/mnt/data/salm-duplex/` — manifests, checkpoints (active dataset; preserve)
- `/mnt/pool/` — Emilia-YODAS EN 1.4TB, Switchboard 28GB, AMI 28GB, DailyTalk (redownloadable; reclaim if disk pressure hits)

## How to resume

Channel-fusion still looks like the right architecture for sub-300 ms duplex if a new consumer appears (a game with a live NPC, an experience surface). Resume order: Phase 3 (UltraChat Speech synthesis at scale) → Phase 4 (AMI/ICSI duplex SFT) → Phase 5 (tool calling). Phase 0–2 artifacts are reusable as-is.

Phase 0–2 alone is a breakdown candidate: *"What it takes to do channel-fusion duplex on a 4B model — data, encoders, the 80 ms tick."*
