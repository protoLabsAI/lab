# Challenger benchmark raw logs — 2026-06-27

Raw per-model logs behind the "Challengers" table in `../README.md`. Models served
off-gateway on GPU1:8005 (production untouched), same harness/judge as the baseline.

- `27b-mtp/` — Qwen3.6-27B-FP8 + MTP (prior smart lane)
- `gemma-fast/` — RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (prior fast lane)
- `ornith-9b/` — deepreinforce-ai/Ornith-1.0-9B bf16 (MTP-experiment gate)

Each: `claw.log` (30 non-coding), `ca-T10*.log` (5 coding-agentic sandbox),
`coding.log` (custom 10), `fc/` + `fc.log`, `speed.txt`. Single-trial — directional.
