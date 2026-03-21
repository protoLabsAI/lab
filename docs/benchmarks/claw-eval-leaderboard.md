# Claw-Eval Agent Benchmark Leaderboard

Tasks: T02 (email triage), T04 (calendar scheduling), T06 (email reply draft), T08 (todo management).
Metric: pass^3 (all 3 trials must pass, threshold 0.75).

## Local Models (with CUDA Graphs, March 2026)

| Rank | Model | T02 | T04 | T06 | T08 | Avg | pass^3 | tok/s | Config |
|------|-------|-----|-----|-----|-----|-----|--------|-------|--------|
| 1 | **35B MoE BF16 TP=2** | 0.87 | 0.53 | 0.85 | 0.86 | 0.78 | **3/4** | **170** | Both GPUs, 250K |
| 2 | **27B INT4** | 0.87 | 0.53 | 0.86 | 0.86 | 0.79 | **3/4** | 44 | 1 GPU, 160K |
| 3 | **122B INT4 1GPU** | 0.87 | 0.52 | 0.85 | 0.88 | 0.78 | **3/4** | ~30 | enforce-eager |
| 4 | OmniCoder 9B | 0.79 | 0.54 | 0.86 | 0.85 | 0.76 | 2/4 | 92 | 1 GPU, 262K |
| 5 | 35B MoE BF16 1GPU | 0.87 | 0.52 | 0.72 | 0.88 | 0.75 | 2/4 | 170 | 64K |
| 6 | Llama 70B AWQ | 0.71 | 0.52 | 0.58 | 0.79 | 0.65 | 1/4 | 38 | 128K |

## Cloud Models

| Rank | Model | T02 | T04 | T06 | T08 | Avg | pass^3 | Cost |
|------|-------|-----|-----|-----|-----|-----|--------|------|
| 1 | **GLM 5 Turbo** | 0.91 | 0.70 | 0.91 | 0.90 | 0.85 | **3/4** | $0.96/$3.20 |
| 2 | **Sonnet 4.6** | 0.89 | 0.71 | 0.89 | 0.91 | 0.85 | **3/4** | $3/$15 |
| 3 | **Opus 4.6** | 0.87 | 0.70 | 0.86 | 0.91 | 0.84 | **3/4** | $5/$25 |
| 4 | Sonnet 4.0 | 0.85 | 0.54 | 0.83 | 0.87 | 0.77 | 3/4 | $3/$15 |
| 5 | Kimi K2.5 | 0.76 | 0.64 | 0.89 | 0.88 | 0.79 | 2/4 | $0.45/$2.20 |
| 6 | DeepSeek V3.2 | 0.86 | 0.53 | 0.59 | 0.87 | 0.72 | 2/4 | $0.26/$0.38 |
| 7 | GPT-5.4 | 0.80 | 0.69 | 0.74 | 0.88 | 0.78 | 1/4 | $2.50/$15 |
| 8 | Haiku 4.5 | 0.60 | 0.70 | 0.86 | 0.82 | 0.75 | 1/4 | $1/$5 |
| 9 | MiniMax M2.5 | 0.76 | 0.59 | 0.69 | 0.89 | 0.73 | 1/4 | $0.20/$1.17 |
| 10 | GPT-OSS 120B | 0.78 | 0.56 | 0.57 | 0.84 | 0.69 | 1/4 | ~free |
| 11 | Gemini Pro | 0.82 | 0.55 | 0.53 | 0.70 | 0.65 | 1/4 | $3.50/$10.50 |
| 12 | GPT-4o-mini | 0.71 | 0.55 | 0.64 | 0.63 | 0.63 | 0/4 | $0.15/$0.60 |
| 13 | Gemini Flash | 0.20 | 0.21 | 0.28 | 0.88 | 0.39 | 1/4 | $0.50/$3 |

## Notes

- **Port 9100 conflict**: node-exporter occupies port 9100. Claw-eval Gmail mock defaults to 9100. Always use `--port-offset 200`.
- **T04 calendar scheduling**: No model passes pass^3. Opus/GLM/Haiku closest at 0.70.
- **Communication score**: Always 0.00 across all models — grader expects specific entity strings from tool responses.
- **INT4 on MoE**: Causes fluke 0.00 scores on some trials. Use BF16 for MoE models.
