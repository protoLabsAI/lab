# Challenger eval — InternScience/Agents-A1 (2026-06-30)

Agentic-tuned `qwen3_5_moe` VLM, **35B-A3B MoE** (same arch family as the Ornith daily
driver), 262K ctx, Apache-2.0. Served the **official `Agents-A1-FP8-dynamic`** (37.7 GB,
compressed-tensors: `linear_attn`/SSM + router + shared-expert + vision kept bf16, experts
FP8 → Triton fused-MoE, sm120-safe). Weights nuked after this run; re-pull from
`InternScience/Agents-A1-FP8-dynamic` to reproduce.

## Serve config

    GPU 1 :8003  --language-model-only  (VLM → text-only)
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder        # template uses <function=name><parameter=k>v
    --max-model-len 131072  --gpu-memory-utilization 0.72
    auto: CutlassFP8ScaledMM + TRITON Fp8 MoE + Triton/FLA GDN linear-attn
    speed: 208.7 decode tok/s, 24 ms TTFT   (= Ornith 35B-FP8)

## Methodology

    Suite     full profile (30 claw + all custom suites + function-call)
    Judge     Ornith replica A, raw vLLM local on :8000 (model under test on :8003)
    Trials    1 (single-trial; pass^3 dropped 2026-06-29)
    Caveat    judge pinned via evals/.env override (exported JUDGE_GATEWAY_URL is
              clobbered by run.sh --local sourcing .env); 3 transient judge
              fallbacks across the whole run (negligible)

## Headline numbers

    Dimension                       Score        Note
    ------------------------------  -----------  --------------------------------
    claw-eval (agentic, 30 tasks)   0.734 mean   22/30 pass@0.7  (Ornith 0.741)
    function-calling                0.91         49/54
    FC channel reliability          1.00         5/5
    tool reliability under load     0.88
    code: execution-graded (hard)   0.97         6 tasks
    code: analysis & review         0.95
    code: generation quality        0.85
    reasoning & logic               1.00
    structured output               0.90
    instruction following           0.85
    safety & boundaries             1.00
    summarization (news/tech)       0.97 / 0.84
    factual consistency             0.97
    research synthesis              0.94
    svg generation                  0.88
    creative: character & voice     0.69         weakest area
    creative: narrative craft       0.76
    creative: prose                 0.80
    rpg game master                 0.80
    alias routing fitness           0.80
    protolabs/vision                FAIL         expected (language-model-only)

## claw hard-fails (score 0.00)

    T06_email_reply_draft
    T12_expense_report
    T26_ambiguous_contact_email
    T28_api_config_audit        # fails on ALL our baselines, not A1-specific

## Verdict

Clean serve on Blackwell; strong agentic / tool / code; **ties Ornith, does not beat it**
(same base arch, ~same scores). No reason to displace the daily driver. Candidate agentic
policy/eval model for the game-rlvr / AgentWorld RL-substrate direction (PR #11 / #10).
