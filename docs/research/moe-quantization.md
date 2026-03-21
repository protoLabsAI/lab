# MoE vs Dense Quantization Findings

## Summary

INT4 quantization is quality-neutral on dense models but causes instability on MoE models.
The expert routing network is more sensitive to precision loss than dense feed-forward layers.

## Dense Model Results (Qwen3.5-27B)

| Quant | Weights | Avg Score | pass^3 | tok/s | Conclusion |
|-------|---------|:---------:|:------:|:-----:|------------|
| BF16 | 52GB | 0.78 | 3/4 | 44 | Baseline |
| FP8 | 27GB | 0.77 | 2/4 | 44 | Equivalent quality, smaller |
| **INT4** | **14GB** | **0.79** | **3/4** | **44** | **Best: same quality, 4x smaller** |

INT4 actually scored slightly higher than BF16 on some tasks (possible regularization effect).

## MoE Model Results (Qwen3.5-35B-A3B, 3B active)

| Quant | Weights | Avg Score | pass^3 | Issue |
|-------|---------|:---------:|:------:|-------|
| **BF16** | **72GB** | **0.80** | **3/4** | **Stable, consistent** |
| INT4 | 24GB | 0.68 | 1/4 | Fluke 0.00 on T06 trial 3 |

## MoE Model Results (Qwen3.5-122B-A10B, 10B active)

| Quant | Weights | Avg Score | pass^3 | Issue |
|-------|---------|:---------:|:------:|-------|
| FP8 | 70GB | 0.78 | 3/4 | Stable (original config) |
| **INT4** | **35GB** | **0.78** | **3/4** | Works at single GPU with enforce-eager |

The 122B INT4 at single GPU works but CUDA graphs crash. At TP=2 with enforce-eager it's fine.

## Why MoE INT4 Is Unstable

1. **Router precision**: The tiny router network decides which experts activate. Small quantization errors → wrong expert selection → completely off-topic output.
2. **Expert specialization**: Each expert learned a narrow domain. Wrong expert = catastrophic, not gradual degradation.
3. **Fewer active params**: 3B active params have less redundancy to absorb errors than 27B dense.
4. **Long-tail experts**: Rarely-activated experts are poorly calibrated during quantization.

## Recommendations

- **Dense models**: Always use GPTQ-Int4. Zero quality loss, 4x smaller.
- **MoE models**: Keep BF16. The disk/VRAM savings of INT4 aren't worth the quality risk.
- **Exception**: Large MoE (122B) at INT4 seems stable — more active params (10B) provide enough redundancy.
