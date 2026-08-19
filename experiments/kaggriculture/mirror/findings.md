# Findings

## What the task actually is

Kaggriculture is a deterministic-policy and adversarial-market problem, not a modeling dataset. The meaningful offline metric is seeded, slot-swapped head-to-head win rate against a diverse frozen policy pool. Average coins against passive agents is useful for debugging but does not match the Bradley–Terry leaderboard objective.

## Engine audit

Experiments are pinned to `kaggle-environments==1.32.3`; the installed `kaggriculture.py` SHA-256 is `2f5f94e3da0f007f6d7628e30889bd19c83716183eeaa05b4922430db5021737`.

Observed source-versus-documentation differences:

1. Organizer prose says a fed-and-cared animal banks +2 care bonus; the installed engine increments `pending_care_bonus` by +1.
2. Organizer prose describes fertilizer as non-sellable, but the pinned engine includes `FERTILIZER` in `PRODUCTS` and accepts it in `SELL` orders.
3. Live discussions report additional mechanics discrepancies and a recently rolling hired-hand fix. The server version may change during the competition.
4. Raw CLI page content leaves runtime resource tokens unresolved, while the rendered page currently fills them with concrete values.

These are version risks, not invitations to access hidden state. The policy does not depend on the CARE discrepancy or other questionable exploits.

## Strategy findings

- Cheap early Fibonacci hires have enormous action value, but hiring ten every day costs 143 coins/day. A first baseline exhausted cash before its first melon harvest and lost crops. Workload-scaled crews plus a labor cash reserve improved the starter mean from roughly 21k to about 40.5k in the 25-seed pre-submission gate.
- Melon dominates the initial static economics, but its convex glut curve makes monoculture fragile. Crop scoring therefore projects visible self/opponent output through the published market curve and shifts toward staples when premium supply is crowded.
- Simultaneous PLANT orders are all canceled for a crop if requested plants exceed owned seeds. The scheduler creates no more plant tasks than the private seed count.
- The shed holds only 100 items. A 20-tile max melon harvest creates 120 units, so naive end-of-day dropping loses value. The policy batches carrier returns late in the day and sells shed contents on the next observable turn.
- Immediate liquidation is safe at the terminal boundary and funds reinvestment, but analytic tranching around town demand remains a major unimplemented opportunity.
- Plants bought too late cannot mature. Crop scoring removes candidates whose maturity is not strictly shorter than the remaining season.

## Current limitations

- Greedy Manhattan assignment lacks explicit deadline/slack matching.
- Market value uses a current/projected quote average rather than exact marginal revenue across the curve.
- Opponent modeling uses visible crop counts but not planting ages, forecast harvest windows, inferred shed stock, or observed bank deltas.
- No animal/fertilizer branch has passed an ablation gate.
- The local opponent pool is still shallow; built-ins are plumbing baselines, not competitive proxies.
- Seed purchases are capacity-based rather than throughput-based. The first three public episodes ended with $1,120–$2,330 of seed capital stranded at zero terminal value.
- The original local diagnostic excluded seeds from terminal inventory and overstated liquidation quality. Promotion reports must include terminal seed cost separately.
- Broad mixed-livestock and naive seed-cap candidates both failed paired v1 gates. Replay economics support testing a bounded sheep-only module next rather than adding all animal types simultaneously.

## Pre-submission validation

Agent SHA-256: `04870c4342d289992cea2a5e3085588708852686eb720a090249ae2a705a3e24`.

| Opponent | Seeds | Wins | Ties | Mean agent bank | Mean opponent bank | Preventable weeds | Zero-cash days | Terminal unsold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| starter | 25 | 25 | 0 | 40,499.20 | 3,497.08 | 0 | 0 | 0 |
| random | 25 | 25 | 0 | 40,118.68 | 0.00 | 0 | 0 | 0 |
| mirror | 10 | 6 | 0 | 51,347.30 | 49,597.00 | 0 | 0 | 0 |

Seat order alternated every game. The mirror result is not a strength estimate—the environment applies independent weed RNG sequences to the two farms—but it verifies valid adversarial market interaction and terminal liquidation. Replayed-action timing over one 719-call episode was 0.092 ms mean, 0.320 ms p95, and 0.420 ms maximum on the local machine.
