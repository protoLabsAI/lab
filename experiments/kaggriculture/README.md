# kaggriculture — Kaggle Featured Simulation ($50k, deadline 2026-09-30)

Two-player turn-based farming sim: 720 turns (30 days × 24), most coins wins.
Elo-style ladder, final Bradley-Terry tournament. Top-10 pays $5k each.
https://www.kaggle.com/competitions/kaggriculture

## Status (2026-08-04 end of day 1)

**ON THE LADDER**: 600 -> ~940 rating in one evening, 12-3 (80%) vs live field.
`agents/v10.py` = flagship (reference-curve planner + NN-reordered routes +
r138 optimizer config baked): **solo $162.9k/$169.3k records**, mirror clean.

| matchup | result |
|---|---|
| vs `pass` (solo) | $162.9k / $169.3k (records) |
| vs live ladder field | 12-3, wins mostly 2-4x |
| vs `opponents/sey_v7.py` (top-cluster proxy) | margin -$25.5k (from -$42k), 0-for-all |
| mirror | $122.6k / $121.1k clean |

**Margin plateau #2 at -25.5k.** Day-1 findings, in order of importance:
1. The ENTIRE top cluster (~3055 rating) runs ONE convergent build (8/10 top
   players byte-identical: 46 straw / 8 cow / 10 melon @ d15; sources in
   `replays/sources/`). Mutual top margins are $1-7k — execution-decided.
2. Our REF_CURVE matches their build; remaining gap = executor efficiency.
   NN route reorder freed moves 58%->51%; optimizer converted the freed
   capacity to farm scale (melon x1.5, sheep x1.3, 15 hands) = r138.
3. Exhausted (all tested, all at/below plateau): curve scales beyond r138,
   time-shifts, sell-policy knobs (race-always is optimal vs strong opps),
   wheat corner/warfare (they're market-makers, near-flat exposure), fert
   loop in v10 (no slack), goose/carrot pivots, radial dispatch, cluster
   placement.
4. Next real step: executor generation 3 — behavior-level imitation of the
   convergent build's unit actions, or exact-schedule execution with
   closed-loop repair (what the v18-family does). The margin is pure
   execution now.

Portfolio: `agents/v9.py` (adaptive fert loop, weak-field band) + `agents/v10.py`
(top band). Optimizer `optimize.py` runs continuously (`opt_log_run*.jsonl`);
ship automation via in-session cron at 00:08 UTC daily-reset.

## Architecture (v4)

- **Day-route planner**: at day start (and crew changes) build per-unit routes:
  placement chains (pickup→build→place animals), urgent singles (water/feed/
  harvest), collect→fertilize chains, plant+water compounds, rest.
- **Budget envelopes** each turn: hires → feed wheat → land → seeds → animals → sells.
- **Opening** (day 0): 5 hires, BUY_LAND (NE), 2 cows, 6 melon + 20 wheat seeds, 5 feed wheat.
- **Economy**: wheat engine early → strawberries (cap 45) + rolling melons (cap 12)
  + cows/sheep (cap 13, feed-budgeted) + fertilizer loop → liquidation day 28+.
- Sells: marginal-price aware (exact engine price curve mirrored), floor 0.3×base,
  feed-wheat reserve, full dump from day 28.

## Engine gotchas learned (hard-won)

- 3 of 4 shed-access tiles start LOCKED → PICKUP/DROP silently no-op there.
- New plants count as unwatered on plant day — water same day or weed overnight.
- kaggle_environments loads the **last** callable in the file as the agent.
- Hands' first spawn tile (5,4) is LOCKED (passable; tile-ops no-op).
- Feed/water miss ×2 consecutive days = animal escapes / plant→weed. Diag catches these.
- Fibonacci hire costs reset daily; 12 hands ≈ $376/day — cheap vs marginal product.
- Engine mirror: care banks +1 (docs said +2 in places); unfed production day still
  yields base 1. Trust `kaggriculture.py`, not prose.

## Files

- `agents/v1..v4.py` — agent iterations (v4 current; v1 simple crops baseline)
- `arena.py` — seeded seat-swapped parallel head-to-head
- `diag.py` — per-day loss telemetry (weeds, escapes, unwatered, idle, crew)
- `opponents/sey_v7.py` — public v18 expert-replay (sparring bar)
- `mirror/` — official competition docs + competitor findings mirrors
- `.venv/` — kaggle-environments 1.32.4 (engine at
  `.venv/lib/python3.12/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`)

## Next

1. Close yield gap: fertilizer coverage ratio (target ~every producing strawberry
   fertilized), care every animal every day, harvest-at-peak discipline. Use diag
   + compare day-curves vs expert replay (`opponents/sey_v7.py` vs pass).
2. Sell tranching around town demand ticks; don't crash premium curves.
3. Route efficiency: zone-stable assignment (moves are ~50% of unit-turns).
4. Study embedded expert schedules in sey_v7 (4 top players' episodes) for
   opening + mix timing; consider replay-guided opening for our closed loop.
5. Submit once rules accepted (manual: "Join Competition" on the site):
   copy agent to main.py, then
   `kaggle competitions submit kaggriculture -f main.py -m "v4 day-route planner"`.
6. Watch daily submit limit (5/day, latest 2 active).

## Workflow

```bash
cd ~/dev/lab/experiments/kaggriculture
.venv/bin/python diag.py agents/v4.py pass 7        # economy telemetry
.venv/bin/python arena.py agents/v4.py opponents/sey_v7.py --games 10 --procs 10
```
