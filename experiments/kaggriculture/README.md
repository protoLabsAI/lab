# kaggriculture — Kaggle Featured Simulation ($50k, deadline 2026-09-30)

Two-player turn-based farming sim: 720 turns (30 days × 24), most coins wins.
Elo-style ladder, final Bradley-Terry tournament. Top-10 pays $5k each.
https://www.kaggle.com/competitions/kaggriculture

## Status (2026-08-04)

`agents/v4.py` — day-route planner, best local agent:

| matchup | result |
|---|---|
| vs `starter` | 10-0, mean $61k (starter $3.5k) |
| vs `pass` (solo econ) | $54-85k depending on seed |
| vs `opponents/sey_v7.py` (public v18 expert-replay, ~LB top-10 proxy) | 0-10 — they hold ~$170k |
| mirror v4 vs v4 | clean, no crashes, ~$57-61k each |

**Gap to close: ~2.8×.** The public leader replays embedded 719-action expert
schedules (Kaito Fukami "v18 closed loop", Apache-2.0, mirrors top players'
public episodes) reaching ~$170-190k. Same tile counts as us — the gap is
yield/tile (their fertilize+care discipline) and sell quality, not scale.

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
