# Solution plan and debate synthesis

Codex and an independent OpenCode `kimi-for-coding/k3` review converged on a hierarchical deterministic planner before deep reinforcement learning. Kimi's verbatim invocation record and independent plan are in `research/kimi_k3_independent_plan.md`.

## Agreed architecture

1. **Day-level capital plan:** choose product mix, land, labor, and seed budget using remaining maturity windows, cash runway, town demand, market curves, and visible opponent production.
2. **Turn-level scheduler:** create watering/feeding deadlines, harvest/decay tasks, planting, weed, and logistics work; match workers by distance plus urgency.
3. **Market module:** compute exact per-unit marginal revenue, forecast concurrent supply, tranche nonterminal sales near demand ticks, and force liquidation before time expires.
4. **Opponent model:** infer harvest timing and likely hidden stock from public tiles and bank deltas; avoid crowded premium products or sell first.
5. **Safety layer:** central seed accounting, legal-action checks, ten-order cap, runtime fallback, and end-of-season inventory checks.

## Debate outcomes

- **Planning versus RL:** both reviews reject end-to-end RL as the first move. Exact mechanics and a large structured action space favor a transparent planner. CMA-ES or evolutionary tuning of a small weight vector is the next learning step only after the simulator harness is trustworthy.
- **Labor:** Kimi argued for marginal-value hiring; the first implementation used ten daily hands and empirically ran out of cash. The promoted implementation scales labor to active workload and reserves future hire costs.
- **Selling:** Kimi recommends exact tranches; the current policy still prioritizes reliable liquidation and reinvestment. Tranching is deferred until it beats immediate sale across a frozen pool.
- **Validation scale:** 10–20 episodes are sufficient only for syntax/plumbing and large baseline deltas. Promotion against competitive policies requires at least 200 quick-check episodes and 500+ seeded, slot-swapped games per important pairing.
- **Engine discrepancies:** code behavior is recorded and tested, but the core policy avoids relying on unsettled CARE or market-floor edge cases.

## Experiment ladder

| Stage | Candidate | Promotion gate |
|---|---|---|
| 0 | Valid PASS/wheat smoke bot | No errors; full 720-turn completion |
| 1 | Greedy crops + legal seed accounting | Essentially always beats pass/random; ≥95% vs starter |
| 2 | Cash-runway labor and land planner | Positive head-to-head lower confidence bound versus Stage 1 |
| 3 | Deadline/slack assignment and shed logistics | Zero preventable crop loss; positive pool win-rate delta |
| 4 | Exact market/tranche policy | ≥10% mean-money gain and nonnegative pool win rate |
| 5 | Opponent harvest/stock inference | ≥60% versus frozen Stage 4 variants without baseline regression |
| 6 | CMA-ES planner weights | ≥10% robust improvement or reject the learned layer |

## Frozen pool to build

- built-in pass, random, and starter;
- immutable prior promoted agents;
- wheat, carrot, melon, recurring-crop, and animal specialists;
- immediate seller, delayed seller, and market-floor pressure variants;
- high/low labor and expansion ablations;
- randomized but legal policy parameters.

Each report must record package/hash, agent hash, configuration, seed list, both player slots, wins/ties, money distribution, errors, weeds/escapes, discarded inventory, unsold terminal stock, and action-time percentiles.

## Submission cadence

1. Submit one locally valid plumbing candidate.
2. Confirm self-validation, server status, version behavior, episodes, and logs.
3. Keep at most one experimental and one champion bot active because only the latest two count.
4. Submit subsequent agents only after a frozen-pool promotion gate.
5. Before the deadline, re-run the full pool, verify the exact file hash, and ensure both final active slots are intentional.

