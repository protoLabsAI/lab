#!/usr/bin/env python3
"""Diagnostic runner: play agent vs opponent, report per-day loss events."""
import sys
from collections import Counter

from kaggle_environments import make


def main():
    agent = sys.argv[1] if len(sys.argv) > 1 else "agents/v4.py"
    opp = sys.argv[2] if len(sys.argv) > 2 else "pass"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    env.run([agent, opp])
    steps = env.steps

    def snap(i):
        return steps[min(i, len(steps) - 1)][0].observation

    print(f"{'day':>3} {'money':>7} {'crew':>4} {'plants':>6} {'animals':>7} "
          f"{'newweed':>7} {'escaped':>7} {'unwat':>5} {'unfed':>5} {'idle':>4} "
          f"{'harv':>4} {'plant':>5} {'sellrev':>8}")

    prev_weeds = 0
    prev_animals = 0
    for d in range(30):
        i0, i1 = d * 24, min((d + 1) * 24 - 1, len(steps) - 1)
        obs_end = snap(i1)
        farm = obs_end["farms"][0]
        tiles = farm["tiles"]
        n_plants = n_weeds = n_anim = unwat = unfed = 0
        for row in tiles:
            for t in row:
                if isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        n_plants += 1
                        if not t["watered_today"]:
                            unwat += 1
                    elif k == "WEED":
                        n_weeds += 1
                    elif t.get("animal"):
                        n_anim += 1
                        if not t["fed_today"]:
                            unfed += 1
        acts = Counter()
        idle = 0
        sellrev = 0.0
        for i in range(i0, i1):
            if i >= len(steps) - 1:
                break
            st = steps[i][0]
            act = st.action or {}
            for a in [act.get("farmer")] + (act.get("hands") or []):
                if a:
                    acts[a[0]] += 1
                    if a[0] == "PASS":
                        idle += 1
        # realized sell revenue: SELL n at observed price that turn (approx)
        sellrev = 0
        sold = Counter()
        for i in range(i0, i1):
            if i >= len(steps) - 1:
                break
            st = steps[i][0]
            act = st.action or {}
            prices = st.observation["market"]["prices"]
            for m in act.get("market") or []:
                if m and m[0] == "SELL" and len(m) >= 3:
                    sellrev += prices.get(m[1], 0) * m[2]
                    sold[m[1]] += m[2]
        new_weeds = max(0, n_weeds - prev_weeds)
        escaped = max(0, prev_animals - n_anim)
        prev_weeds, prev_animals = n_weeds, n_anim
        crew = 1 + len(farm["hands"])
        print(f"{d:>3} {farm['money']:>7.0f} {crew:>4} {n_plants:>6} {n_anim:>7} "
              f"{new_weeds:>7} {escaped:>7} {unwat:>5} {unfed:>5} {idle:>4} "
              f"{acts['HARVEST']:>4} {acts['PLANT']:>5} {sellrev:>8} "
              f"F{acts['FERTILIZE']:>3} C{acts['CARE']:>3} X{acts['COLLECT_FERTILIZER']:>3} "
              f"{dict(sold.most_common(4))}")
    final = steps[-1]
    print("final:", [(i, s.reward, s.status) for i, s in enumerate(final)])


if __name__ == "__main__":
    main()
