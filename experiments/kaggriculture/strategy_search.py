#!/usr/bin/env python3
"""Strategy-space search for kaggriculture.

Evaluates PORTFOLIO GENOMES (what to produce and how much) rather than tuning
one fixed strategy. Every genome plays the same seeded, seat-swapped games
against the same opponent (common random numbers), so differences between
genomes are signal, not seed luck.

    ./strategy_search.py named          # evaluate the hand-designed candidates
    ./strategy_search.py evolve 40      # hill-climb from the best known genome
    ./strategy_search.py evolve 40 --opp opponents/sey_v7.py
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
N_GAMES = int(os.environ.get("SS_GAMES", "24"))
SEED0 = int(os.environ.get("SS_SEED0", "9000"))
PROCS = min(30, (os.cpu_count() or 8) - 2)
AGENT = os.path.join(HERE, "agents/v16.py")

# The convergent top-cluster build, as a genome (our control).
META = {"G_STRAW": 40, "G_MELON": 10, "G_WHEAT": 8, "G_CARROT": 0, "G_TOMATO": 0,
        "G_COW": 8, "G_SHEEP": 5, "G_GOOSE": 0, "G_QUADS": 3,
        "G_RAMP": 13, "G_ARAMP": 10}

NAMED = {
    "meta":         {},
    "model_opt":    {"G_COW": 31, "G_STRAW": 26, "G_MELON": 17, "G_SHEEP": 0, "G_WHEAT": 0},
    "cow_heavy":    {"G_COW": 20, "G_STRAW": 30, "G_MELON": 12, "G_SHEEP": 2},
    "cow_max":      {"G_COW": 28, "G_STRAW": 20, "G_MELON": 14, "G_SHEEP": 0, "G_WHEAT": 4},
    "goose_rush":   {"G_GOOSE": 18, "G_COW": 6, "G_STRAW": 24, "G_MELON": 10, "G_SHEEP": 0},
    "melon_heavy":  {"G_MELON": 26, "G_STRAW": 24, "G_COW": 10, "G_SHEEP": 3},
    "animal_stack": {"G_COW": 16, "G_SHEEP": 8, "G_GOOSE": 10, "G_STRAW": 18, "G_MELON": 8},
    "diversified":  {"G_COW": 14, "G_STRAW": 22, "G_MELON": 12, "G_GOOSE": 8,
                     "G_CARROT": 6, "G_TOMATO": 6, "G_SHEEP": 2},
    "four_quad":    {"G_COW": 20, "G_STRAW": 34, "G_MELON": 16, "G_SHEEP": 4, "G_QUADS": 4},
    "fast_ramp":    {"G_COW": 14, "G_STRAW": 32, "G_MELON": 12, "G_ARAMP": 6, "G_RAMP": 9},
}


def _play(args):
    cfg_path, seed, swap, opp = args
    os.environ["KAGG_CFG"] = cfg_path
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    pair = [opp, AGENT] if swap else [AGENT, opp]
    env.run(pair)
    r = [s.reward or 0.0 for s in env.steps[-1]]
    me, them = (r[1], r[0]) if swap else (r[0], r[1])
    return me - them, me, them


def evaluate(genome, tag, opp):
    cfg = dict(META)
    cfg.update(genome)
    path = os.path.join(HERE, f"sweep/ss_{tag}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    jobs = [(path, SEED0 + i, i % 2 == 1, opp) for i in range(N_GAMES)]
    with ProcessPoolExecutor(max_workers=PROCS) as ex:
        res = list(ex.map(_play, jobs))
    n = len(res)
    return {"margin": sum(r[0] for r in res) / n,
            "ours": sum(r[1] for r in res) / n,
            "theirs": sum(r[2] for r in res) / n,
            "wins": sum(1 for r in res if r[0] > 0), "n": n}


def show(rows):
    rows.sort(key=lambda r: -r[1]["margin"])
    print(f"\n{'strategy':<14}{'margin':>10}{'ours':>10}{'theirs':>10}{'wins':>8}")
    for tag, r in rows:
        print(f"{tag:<14}{r['margin']:>10,.0f}{r['ours']:>10,.0f}"
              f"{r['theirs']:>10,.0f}{r['wins']:>5}/{r['n']}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "named"
    opp = os.path.join(HERE, "agents/v15.py")
    if "--opp" in sys.argv:
        opp = os.path.join(HERE, sys.argv[sys.argv.index("--opp") + 1])
    os.makedirs(os.path.join(HERE, "sweep"), exist_ok=True)
    log = open(os.path.join(HERE, "strategy_log.jsonl"), "a")
    print(f"agent={os.path.basename(AGENT)} opp={os.path.basename(opp)} "
          f"games={N_GAMES} seeds={SEED0}+ procs={PROCS}")

    if mode == "named":
        rows = []
        for tag, g in NAMED.items():
            t0 = time.time()
            r = evaluate(g, tag, opp)
            rows.append((tag, r))
            log.write(json.dumps({"tag": tag, "genome": g, **r}) + "\n")
            log.flush()
            print(f"  {tag:<14} margin {r['margin']:>9,.0f}  ours {r['ours']:>8,.0f}"
                  f"  wins {r['wins']}/{r['n']}   ({time.time()-t0:.0f}s)", flush=True)
        show(rows)
        best = max(rows, key=lambda r: r[1]["margin"])
        json.dump({"tag": best[0], "genome": NAMED[best[0]], **best[1]},
                  open(os.path.join(HERE, "best_genome.json"), "w"), indent=1)
        return

    # evolve: hill-climb from the best genome on disk (or the meta)
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    bp = os.path.join(HERE, "best_genome.json")
    cur = json.load(open(bp))["genome"] if os.path.exists(bp) else {}
    base = evaluate(cur, "evo_base", opp)
    best_m, best_g = base["margin"], dict(cur)
    print(f"  base margin {best_m:,.0f} (wins {base['wins']}/{base['n']})", flush=True)
    rng = random.Random(7)
    BOUNDS = {"G_STRAW": (0, 50), "G_MELON": (0, 30), "G_WHEAT": (0, 20),
              "G_CARROT": (0, 20), "G_TOMATO": (0, 20), "G_COW": (0, 34),
              "G_SHEEP": (0, 20), "G_GOOSE": (0, 30), "G_QUADS": (2, 4),
              "G_RAMP": (7, 16), "G_ARAMP": (4, 14)}
    for i in range(iters):
        cand = dict(best_g)
        for k in rng.sample(list(BOUNDS), rng.randint(1, 3)):
            lo, hi = BOUNDS[k]
            base_v = cand.get(k, META[k])
            span = max(2, int((hi - lo) * 0.3))
            cand[k] = max(lo, min(hi, base_v + rng.randint(-span, span)))
        r = evaluate(cand, f"evo{i}", opp)
        ok = r["margin"] > best_m + 400
        log.write(json.dumps({"tag": f"evo{i}", "genome": cand, **r, "accept": ok}) + "\n")
        log.flush()
        print(f"  evo{i:<3} margin {r['margin']:>9,.0f}  wins {r['wins']}/{r['n']}"
              f"  {'ACCEPT' if ok else ''}", flush=True)
        if ok:
            best_m, best_g = r["margin"], cand
            json.dump({"tag": f"evo{i}", "genome": best_g, "margin": best_m},
                      open(bp, "w"), indent=1)
    print(f"\nbest margin {best_m:,.0f}\n  {best_g}")


if __name__ == "__main__":
    main()
