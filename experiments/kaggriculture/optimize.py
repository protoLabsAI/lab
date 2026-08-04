#!/usr/bin/env python3
"""Overnight knob optimizer: hill-climb v6 config on win margin vs sey_v7.

Common random numbers: every candidate plays the same (seed, seat) pairs.
Accept if mean margin improves. Logs to opt_log.jsonl, best to best_cfg.json.
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [(3000 + i, i % 2 == 1) for i in range(24)]
N_PROCS = min(24, (os.cpu_count() or 8) - 2)

SPACE = {  # name: (default, min, max, is_int)
    "MAX_HANDS": (14, 11, 16, True),
    "MAX_ANIMALS": (14, 10, 18, True),
    "ANIMAL_LAST_BUY": (16, 10, 20, True),
    "STRAW_CAP": (45, 32, 55, True),
    "LIQ_DAY": (28, 26, 28, True),
    "OPEN_HIRES": (4, 3, 6, True),
    "OPEN_COWS": (3, 1, 4, True),
    "OPEN_SHEEP": (1, 0, 3, True),
    "OPEN_MELON": (7, 5, 12, True),
    "OPEN_WHEAT": (10, 6, 20, True),
    "OPEN_FEED": (6, 4, 16, True),
    "MELON_ROLL": (8, 5, 14, True),
    "WOOL_FLOOR": (130, 90, 160, True),
    "MILK_FLOOR": (100, 70, 130, True),
    "SHEEP_CAP": (5, 3, 8, True),
    "ANIMAL_GATE": (250, 100, 900, True),
    "RUNWAY": (400, 200, 700, True),
    "BURST": (3, 2, 5, True),
}


def play(args):
    cfg_path, seed, swap = args
    os.environ["KAGG_CFG"] = cfg_path
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    a, b = os.path.join(HERE, "agents/v7.py"), os.path.join(HERE, "opponents/sey_v7.py")
    pair = [b, a] if swap else [a, b]
    env.run(pair)
    r = [s.reward or 0.0 for s in env.steps[-1]]
    me = r[1] if swap else r[0]
    opp = r[0] if swap else r[1]
    return me - opp, me, opp


def evaluate(cfg, tag):
    cfg_path = os.path.join(HERE, f"sweep/opt_{tag}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    jobs = [(cfg_path, s, sw) for s, sw in SEEDS]
    margins, mes, opps = [], [], []
    with ProcessPoolExecutor(max_workers=N_PROCS) as ex:
        for m, me, opp in ex.map(play, jobs):
            margins.append(m)
            mes.append(me)
            opps.append(opp)
    n = len(margins)
    return {
        "margin": sum(margins) / n,
        "our_mean": sum(mes) / n,
        "opp_mean": sum(opps) / n,
        "wins": sum(1 for m in margins if m > 0),
        "n": n,
    }


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    deadline = time.time() + hours * 3600
    rng = random.Random(1234)
    os.makedirs(os.path.join(HERE, "sweep"), exist_ok=True)
    log = open(os.path.join(HERE, "opt_log.jsonl"), "a")

    best_cfg = {k: v[0] for k, v in SPACE.items()}
    if os.path.exists(os.path.join(HERE, "sweep/combo.json")):
        best_cfg.update(json.load(open(os.path.join(HERE, "sweep/combo.json"))))
    res = evaluate(best_cfg, "base")
    best = res["margin"]
    print(f"baseline margin={best:.0f} our={res['our_mean']:.0f} wins={res['wins']}/{res['n']}", flush=True)
    log.write(json.dumps({"tag": "base", "cfg": best_cfg, **res}) + "\n")
    log.flush()

    # seeded candidates from expert-schedule study
    seeded = [
        {"OPEN_MELON": 11, "OPEN_WHEAT": 11, "SHEEP_CAP": 6, "BURST": 4},
        {"OPEN_SHEEP": 2, "OPEN_COWS": 2, "ANIMAL_GATE": 150},
        {"STRAW_CAP": 40, "MELON_ROLL": 11, "RUNWAY": 300},
        {"OPEN_FEED": 14, "MAX_ANIMALS": 16, "SHEEP_CAP": 7},
        {"OPEN_HIRES": 5, "OPEN_WHEAT": 16, "RUNWAY": 250},
    ]
    i = 0
    while time.time() < deadline:
        i += 1
        if i <= len(seeded):
            cand = dict(best_cfg)
            cand.update(seeded[i - 1])
            tag = f"seed{i}"
        else:
            cand = dict(best_cfg)
            for k in rng.sample(list(SPACE), rng.randint(2, 4)):
                d, lo, hi, is_int = SPACE[k]
                span = (hi - lo) * 0.35
                v = cand[k] + rng.uniform(-span, span)
                v = max(lo, min(hi, v))
                cand[k] = int(round(v)) if is_int else v
            tag = f"r{i}"
        res = evaluate(cand, tag)
        accept = res["margin"] > best + 300
        log.write(json.dumps({"tag": tag, "cfg": cand, **res, "accept": accept}) + "\n")
        log.flush()
        print(f"{tag}: margin={res['margin']:.0f} our={res['our_mean']:.0f} "
              f"wins={res['wins']}/{res['n']} {'ACCEPT' if accept else ''}", flush=True)
        if accept:
            best, best_cfg = res["margin"], cand
            with open(os.path.join(HERE, "best_cfg.json"), "w") as f:
                json.dump({"margin": best, "cfg": best_cfg}, f, indent=1)
    print("done. best margin:", best)
    with open(os.path.join(HERE, "best_cfg.json"), "w") as f:
        json.dump({"margin": best, "cfg": best_cfg}, f, indent=1)


if __name__ == "__main__":
    main()
