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
    "STRAW_SCALE": (1.0, 0.5, 1.4, False),
    "MELON_SCALE": (1.0, 0.5, 1.6, False),
    "WHEAT_SCALE": (1.0, 0.4, 2.5, False),
    "COW_SCALE": (1.0, 0.6, 1.5, False),
    "SHEEP_SCALE": (1.0, 0.6, 1.5, False),
    "RUNWAY": (400, 200, 700, True),
    "MAX_HANDS": (14, 12, 16, True),
    "LIQ_DAY": (28, 26, 28, True),
}


def play(args):
    cfg_path, seed, swap = args
    os.environ["KAGG_CFG"] = cfg_path
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    a, b = os.path.join(HERE, "agents/v10.py"), os.path.join(HERE, "opponents/sey_v7.py")
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

    res = evaluate(best_cfg, "base")
    best = res["margin"]
    print(f"baseline margin={best:.0f} our={res['our_mean']:.0f} wins={res['wins']}/{res['n']}", flush=True)
    log.write(json.dumps({"tag": "base", "cfg": best_cfg, **res}) + "\n")
    log.flush()

    # seeded candidates from expert-schedule study
    seeded = [
        {"STRAW_SCALE": 0.8},
        {"MELON_SCALE": 1.3},
        {"WHEAT_SCALE": 1.8},
        {"COW_SCALE": 1.25, "SHEEP_SCALE": 1.25},
        {"STRAW_SCALE": 0.7, "MELON_SCALE": 1.4, "WHEAT_SCALE": 1.6},
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
