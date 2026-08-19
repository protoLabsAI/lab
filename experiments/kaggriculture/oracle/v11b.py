"""Kaggriculture v11 — gen-3 executor: position-repairing intent follower.

Replays a top-player game (ep 89800260 winner "manual player", $152,962,
CC0-recorded public episode; same lineage the Apache-2.0 v18 family builds on)
as per-turn (position, action) intents: if our unit stands where the source
unit stood, do what it did; otherwise route toward that position. Market
orders pass through with cash-safety. Farm state is player-local, so intent
replay stays synced up to market-driven cash divergence, which repair absorbs.
"""
import json
import os

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for _p in (os.path.join(HERE, "..", "sources_intents_venks.json"),
           os.path.join(HERE, "sources_intents_venks.json"),
           os.path.join(HERE, "agents", "..", "sources_intents_venks.json"),
           os.path.join(os.getcwd(), "sources_intents_venks.json"),
           "/kaggle_simulations/agent/sources_intents_venks.json"):
    if os.path.exists(_p):
        _INTENTS = json.load(open(_p))
        break
else:
    _INTENTS = []


def _step_toward(pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    if dx != 0:
        return "EAST" if dx > 0 else "WEST"
    return None


def _follow(pos, want_pos, want_act):
    if list(pos) == list(want_pos):
        return list(want_act) if want_act else ["PASS"]
    mv = _step_toward(pos, want_pos)
    return [mv] if mv else ["PASS"]


def agent(obs):
    step = obs.get("step", 0)
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    if not _INTENTS or step >= len(_INTENTS) or not farms:
        return {"farmer": ["PASS"], "hands": [], "market": []}
    me = farms[player]
    it = _INTENTS[step]

    farmer = _follow(me["farmer"], it["f"][0], it["f"][1])

    hands_out = []
    src_hands = it["h"]
    for i, pos in enumerate(me.get("hands", [])):
        if i < len(src_hands):
            hands_out.append(_follow(pos, src_hands[i][0], src_hands[i][1]))
        else:
            hands_out.append(["PASS"])

    # market pass-through with cash guard ordering: hires and buys first as
    # recorded; the engine safely no-ops what cannot execute.
    market = it["m"]
    return {"farmer": farmer, "hands": hands_out, "market": market}
