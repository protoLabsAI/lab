"""Kaggriculture v13 - wheat-squeeze experiment layered on v11.

See v11.py for the gen-3 intent-follower design.
"""
import json
import os

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
for _p in (os.path.join(HERE, "..", "sources_intents.json"),
           os.path.join(HERE, "sources_intents.json"),
           os.path.join(HERE, "agents", "..", "sources_intents.json"),
           os.path.join(os.getcwd(), "sources_intents.json"),
           "/kaggle_simulations/agent/sources_intents.json"):
    if os.path.exists(_p):
        _INTENTS = json.load(open(_p))
        break
else:
    _INTENTS = []


_PENDING = {}
_SQ = {}

import os as _os2
SQ_UNITS = int(_os2.environ.get("SQ_UNITS", "0"))      # extra wheat to accumulate early
SQ_BUY_MAX = int(_os2.environ.get("SQ_BUY_MAX", "36")) # don't buy above this price
SQ_START = int(_os2.environ.get("SQ_START", "1"))      # first day to accumulate
SQ_END = int(_os2.environ.get("SQ_END", "9"))          # last day to accumulate
SQ_SELL_DAY = int(_os2.environ.get("SQ_SELL_DAY", "22"))   # unload from this day
SQ_SELL_MIN = int(_os2.environ.get("SQ_SELL_MIN", "48"))   # only unload above this price


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

    # market pass-through with affordability-aware deferral: recorded buys
    # that cannot clear now are queued and retried, not silently dropped.
    SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
    ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
    st = _PENDING.setdefault(player, [])
    money = me.get("money", 0)
    hires_today = me.get("hires_today", 0)
    prices = (obs.get("market") or {}).get("prices") or {}

    def cost_of(m):
        if not m:
            return 0
        if m[0] == "HIRE":
            a, b = 1, 1
            for _ in range(hires_today):
                a, b = b, a + b
            return a
        if m[0] == "BUY_SEED" and len(m) >= 3:
            return SEED_COST.get(m[1], 0) * m[2]
        if m[0] == "BUY_ANIMAL" and len(m) >= 3:
            return ANIMAL_COST.get(m[1], 0) * m[2]
        if m[0] == "BUY_PRODUCT" and len(m) >= 3:
            return prices.get(m[1], 30) * m[2]
        if m[0] == "BUY_LAND":
            q = len(me.get("unlocked_quadrants", ["NW"])) - 1
            return [1000, 2000, 4000][q] if q < 3 else 10**9
        return 0

    queue = st + list(it["m"])
    market, defer, budget = [], [], money
    for m in queue:
        if len(market) >= 10:
            defer.append(m)
            continue
        c = cost_of(m)
        if c <= budget:
            market.append(m)
            budget -= c
        elif m and m[0] in ("BUY_SEED", "BUY_ANIMAL", "BUY_LAND"):
            defer.append(m)  # retry when cash arrives
        else:
            market.append(m)  # sells/product-buys: engine partial-fills safely
    _PENDING[player] = defer[-12:]

    # ---- wheat squeeze layer ----
    # The price climbs all season (town drain > net supply), and BOTH sides run
    # the same 967-unit buy program. Front-running it means we hold the cheap
    # units and their fixed-schedule buys pay the elevated price.
    if SQ_UNITS and len(market) < 10:
        day = obs.get("day", step // 24)
        priv = obs.get("private") or {}
        shed = priv.get("shed") or {}
        shed_total = sum(shed.values())
        wheat_stock = shed.get("WHEAT", 0)
        wprice = prices.get("WHEAT", 25)
        sq = _SQ.setdefault(player, {"bought": 0})
        room = 100 - shed_total
        if (SQ_START <= day <= SQ_END and wprice <= SQ_BUY_MAX
                and sq["bought"] < SQ_UNITS and room > 12 and budget > 1200):
            n = min(SQ_UNITS - sq["bought"], room - 12, int((budget - 900) // max(1, wprice)), 12)
            if n > 0:
                market.append(["BUY_PRODUCT", "WHEAT", n])
                sq["bought"] += n
                budget -= n * wprice
        elif day >= SQ_SELL_DAY and wheat_stock > 4 and wprice >= SQ_SELL_MIN:
            market.append(["SELL", "WHEAT", max(0, wheat_stock - 4)])

    return {"farmer": farmer, "hands": hands_out, "market": market[:10]}
