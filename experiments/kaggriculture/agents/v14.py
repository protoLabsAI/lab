"""Kaggriculture v14 - market front-run experiment on v11 (SHIFT turns early)."""
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
import os as _os2
SHIFT = int(_os2.environ.get("SHIFT", "0"))  # issue market orders N turns early


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

    src_i = min(step + SHIFT, len(_INTENTS) - 1)
    queue = st + list(_INTENTS[src_i]["m"])
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
    return {"farmer": farmer, "hands": hands_out, "market": market}
