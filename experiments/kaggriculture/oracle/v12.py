"""Kaggriculture v12 — gen-3 executor: position-repairing intent follower.

Replays a top-player game (ep 89800260 winner "manual player", $152,962,
CC0-recorded public episode; same lineage the Apache-2.0 v18 family builds on)
as per-turn (position, action) intents: if our unit stands where the source
unit stood, do what it did; otherwise route toward that position. Market
orders pass through with cash-safety. Farm state is player-local, so intent
replay stays synced up to market-driven cash divergence, which repair absorbs.
"""
import json
import math
import os


def _k(name, default):
    return default


CROPS = {
    "WHEAT":      {"seed": 10,  "first": 2,  "max_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first": 2,  "max_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first": 8,  "max_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first": 10, "max_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}
ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}


MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "log",    "below_target": 0.20, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}
I0 = 10000

def _shape(func, x):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    return x


def _price(item, inv):
    p = MARKET_PARAMS[item]
    base = p["base"]
    if inv < I0:
        amp = p["below_target"] * base / _shape(p["below_func"], p["T"])
        v = base + amp * _shape(p["below_func"], I0 - inv)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], p["T"])
        v = base - amp * _shape(p["above_func"], inv - I0)
    return max(1, int(round(v)))

SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}


def _opp_capacity(farms, player):
    """Visible opponent production capacity in units/product (rough)."""
    cap = {}
    for pid, farm in enumerate(farms):
        if pid == player:
            continue
        for row in farm.get("tiles", []):
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    cap[t["crop"]] = cap.get(t["crop"], 0) + 1
                elif t.get("animal"):
                    prod = ANIMALS[t["animal"]]["product"]
                    cap[prod] = cap.get(prod, 0) + 2
    return cap



def _opp_capacity(farms, player):
    """Visible opponent production capacity in units/product (rough)."""
    cap = {}
    for pid, farm in enumerate(farms):
        if pid == player:
            continue
        for row in farm.get("tiles", []):
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    cap[t["crop"]] = cap.get(t["crop"], 0) + 1
                elif t.get("animal"):
                    prod = ANIMALS[t["animal"]]["product"]
                    cap[prod] = cap.get(prod, 0) + 2
    return cap


def _town_drain_per_day(unlocked_shops, day):
    """Units/day the town removes from market inventory, per product."""
    drain = {p: 0.0 for p in MARKET_PARAMS}
    for shop in unlocked_shops:
        products = SHOPS.get(shop, [])
        mult = 2 if len(products) == 1 else 1
        for p in products:
            drain[p] += mult * (24 / 4)
    center = 4 if day >= 20 else (2 if day >= 10 else 1)
    for p in MARKET_PARAMS:
        if p != "FERTILIZER":
            drain[p] += center * (24 / 12)
    return drain
HARVEST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}
LAST_PLANT = {"WHEAT": 25, "CARROT": 26, "MELON": 18, "STRAWBERRY": 16, "TOMATO": 18}
TPD = 24
LAST_DAY = 29

MAX_HANDS = _k("MAX_HANDS", 14)
HIRE_COST_CAP = 250
LIQ_DAY = _k("LIQ_DAY", 28)
MAX_ANIMALS = _k("MAX_ANIMALS", 14)
ANIMAL_LAST_BUY = _k("ANIMAL_LAST_BUY", 16)
STRAW_CAP = _k("STRAW_CAP", 45)
MELON_CAP = 16
SELL_FLOOR_FRAC = 0.30
FEED_STOP = {"GOOSE": 28, "COW": 28, "SHEEP": 28}

# Per-day build targets decoded from the V17 expert schedule (Apache-2.0
# public lineage): tile counts to steer toward, not hard caps.
REF_CURVE = {
    0:  {"WHEAT": 11, "MELON": 7,  "STRAWBERRY": 0,  "COW": 3, "SHEEP": 1, "QUADS": 1},
    2:  {"WHEAT": 11, "MELON": 8,  "STRAWBERRY": 0,  "COW": 3, "SHEEP": 1, "QUADS": 1},
    4:  {"WHEAT": 12, "MELON": 11, "STRAWBERRY": 2,  "COW": 3, "SHEEP": 1, "QUADS": 1},
    5:  {"WHEAT": 12, "MELON": 11, "STRAWBERRY": 3,  "COW": 4, "SHEEP": 2, "QUADS": 1},
    6:  {"WHEAT": 10, "MELON": 11, "STRAWBERRY": 4,  "COW": 5, "SHEEP": 2, "QUADS": 1},
    7:  {"WHEAT": 9,  "MELON": 11, "STRAWBERRY": 6,  "COW": 5, "SHEEP": 2, "QUADS": 2},
    8:  {"WHEAT": 8,  "MELON": 11, "STRAWBERRY": 17, "COW": 5, "SHEEP": 6, "QUADS": 2},
    9:  {"WHEAT": 8,  "MELON": 11, "STRAWBERRY": 21, "COW": 6, "SHEEP": 6, "QUADS": 2},
    10: {"WHEAT": 6,  "MELON": 10, "STRAWBERRY": 24, "COW": 8, "SHEEP": 6, "QUADS": 3},
    11: {"WHEAT": 5,  "MELON": 9,  "STRAWBERRY": 32, "COW": 8, "SHEEP": 6, "QUADS": 3},
    12: {"WHEAT": 4,  "MELON": 8,  "STRAWBERRY": 39, "COW": 8, "SHEEP": 6, "QUADS": 3},
    13: {"WHEAT": 4,  "MELON": 10, "STRAWBERRY": 44, "COW": 8, "SHEEP": 6, "QUADS": 3},
    15: {"WHEAT": 3,  "MELON": 10, "STRAWBERRY": 42, "COW": 8, "SHEEP": 6, "QUADS": 3},
    18: {"WHEAT": 4,  "MELON": 9,  "STRAWBERRY": 40, "COW": 8, "SHEEP": 6, "QUADS": 3},
    21: {"WHEAT": 5,  "MELON": 8,  "STRAWBERRY": 36, "COW": 8, "SHEEP": 6, "QUADS": 3},
    24: {"WHEAT": 5,  "MELON": 4,  "STRAWBERRY": 30, "COW": 8, "SHEEP": 6, "QUADS": 3},
    26: {"WHEAT": 32, "MELON": 0,  "STRAWBERRY": 20, "COW": 8, "SHEEP": 6, "QUADS": 3},
}


def _ref(day):
    day = day + _k("CURVE_LEAD", 0)  # build the curve N days early
    best = None
    for d in sorted(REF_CURVE):
        if d <= day:
            best = REF_CURVE[d]
    base = dict(best or REF_CURVE[0])
    alead = _k("ANIMAL_LEAD", 0)
    if alead:
        d2 = day + alead
        b2 = None
        for d in sorted(REF_CURVE):
            if d <= d2:
                b2 = REF_CURVE[d]
        if b2:
            base["COW"], base["SHEEP"] = b2["COW"], b2["SHEEP"]
    # curve-scaling knobs for margin optimization
    base["STRAWBERRY"] = int(round(base["STRAWBERRY"] * _k("STRAW_SCALE", 1.0)))
    base["MELON"] = int(round(base["MELON"] * _k("MELON_SCALE", 1.0)))
    base["WHEAT"] = int(round(base["WHEAT"] * _k("WHEAT_SCALE", 1.0)))
    base["COW"] = int(round(base["COW"] * _k("COW_SCALE", 1.0)))
    base["SHEEP"] = int(round(base["SHEEP"] * _k("SHEEP_SCALE", 1.0)))
    return base


_STATE = {}


def _shape(func, x):
    x = max(0.0, x)
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    return x


def _price(item, inv):
    p = MARKET_PARAMS[item]
    base = p["base"]
    if inv < I0:
        amp = p["below_target"] * base / _shape(p["below_func"], p["T"])
        v = base + amp * _shape(p["below_func"], I0 - inv)
    else:
        amp = p["above_target"] * base / _shape(p["above_func"], p["T"])
        v = base - amp * _shape(p["above_func"], inv - I0)
    return max(1, int(round(v)))


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "EAST" if dx > 0 else "WEST"
    if dy != 0:
        return "SOUTH" if dy > 0 else "NORTH"
    if dx != 0:
        return "EAST" if dx > 0 else "WEST"
    return None

_SELL_STATE = {}


def _adaptive_sells(obs, player, day, hour, keep_wheat):
    farms = obs.get("farms", [])
    me = farms[player]
    shed = dict((obs.get("private") or {}).get("shed") or {})
    inv_mkt = (obs.get("market") or {}).get("inventory") or {}
    liquidation = day >= 28
    st = _SELL_STATE.setdefault(player, {})
    drain = _town_drain_per_day((obs.get("town") or {}).get("unlocked_shops", []), day)
    sold = st.setdefault("sold", {})
    if st.get("day") != day:
        st["day"] = day
        sold.clear()
    orders = []
    sellable = [i for i in shed if i in MARKET_PARAMS and shed.get(i, 0) > 0]
    for item in sorted(sellable, key=lambda i: -_price(i, inv_mkt.get(i, I0)) * shed[i]):
        if len(orders) >= 8:
            break
        n = shed[item]
        if item == "WHEAT" and not liquidation and day < 29:
            n = max(0, n - keep_wheat)
            if n > 0 and _price("WHEAT", inv_mkt.get("WHEAT", I0)) < 32 and day < 26                     and sum(shed.values()) < 70:
                continue
        if n <= 0:
            continue
        if liquidation or day == 29 or sum(shed.values()) > 80:
            orders.append(["SELL", item, n])
            continue
        inv0 = inv_mkt.get(item, I0)
        p = MARKET_PARAMS[item]
        if p["above_func"] in ("log", "log10"):
            orders.append(["SELL", item, n])
            continue
        excess = max(0, inv0 - I0) + n
        remaining_drain = drain.get(item, 0) * max(0, 28 - day)
        opp_supply = _opp_capacity(farms, player).get(item, 0)
        if excess > remaining_drain * 0.8 or opp_supply >= 4:
            orders.append(["SELL", item, n])
            sold[item] = sold.get(item, 0) + n
            continue
        allowance = drain.get(item, 0) * 1.3 + 4
        k = int(max(0, allowance - sold.get(item, 0)))
        floor = MARKET_PARAMS[item]["base"] * max(0.15, 0.55 - 0.015 * day)
        kk = 0
        while kk < min(n, k) and _price(item, inv0 + kk) >= floor:
            kk += 1
        if kk > 0:
            orders.append(["SELL", item, kk])
            sold[item] = sold.get(item, 0) + kk
    return orders


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

    # purchases/hires from the record; sells from our adaptive engine
    day = obs.get("day", step // 24)
    hour = obs.get("hour", step % 24)
    buys = [m for m in it["m"] if m and m[0] != "SELL"]
    n_animals = 0
    for row in me.get("tiles", []):
        for t in row:
            if isinstance(t, dict) and t.get("animal"):
                n_animals += 1
    sells = _adaptive_sells(obs, player, day, hour, keep_wheat=n_animals + 2)
    market = (buys + sells)[:10]
    return {"farmer": farmer, "hands": hands_out, "market": market}
