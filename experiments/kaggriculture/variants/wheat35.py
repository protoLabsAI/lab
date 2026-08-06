"""Kaggriculture agent v4 — day-route planner.

At the start of each day (and when the crew changes), build explicit routes:
every unit gets an ordered list of (tile, job) stops chosen by greedy
nearest-neighbor routing. Each turn a unit executes its next stop: move
toward it, or perform the job. Jobs that became invalid are skipped.

Jobs per day:
- each plant: WATER (+ HARVEST when due)
- each empty tile (up to seed budget): PLANT then WATER (two stops, same tile)
- each animal: FEED, CARE, HARVEST (if yield), COLLECT_FERTILIZER
- feeders get a PICKUP WHEAT stop at the shed first
- carried FERTILIZER gets applied to producing ongoing crops
- end of day / final day: return to shed and DROP

Economy: day-0 all-in opening, strawberry wave + rolling melons + wheat fill,
cows+sheep to 13, land NE+SW (SE if rich), 12 hands, liquidation day 28+.
"""
import math
import json as _json
import os as _os

_BAKED = {"STRAW_SCALE": 0.8413716484597732, "MELON_SCALE": 1.5017098633961907, "WHEAT_SCALE": 3.5, "COW_SCALE": 1.1784658783798048, "SHEEP_SCALE": 1.2918498394278237, "RUNWAY": 255, "MAX_HANDS": 15, "LIQ_DAY": 28, "PORT_SHIFT": 0.15, "WHEAT_WAVE": 23, "SPREAD": 0, "SELL_FIRST": 1, "OROPT": 1, "WHEAT_HOLD": 0, "SELL_ALL": 1}
_cfg_path = _os.environ.get("KAGG_CFG")
try:
    # MERGE over the baked config - an override file must not silently
    # revert every other tuned value to its default.
    _CFG = dict(_BAKED)
    if _cfg_path:
        _CFG.update(_json.load(open(_cfg_path)))
except Exception:
    _CFG = dict(_BAKED)


def _k(name, default):
    return _CFG.get(name, default)


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
LAST_PLANT = {"WHEAT": 25, "CARROT": 26, "MELON": 18,
              "STRAWBERRY": _k("SB_LAST", 16), "TOMATO": 18}
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



# ---------------------------------------------------------------------------
# LAYER 0 - WORLD MODEL
# Every plant carries planted_day and every animal placed_day, on BOTH farms
# (farms are public). Production is deterministic given those, so the whole
# future supply calendar is computable. Nothing in the meta uses this: their
# bots replay fixed schedules and cannot react.
# ---------------------------------------------------------------------------

# realised units per production event (care bonus / fertiliser make these >1)
# Animals: CARE banks +1/day and the whole bank pays out on the production
# day, so realised yield is 1 + interval (capped by max_held), not 1.
EVENT_YIELD = {"STRAWBERRY": 1.35, "TOMATO": 1.35,
               "EGG": 2.0,     # interval 1 -> 1 + 1
               "MILK": 3.0,    # interval 2 -> 1 + 2
               "WOOL": 4.0}    # interval 3 -> 1 + 3
# one-time crops: units delivered when harvested, and the age they land at
ONESHOT = {"WHEAT": (4.5, 4), "CARROT": (3.2, 3), "MELON": (5.4, 11)}
SELL_LAG = 1          # harvest -> reaches the market
HORIZON = 7           # days of forecast


def _supply_calendar(farms, day, last_day=LAST_DAY):
    """Projected units hitting the market per product per future day, from
    BOTH farms' visible plant/animal timers."""
    cal = {p: [0.0] * (last_day + 2) for p in MARKET_PARAMS}
    for farm in farms:
        for row in farm.get("tiles", []):
            for t in row:
                if not isinstance(t, dict):
                    continue
                if t.get("kind") == "PLANT":
                    crop = t.get("crop")
                    c = CROPS.get(crop)
                    if not c:
                        continue
                    p0 = t.get("planted_day", day)
                    if c["ongoing"]:
                        fired = 0
                        d = p0 + c["first"] - 1
                        while d <= last_day and fired < c["max_yield"]:
                            if d >= day:
                                idx = min(last_day + 1, d + SELL_LAG)
                                cal[crop][idx] += EVENT_YIELD.get(crop, 1.2)
                            fired += 1
                            d += max(1, c["interval"])
                    else:
                        units, at_age = ONESHOT.get(crop, (c["max_yield"] * 0.7, c["max_day"]))
                        d = p0 + at_age
                        if day <= d <= last_day:
                            cal[crop][min(last_day + 1, d + SELL_LAG)] += units
                elif t.get("animal"):
                    a = ANIMALS[t["animal"]]
                    prod = a["product"]
                    p0 = t.get("placed_day", day)
                    d = p0 + a["first"] - 1
                    while d <= last_day:
                        if d >= day:
                            idx = min(last_day + 1, d + SELL_LAG)
                            cal[prod][idx] += EVENT_YIELD.get(prod, 1.5)
                        d += max(1, a["interval"])
    return cal


def _update_bias(st, day, inv_mkt, shops, farms):
    """Self-correction. The opponent's SHED is hidden, so harvested-but-unsold
    backlog is invisible and our calendar under-counts supply. Compare what we
    predicted for today against what actually happened and carry the residual
    forward as a per-product bias (units/day)."""
    bias = st.setdefault("fc_bias", {})
    prev = st.get("fc_prev")
    if prev is not None and prev["day"] < day:
        span = max(1, day - prev["day"])
        cal = prev["cal"]
        for item in MARKET_PARAMS:
            modelled = -sum(_town_drain_per_day(shops, prev["day"] + k).get(item, 0)
                            for k in range(span))
            modelled += sum(cal[item][prev["day"] + k]
                            for k in range(span) if prev["day"] + k < len(cal[item]))
            actual = inv_mkt.get(item, I0) - prev["inv"].get(item, I0)
            resid = (actual - modelled) / span
            b = 0.75 * bias.get(item, 0.0) + 0.25 * resid
            # only carry PERSISTENT error forward; clamp so a noisy day cannot
            # swamp a product the calendar already predicts well
            cap = _town_drain_per_day(shops, day).get(item, 0) + 4
            bias[item] = max(-cap, min(cap, b))
    st["fc_prev"] = {"day": day, "inv": dict(inv_mkt),
                     "cal": _supply_calendar(farms, day)}
    return bias


def _price_forecast(farms, day, inv_mkt, shops, bias=None, horizon=HORIZON):
    """Projected market price per product for the next `horizon` days:
    inventory walks forward under projected supply, town drain and the
    learned residual bias."""
    cal = _supply_calendar(farms, day)
    bias = bias or {}
    out = {}
    for item in MARKET_PARAMS:
        inv = float(inv_mkt.get(item, I0))
        b = bias.get(item, 0.0)
        path = []
        for k in range(horizon + 1):
            d = day + k
            path.append(_price(item, int(round(inv))))
            inv += (cal[item][d] if d < len(cal[item]) else 0.0)
            inv -= _town_drain_per_day(shops, d).get(item, 0)
            inv += b
        out[item] = path
    return out


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


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _survey(farm):
    tiles = farm["tiles"]
    board = len(tiles)
    out = {"plants": [], "animals": [], "empty_structs": [], "empty": [], "weeds": []}
    for y in range(board):
        for x in range(board):
            t = tiles[y][x]
            if t is None:
                out["empty"].append((x, y))
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    out["plants"].append((x, y, t))
                elif k == "WEED":
                    out["weeds"].append((x, y))
                elif k in ("COOP", "PASTURE"):
                    (out["animals"] if t.get("animal") else out["empty_structs"]).append((x, y, t))
    return out


def _build_day_jobs(sv, day, liquidation):
    """Single-stop jobs: (prio, tile, op). PLANT handled separately as chains."""
    jobs = []
    for x, y, t in sv["plants"]:
        c = CROPS.get(t["crop"])
        if c is None:
            continue
        age = day - t["planted_day"]
        if not t["watered_today"] and not (day == LAST_DAY and age < c["first"]):
            streak = t["consecutive_unwatered"]
            if _k("WATER_ALL", 0):
                # leader waters 851x/game to our 558; water everything, let the
                # router's capacity budget decide what actually fits
                jobs.append((0 if streak >= 1 else 2, (x, y), ("WATER", None)))
                continue
            if c["ongoing"]:
                # base yield is water-independent; water for survival and for
                # the fertilizer doubling on production eves
                produces_tonight = (day + 1 - t["planted_day"] - c["first"]) >= 0 and \
                    (day + 1 - t["planted_day"] - c["first"]) % max(1, c["interval"]) == 0
                fertd = t.get("fertilized_until_day", -1) >= day
                need = streak >= 1 or (produces_tonight and fertd)
            else:
                window_start = (c["max_day"] + 1) // 2
                in_window = window_start <= age <= c["max_day"]
                need = streak >= 1 or in_window
            if need:
                jobs.append((0 if streak >= 1 else 1, (x, y), ("WATER", None)))
        if c["ongoing"]:
            if t["yield_units"] >= _k("H_CROP", 2) or (t["yield_units"] > 0 and (
                    liquidation or day >= LAST_DAY - 1)):
                jobs.append((_k("H_PRIO", 2), (x, y), ("HARVEST", None)))
        else:
            ready_age = HARVEST_AGE.get(t["crop"], c["max_day"])
            if t["yield_units"] > 0 and age >= c["first"] and (
                    age >= ready_age or day == LAST_DAY):
                # mature one-time crops decay fast: harvesting beats watering
                jobs.append((0.5 if age >= ready_age else 2, (x, y), ("HARVEST", None)))
    for x, y, t in sv["animals"]:
        a = ANIMALS[t["animal"]]
        active = day <= FEED_STOP.get(t["animal"], 27)
        if active and not t["fed_today"]:
            jobs.append((0, (x, y), ("FEED", None)))
        if active and not t["cared_today"]:
            jobs.append((2, (x, y), ("CARE", None)))
        if t.get("fertilizer_available") and not liquidation:
            jobs.append((3, (x, y), ("COLLECT_FERTILIZER", None)))
        if t["yield_units"] > 0:
            next_prod = 1 + t.get("pending_care_bonus", 0)
            overflow = (t["yield_units"] + next_prod > a["max_held"]
                        or t["yield_units"] >= _k("H_ANIM", 99))
            prio = 0.5 if (day >= LAST_DAY - 1 or overflow) else (
                1 if t["yield_units"] >= 3 else 2)
            jobs.append((prio, (x, y), ("HARVEST", None)))

    for x, y in sv["weeds"]:
        if day <= LAST_DAY - 4:
            jobs.append((2.5 if day <= 22 else 4, (x, y), ("DIG", None)))
    return jobs



def _crop_value(crop, day, fc, horizon_end=LAST_DAY):
    """Expected $/tile for planting `crop` today, priced at ITS harvest date
    using the forecast. This is the adaptive strategy signal: if the opponent's
    visible plants will flood a market exactly when ours matures, that crop is
    worth less to us than one maturing into a thin market."""
    c = CROPS.get(crop)
    if not c:
        return 0.0
    path = fc.get(crop) or []
    if not path:
        return 0.0

    def p_at(d):
        k = max(0, min(len(path) - 1, d - day))
        return path[k]

    if c["ongoing"]:
        total, fired, d = 0.0, 0, day + c["first"]
        while d <= horizon_end and fired < c["max_yield"]:
            total += 1.35 * p_at(d)
            fired += 1
            d += max(1, c["interval"])
        return total - c["seed"]
    units, at_age = ONESHOT.get(crop, (c["max_yield"] * 0.7, c["max_day"]))
    d = day + at_age
    if d > horizon_end:
        return -1e9
    return units * p_at(d) - c["seed"]


def _plan_planting(sv, day, seeds, n_animals_total, fc=None):
    """Plant toward the reference build curve (largest deficit first)."""
    if day > 26:
        return []
    ref = _ref(day)
    counts = {}
    for _, _, t in sv["plants"]:
        counts[t["crop"]] = counts.get(t["crop"], 0) + 1
    budget = {c: seeds.get(c, 0) for c in CROPS}
    # LAYER 1 - ROLLING PORTFOLIO. Re-allocate tiles between crops by their
    # forecast harvest-date value, bounded so we stay near the (validated)
    # reference mix rather than chasing noise.
    # LATE WHEAT WAVE. Wheat is $10, first-yields in 2 days, peaks at 4, and
    # its glut curve is log (effectively bottomless). Late season we have spare
    # cash and idle crew but no time for anything slower, so every free tile
    # goes to wheat. The leader runs 31 wheat tiles on d26; we ran 3.
    wave = _k("WHEAT_WAVE", 20) <= day <= LAST_PLANT["WHEAT"]
    cand = [c for c in ("STRAWBERRY", "MELON", "WHEAT") if day <= LAST_PLANT.get(c, 26)]
    vals = {c: max(0.0, _crop_value(c, day, fc)) for c in cand} if fc else {}
    tgt = dict(ref)
    if vals and sum(vals.values()) > 0:
        shift = _k("PORT_SHIFT", 0.35)
        pool = sum(ref.get(c, 0) for c in cand)
        if pool > 0:
            share = {c: vals[c] / sum(vals.values()) for c in cand}
            for c in cand:
                base_t = ref.get(c, 0)
                want = share[c] * pool
                tgt[c] = int(round(base_t + shift * (want - base_t)))
    if wave:
        tgt["WHEAT"] = counts.get("WHEAT", 0) + len(sv["empty"])
    deficits = []
    for crop in cand:
        d = tgt.get(crop, 0) - counts.get(crop, 0)
        if d > 0 and budget.get(crop, 0) > 0:
            deficits.append((vals.get(crop, 0.0), d, crop))
    deficits.sort(reverse=True)
    deficits = [(d, crop) for _v, d, crop in deficits]
    out = []
    reserve_inner = n_animals_total < (ref.get("COW", 0) + ref.get("SHEEP", 0)) and day <= 12
    empties = sorted(sv["empty"], key=lambda p: _dist(p, (4.5, 4.5)))
    ei = 0
    for d, crop in deficits:
        take = min(d, budget.get(crop, 0))
        while take > 0 and ei < len(empties):
            x, y = empties[ei]
            ei += 1
            if reserve_inner and _dist((x, y), (4.5, 4.5)) <= 1.5:
                continue
            out.append(((x, y), crop))
            take -= 1
    return out


def _fert_pairs(sv, day, liquidation):
    """{animal_tile: strawberry/melon tile} pairing each available fertilizer
    with the nearest unfertilized target worth fertilizing."""
    if liquidation:
        return {}
    sources = [(x, y) for x, y, t in sv["animals"] if t.get("fertilizer_available")]
    targets = []
    for x, y, t in sv["plants"]:
        c = CROPS.get(t["crop"])
        if c is None or t.get("fertilized_until_day", -1) >= day:
            continue
        age = day - t["planted_day"]
        remaining = LAST_DAY - day
        if c["ongoing"] and age >= c["first"] - 3 and remaining >= 2:
            done = max(0, (age - c["first"]) // max(1, c["interval"]) + 1) if age >= c["first"] else 0
            if c["max_yield"] - done >= 2:
                targets.append((x, y))
        elif t["crop"] == "MELON" and 2 <= age <= 9:
            targets.append((x, y))
    pairs = {}
    used = set()
    sources = sources[:8]
    for s in sources:
        best = None
        for tgt in targets:
            if tgt in used:
                continue
            d = _dist(s, tgt)
            if best is None or d < best[0]:
                best = (d, tgt)
        if best is not None:
            used.add(best[1])
            pairs[s] = best[1]
    return pairs


def _placement_chains(sv, shed, open_shed):
    """One chain per unhoused animal: pickup -> (build) -> place."""
    chains = []
    free_structs = {"COOP": [(x, y) for x, y, t in sv["empty_structs"] if t["kind"] == "COOP"],
                    "PASTURE": [(x, y) for x, y, t in sv["empty_structs"] if t["kind"] == "PASTURE"]}
    build_spots = sorted((p for p in sv["empty"] if _dist(p, (4.5, 4.5)) <= 4.5),
                         key=lambda p: _dist(p, (4.5, 4.5)))
    used = set()
    for a_kind in ("COW", "SHEEP", "GOOSE"):
        for _ in range(shed.get(a_kind, 0)):
            want = ANIMALS[a_kind]["structure"]
            if free_structs[want]:
                spot = free_structs[want].pop(0)
                build = None
            else:
                spot = next((p for p in build_spots if p not in used), None)
                if spot is None:
                    break
                used.add(spot)
                build = "BUILD_COOP" if want == "COOP" else "BUILD_PASTURE"
            shed_tile = min(open_shed, key=lambda s: _dist(s, spot))
            chain = [(shed_tile, ("PICKUP", (a_kind, 1)))]
            if build:
                chain.append((spot, (build, None)))
            chain.append((spot, ("PLACE", a_kind)))
            chains.append(chain)
    return chains


def _route_units(units, hour, jobs, plant_jobs, place_chains, fert_pairs):
    """Bundle jobs per tile, route chains greedily by priority, then improve
    each unit's route with nearest-neighbor reordering inside contiguous
    same-priority runs (travel shrinks; priority order preserved)."""
    budget_turns = max(1, TPD - hour - 1)
    blocks = [[] for _ in units]  # per-unit: (prio_class, chain)
    loads = [0.0] * len(units)
    positions = [tuple(p) for p in units]
    prio_now = [0.0]

    def assign_chain(chain, force=False):
        best, best_cost = None, None
        for ui in range(len(units)):
            d = _dist(positions[ui], chain[0][0])
            if not force and loads[ui] + d + len(chain) > budget_turns:
                continue
            cost = 2.0 * d + 0.6 * loads[ui]
            if best is None or cost < best_cost:
                best, best_cost = ui, cost
        if best is None:
            if not force:
                return False
            best = min(range(len(units)), key=lambda ui: loads[ui])
        blocks[best].append((prio_now[0], list(chain)))
        for tile, opa in chain:
            loads[best] += _dist(positions[best], tile) + 1
            positions[best] = tile
        return True

    prio_now[0] = -1.0
    for chain in place_chains:
        assign_chain(chain, force=True)

    groups = {}
    for prio, tile, opa in jobs:
        groups.setdefault(tile, []).append((prio, opa))
    bundles = []
    for tile, lst in groups.items():
        lst.sort(key=lambda e: e[0])
        chain = [(tile, opa) for _, opa in lst]
        if tile in fert_pairs:
            if not any(opa[0] == "COLLECT_FERTILIZER" for _, opa in chain):
                chain.append((tile, ("COLLECT_FERTILIZER", None)))
            chain.append((fert_pairs[tile], ("FERTILIZE", None)))
        bundles.append((lst[0][0], chain))
    bundles.sort(key=lambda b: b[0])

    urgent = [b for b in bundles if b[0] <= 2.5]
    rest = [b for b in bundles if b[0] > 2.5]
    for prio, chain in urgent:
        prio_now[0] = 1.0
        assign_chain(chain, force=prio <= 0)
    prio_now[0] = 2.0
    for tile, crop in plant_jobs:
        assign_chain([(tile, ("PLANT", crop)), (tile, ("WATER", None))])
    prio_now[0] = 3.0
    for prio, chain in rest:
        assign_chain(chain)

    # ---- CROSS-UNIT ROUTE IMPROVEMENT (or-opt relocate) ----
    # Greedy assignment is typically 20-30% off optimal on this routing
    # problem, and throughput is our measured bottleneck (we complete about
    # half the production events of a top executor on the same farm). Move
    # single jobs between units when it shortens total travel, which frees
    # turns for more jobs. Chains are never split.
    if _k("OROPT", 1):
        def route_cost(ui, blist):
            pos = tuple(units[ui]); c = 0.0
            for _p, ch in blist:
                for tile, _o in ch:
                    c += _dist(pos, tile) + 1
                    pos = tile
            return c
        costs = [route_cost(i, blocks[i]) for i in range(len(units))]
        for _ in range(_k("OROPT_PASSES", 3)):
            improved = False
            for src_u in range(len(units)):
                if not blocks[src_u]:
                    continue
                for bi in range(len(blocks[src_u]) - 1, -1, -1):
                    prio, chain = blocks[src_u][bi]
                    if len(chain) > 1:
                        continue                      # keep chains intact
                    rest = blocks[src_u][:bi] + blocks[src_u][bi + 1:]
                    new_src = route_cost(src_u, rest)
                    gain_src = costs[src_u] - new_src
                    if gain_src <= 0:
                        continue
                    best = None
                    for dst_u in range(len(units)):
                        if dst_u == src_u:
                            continue
                        for at in range(len(blocks[dst_u]) + 1):
                            cand = blocks[dst_u][:at] + [(prio, chain)] + blocks[dst_u][at:]
                            nc = route_cost(dst_u, cand)
                            add = nc - costs[dst_u]
                            if add < gain_src and (best is None or add < best[0]):
                                best = (add, dst_u, cand, nc)
                    if best is not None and best[0] < gain_src - 0.01:
                        _add, dst_u, cand, nc = best
                        blocks[src_u] = rest
                        blocks[dst_u] = cand
                        costs[src_u] = new_src
                        costs[dst_u] = nc
                        improved = True
                        break
            if not improved:
                break

    # within-run NN improvement: reorder blocks inside same-priority runs
    routes = [[] for _ in units]
    for ui in range(len(units)):
        pos = tuple(units[ui])
        route = []
        i = 0
        blist = blocks[ui]
        while i < len(blist):
            j = i
            while j < len(blist) and blist[j][0] == blist[i][0]:
                j += 1
            run = [c for _, c in blist[i:j]]
            while run:
                nxt = min(run, key=lambda c: _dist(pos, c[0][0]))
                run.remove(nxt)
                route.extend(nxt)
                pos = nxt[-1][0]
            i = j
        routes[ui] = route
    return routes


def _stop_valid(op, arg, tile_obj, day):
    if op == "WATER":
        return (isinstance(tile_obj, dict) and tile_obj.get("kind") == "PLANT"
                and not tile_obj["watered_today"])
    if op == "PLANT":
        return tile_obj is None
    if op == "HARVEST":
        if not isinstance(tile_obj, dict):
            return False
        if tile_obj.get("kind") == "PLANT":
            return (tile_obj.get("yield_units", 0) > 0
                    and day - tile_obj["planted_day"] >= CROPS[tile_obj["crop"]]["first"])
        return bool(tile_obj.get("animal")) and tile_obj.get("yield_units", 0) > 0
    if op == "FEED":
        return (isinstance(tile_obj, dict) and tile_obj.get("animal")
                and not tile_obj["fed_today"])
    if op == "CARE":
        return (isinstance(tile_obj, dict) and tile_obj.get("animal")
                and not tile_obj["cared_today"])
    if op == "COLLECT_FERTILIZER":
        return (isinstance(tile_obj, dict) and tile_obj.get("animal")
                and tile_obj.get("fertilizer_available"))
    if op == "DIG":
        return isinstance(tile_obj, dict) and tile_obj.get("kind") == "WEED"
    if op in ("BUILD_COOP", "BUILD_PASTURE"):
        return tile_obj is None
    if op == "PLACE":
        return (isinstance(tile_obj, dict) and tile_obj.get("kind") in ("COOP", "PASTURE")
                and not tile_obj.get("animal"))
    if op == "PICKUP":
        return True
    if op == "FERTILIZE":
        return (isinstance(tile_obj, dict) and tile_obj.get("kind") == "PLANT"
                and tile_obj.get("fertilized_until_day", -1) < day)
    return False


def _go_or_do(pos, target, do_action):
    if tuple(pos) == tuple(target):
        return do_action
    mv = _step_toward(pos, target)
    return [mv] if mv else ["PASS"]


def agent(obs):
    player = obs.get("player", 0)
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private", {}) or {}
    inv_mkt = (obs.get("market", {}) or {}).get("inventory", {}) or {}
    tiles = farm["tiles"]
    board = len(tiles)
    money = farm["money"]
    seeds = dict(private.get("seeds", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    inventories = [dict(i) for i in private.get("inventories", [{}])]
    liquidation = day >= LIQ_DAY

    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
    n_units = len(units)
    while len(inventories) < n_units:
        inventories.append({})

    h = board // 2
    shed_access = [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]
    open_shed = [s for s in shed_access if tiles[s[1]][s[0]] != "LOCKED"] or [shed_access[0]]

    def nearest_shed(p):
        return min(open_shed, key=lambda s: _dist(p, s))

    sv = _survey(farm)
    n_animals = len(sv["animals"])
    unhoused = {a: shed.get(a, 0) + sum(inv.get(a, 0) for inv in inventories) for a in ANIMALS}
    total_unhoused = sum(unhoused.values())
    active_animals = sum(1 for _, _, t in sv["animals"] if day <= FEED_STOP.get(t["animal"], 27))
    feed_animals = active_animals + total_unhoused
    feed_reserve = feed_animals * 2 + 2 if feed_animals else 0

    st = _STATE.setdefault(player, {})
    if st.get("day") != day:
        st.clear()
        st["day"] = day
        st["routes"] = {}

    # ---- (re)build routes when crew size changes or at day start ----
    _shops = obs.get("town", {}).get("unlocked_shops", [])
    st["fc"] = _price_forecast(farms, day, inv_mkt, _shops,
                               _update_bias(st, day, inv_mkt, _shops, farms))
    jobs = _build_day_jobs(sv, day, liquidation)
    if st.get("crew") != n_units or st.get("n_anim") != n_animals:
        st["crew"] = n_units
        st["n_anim"] = n_animals
        plant_jobs = _plan_planting(sv, day, seeds, n_animals + total_unhoused, st.get("fc"))
        chains = _placement_chains(sv, shed, open_shed)
        fpairs = _fert_pairs(sv, day, liquidation)
        routes = _route_units(units, hour, jobs, plant_jobs, chains, fpairs)
        st["routes"] = {ui: list(r) for ui, r in enumerate(routes)}

    routes = st.setdefault("routes", {})
    actions = [None] * n_units
    seeds_live = dict(seeds)

    # ---- unit control ----
    for ui in range(n_units):
        pos = units[ui]
        inv = inventories[ui]
        route = routes.get(ui, [])
        act = None

        # carrying an animal but no PLACE stop queued (safety net)
        carried_kind = next((a for a in ("COW", "SHEEP", "GOOSE") if inv.get(a, 0) > 0), None)
        if carried_kind and not any(opa[0] == "PLACE" for _, opa in route):
            want = ANIMALS[carried_kind]["structure"]
            slots = [(x, y) for x, y, t in sv["empty_structs"] if t["kind"] == want]
            if slots:
                tgt = min(slots, key=lambda s: _dist(pos, s))
                actions[ui] = _go_or_do(pos, tgt, ["PLACE", carried_kind])
            else:
                spots = [p for p in sv["empty"] if _dist(p, (4.5, 4.5)) <= 4.5]
                if spots:
                    tgt = min(spots, key=lambda s: _dist(pos, s))
                    b = "BUILD_COOP" if want == "COOP" else "BUILD_PASTURE"
                    actions[ui] = _go_or_do(pos, tgt, [b])
                else:
                    actions[ui] = ["PASS"]
            continue

        while route:
            tile, (op, arg) = route[0]
            tile_obj = tiles[tile[1]][tile[0]]
            if op == "PLANT":
                if tile_obj is not None or seeds_live.get(arg, 0) <= 0:
                    route.pop(0)
                    # drop the paired WATER stop too if tile isn't ours
                    if tile_obj is not None and route and route[0][0] == tile and route[0][1][0] == "WATER":
                        if not (isinstance(tile_obj, dict) and tile_obj.get("kind") == "PLANT"):
                            route.pop(0)
                    continue
                if pos == tile:
                    act = ["PLANT", arg]
                    seeds_live[arg] -= 1
                    route.pop(0)
                else:
                    act = [_step_toward(pos, tile) or "PASS"]
                break
            if op == "FEED":
                if not _stop_valid(op, arg, tile_obj, day):
                    route.pop(0)
                    continue
                if inv.get("WHEAT", 0) <= 0:
                    if shed.get("WHEAT", 0) > 0:
                        n_feeds = sum(1 for _, (o2, _) in route if o2 == "FEED")
                        k = min(shed.get("WHEAT", 0), max(n_feeds, 2), 10)
                        tgt = nearest_shed(pos)
                        act = _go_or_do(pos, tgt, ["PICKUP", "WHEAT", k])
                        if act[0] == "PICKUP":
                            shed["WHEAT"] -= k
                            inv["WHEAT"] = inv.get("WHEAT", 0) + k
                        break
                    route.pop(0)
                    continue
                if pos == tile:
                    act = ["FEED"]
                    inv["WHEAT"] -= 1
                    route.pop(0)
                else:
                    act = [_step_toward(pos, tile) or "PASS"]
                break
            if op == "PICKUP":
                item, k = arg if isinstance(arg, tuple) else (arg, 1)
                if shed.get(item, 0) <= 0:
                    # nothing to pick up: drop chain stops tied to this item
                    route.pop(0)
                    while route and route[0][1][0] in ("BUILD_COOP", "BUILD_PASTURE", "PLACE"):
                        route.pop(0)
                    continue
                if pos == tile:
                    k = min(k, shed.get(item, 0))
                    act = ["PICKUP", item, k]
                    shed[item] -= k
                    inv[item] = inv.get(item, 0) + k
                    route.pop(0)
                else:
                    act = [_step_toward(pos, tile) or "PASS"]
                break
            if op == "PLACE":
                if not (inv.get(arg, 0) > 0):
                    route.pop(0)
                    continue
                if not _stop_valid(op, arg, tile_obj, day):
                    # structure occupied/missing: find another spot next rebuild
                    route.pop(0)
                    continue
                if pos == tile:
                    act = ["PLACE", arg]
                    inv[arg] -= 1
                    route.pop(0)
                else:
                    act = [_step_toward(pos, tile) or "PASS"]
                break
            # generic single-tile ops
            if not _stop_valid(op, arg, tile_obj, day):
                route.pop(0)
                continue
            if pos == tile:
                act = [op] if arg is None else [op, arg]
                route.pop(0)
            else:
                act = [_step_toward(pos, tile) or "PASS"]
            break
        if act is not None:
            actions[ui] = act
            continue

        # route done: apply carried fertilizer, then courier home
        if inv.get("FERTILIZER", 0) > 0 and day < LIQ_DAY - 1:
            targets = []
            for x, y, t in sv["plants"]:
                c = CROPS.get(t["crop"])
                if c is None or t.get("fertilized_until_day", -1) >= day:
                    continue
                age = day - t["planted_day"]
                if c["ongoing"] and age >= c["first"] - 3:
                    targets.append((x, y))
                elif t["crop"] == "MELON" and 3 <= age <= 9:
                    targets.append((x, y))
            if targets:
                tgt = min(targets, key=lambda p: _dist(pos, p))
                actions[ui] = _go_or_do(pos, tgt, ["FERTILIZE"])
                continue
        carrying = sum(inv.values())
        d_home = _dist(pos, nearest_shed(pos))
        must_go = hour >= TPD - d_home - 3
        if carrying > 0 and (day == LAST_DAY or must_go or carrying >= _k("CARRY", 14)):
            actions[ui] = _go_or_do(pos, nearest_shed(pos), ["DROP"])
            continue
        # Opportunistic work: the day's routes are static, so a unit that
        # finishes early would idle while newly-ripened work sits unserved.
        # Claim the nearest still-valid job no one else is handling.
        claimed = set()
        for r in routes.values():
            for tile, _opa in r:
                claimed.add(tile)
        for uj in range(n_units):
            a = actions[uj]
            if a and a[0] in ("WATER", "HARVEST", "FEED", "CARE",
                              "COLLECT_FERTILIZER", "FERTILIZE", "DIG", "PLANT"):
                claimed.add(tuple(units[uj]))
        best = None
        for jprio, jtile, jopa in jobs:
            if jtile in claimed:
                continue
            if not _stop_valid(jopa[0], jopa[1], tiles[jtile[1]][jtile[0]], day):
                continue
            if jopa[0] == "FEED" and inv.get("WHEAT", 0) <= 0:
                continue
            d = _dist(pos, jtile)
            if hour + d + 1 + min(_dist(jtile, s) for s in open_shed) > TPD - 1:
                continue
            score = jprio * 4 + d
            if best is None or score < best[0]:
                best = (score, jtile, jopa)
        if best is not None:
            actions[ui] = _go_or_do(pos, best[1], [best[2][0]] if best[2][1] is None
                                    else [best[2][0], best[2][1]])
            routes.setdefault(ui, []).append((best[1], best[2]))
        elif carrying > 0:
            actions[ui] = _go_or_do(pos, nearest_shed(pos), ["DROP"])
        else:
            actions[ui] = ["PASS"]

    # emergency watering: dying plants with no route coverage, idle units available
    if hour >= 14:
        critical = [(x, y) for x, y, t in sv["plants"]
                    if t["consecutive_unwatered"] >= 1 and not t["watered_today"]]
        routed = set()
        for r in routes.values():
            for tile, (op, _) in r:
                if op == "WATER":
                    routed.add(tile)
        for (x, y) in critical:
            if (x, y) in routed:
                continue
            idle = [ui for ui in range(n_units) if actions[ui] == ["PASS"]]
            if not idle:
                break
            ui = min(idle, key=lambda u: _dist(units[u], (x, y)))
            actions[ui] = _go_or_do(units[ui], (x, y), ["WATER"])

    # dedupe simultaneous PLANTs beyond seed count (engine drops all if over)
    plant_counts = {}
    for ui in range(n_units):
        a = actions[ui]
        if a and a[0] == "PLANT":
            plant_counts[a[1]] = plant_counts.get(a[1], 0) + 1
    for crop, cnt in plant_counts.items():
        have = dict(private.get("seeds", {})).get(crop, 0)
        if cnt > have:
            over = cnt - have
            for ui in range(n_units - 1, -1, -1):
                if over <= 0:
                    break
                if actions[ui] and actions[ui][0] == "PLANT" and actions[ui][1] == crop:
                    actions[ui] = ["PASS"]
                    over -= 1

    # =====================================================
    # MARKET
    # =====================================================
    market = []

    if day == 0 and hour == 0:
        market = ([["HIRE"]] * _k("OPEN_HIRES", 4) +
                  [["BUY_ANIMAL", "COW", _k("OPEN_COWS", 3)],
                   ["BUY_ANIMAL", "SHEEP", _k("OPEN_SHEEP", 1)],
                   ["BUY_SEED", "MELON", _k("OPEN_MELON", 7)],
                   ["BUY_SEED", "WHEAT", _k("OPEN_WHEAT", 10)],
                   ["BUY_PRODUCT", "WHEAT", _k("OPEN_FEED", 6)]])
        return {"farmer": actions[0], "hands": actions[1:], "market": market}

    # ---- budget envelopes: feed -> hires -> land -> seeds -> animals ----

    # 2) hires: size crew to the whole farm
    if True:
        n_plantable = len(sv["empty"]) if day < LIQ_DAY - 2 else 0
        seeds_ready = sum(seeds.get(c, 0) for c in CROPS)
        to_plant = min(n_plantable, seeds_ready + 8)
        workload = len(sv["plants"]) * 2.2 + to_plant * 3 + active_animals * 5.5
        target_units = 1 + int(workload // 16)
        if day >= 1:
            target_units = max(target_units, 5)
        if day <= 2:
            target_units = min(target_units, 7)
        if len(sv["plants"]) + n_animals > 25 and day >= 3:
            target_units = max(target_units, 13)
        if day >= LAST_DAY - 1:
            target_units = max(min(target_units, 12), 10)
        target_units = min(target_units, MAX_HANDS + 1)
        hires_today = farm.get("hires_today", 0)
        while (n_units + sum(1 for m in market if m == ["HIRE"]) < target_units
               and len(market) < 9):
            cost = _fib(hires_today)
            hire_floor = 0 if cost <= 5 else (150 if day <= 3 else 30)
            if cost > HIRE_COST_CAP or money - cost < hire_floor or hour > 19:
                break
            market.append(["HIRE"])
            money -= cost
            hires_today += 1

    # 1) feed wheat first (starving animals is the worst loss)
    wheat_stock = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
    if feed_animals > 0 and wheat_stock < feed_reserve:
        wp = _price("WHEAT", inv_mkt.get("WHEAT", I0) - 1)
        if wp <= 60:
            n_buy = min(feed_reserve + 2 - wheat_stock, int(max(0, money - 20) // wp))
            if n_buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", n_buy])
                money -= n_buy * wp

    # 3) land (needs crew to work it)
    n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if hour <= 4 and day >= 2:
        land_prices = [1000, 2000, 4000]
        if n_extra == 0 and day <= 16 and money > 1300:
            market.append(["BUY_LAND"])
            money -= 1000
        elif n_extra == 1 and day >= 4 and day <= 16 and money > 2700:
            market.append(["BUY_LAND"])
            money -= 2000
        elif n_extra == 2 and 6 <= day <= 14 and money > 6000:
            market.append(["BUY_LAND"])
            money -= 4000

    # 4a) animals + land toward the reference curve
    if day <= 16 and hour <= 8 and len(market) < 9:
        ref_now = _ref(day)
        n_cow = sum(1 for _, _, t in sv["animals"] if t["animal"] == "COW") + unhoused.get("COW", 0)
        n_sheep2 = sum(1 for _, _, t in sv["animals"] if t["animal"] == "SHEEP") + unhoused.get("SHEEP", 0)
        for kind, have, target in (("COW", n_cow, ref_now.get("COW", 0)),
                                   ("SHEEP", n_sheep2, ref_now.get("SHEEP", 0))):
            need = max(0, target - have)
            while need > 0 and len(market) < 9 and money - ANIMALS[kind]["cost"] > 150:
                market.append(["BUY_ANIMAL", kind, 1])
                money -= ANIMALS[kind]["cost"]
                total_unhoused += 1
                need -= 1
    ref_q = _ref(day).get("QUADS", 1)
    n_extra2 = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if n_extra2 + 1 < ref_q and hour <= 8 and len(market) < 9 and n_extra2 < 3:
        cost = [1000, 2000, 4000][n_extra2]
        if money - cost > 150:
            market.append(["BUY_LAND"])
            money -= cost

    # wheat corner v2: their buy schedule is price-insensitive; ours isn't.
    # Accumulate cheap early, sell into the pumped price their buys create.
    c2_units = _k("CORNER2", 0)
    if c2_units and 1 <= day <= 8 and len(market) < 9:
        wp = _price("WHEAT", inv_mkt.get("WHEAT", I0) - 1)
        stock = shed.get("WHEAT", 0) + sum(iv.get("WHEAT", 0) for iv in inventories)
        if wp <= _k("CORNER2_BUYCAP", 0) and stock < c2_units + feed_reserve:
            n_buy = min(c2_units + feed_reserve - stock,
                        int(max(0, money * 0.4 - 150) // wp), 15)
            if n_buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", n_buy])
                money -= n_buy * wp

    # 4) seeds: buy toward the reference curve (2-day lookahead)
    if day < LIQ_DAY - 2 and len(market) < 8:
        ref_now = _ref(day)
        ref_ahead = _ref(min(26, day + 2))
        counts_now = {}
        for _, _, t in sv["plants"]:
            counts_now[t["crop"]] = counts_now.get(t["crop"], 0) + 1
        spendable = max(0, money - (_k("RESERVE_E", 250) if day <= 9 else _k("RESERVE_L", 200)))
        for crop in ("STRAWBERRY", "MELON", "WHEAT"):
            if day > LAST_PLANT.get(crop, 26) or len(market) >= 9:
                continue
            target = max(ref_now.get(crop, 0), ref_ahead.get(crop, 0))
            _fc = st.get("fc")
            if _fc:
                _vals = {c: max(0.0, _crop_value(c, day, _fc))
                         for c in ("STRAWBERRY", "MELON", "WHEAT")
                         if day <= LAST_PLANT.get(c, 26)}
                if crop in _vals and sum(_vals.values()) > 0:
                    _pool = sum(ref_now.get(c, 0) for c in _vals)
                    _want = _vals[crop] / sum(_vals.values()) * _pool
                    target = int(round(target + _k("PORT_SHIFT", 0.35) * (_want - target)))
            have = counts_now.get(crop, 0) + seeds.get(crop, 0)
            need = max(0, target - have)
            cost = CROPS[crop]["seed"]
            cap = 14
            if crop == "WHEAT" and _k("WHEAT_WAVE", 20) <= day <= LAST_PLANT["WHEAT"]:
                need = max(need, len(sv["empty"]) - seeds.get("WHEAT", 0))
                cap = 40
            n = min(need, spendable // cost, cap)
            if n > 0:
                market.append(["BUY_SEED", crop, n])
                money -= cost * n
                spendable -= cost * n

    # sells: pace to town drain, but TIME them against the projected price
    shops = obs.get("town", {}).get("unlocked_shops", [])
    drain = _town_drain_per_day(shops, day)
    fc = st.get("fc") or {}
    st_sold = st.setdefault("sold_today", {})
    if st.get("sold_day") != day:
        st["sold_day"] = day
        st_sold.clear()
    sellable = [i for i in shed if i in MARKET_PARAMS and shed.get(i, 0) > 0]
    shed_total = sum(shed.get(k, 0) for k in shed)
    for item in sorted(sellable, key=lambda i: -_price(i, inv_mkt.get(i, I0)) * shed[i]):
        if len(market) >= 10:
            break
        n = shed[item]
        if item == "WHEAT" and active_animals > 0 and day < LAST_DAY:
            n = max(0, n - (feed_reserve if not liquidation else active_animals))
        if item == "WHEAT" and _k("CORNER2", 0) and not liquidation and day < 26:
            if _price("WHEAT", inv_mkt.get("WHEAT", I0)) < _k("CORNER2_ASK", 54):
                continue  # hold the corner until their buys pump the price
        if item == "WHEAT" and not liquidation and shed_total < 70:
            if _price("WHEAT", inv_mkt.get("WHEAT", I0)) < _k("WHEAT_HOLD", 32):
                continue  # hold: log-above curve means selling later loses nothing
        if item == "FERTILIZER" and not liquidation:
            n = max(0, n - 4)  # keep a few for application
        if n <= 0:
            continue
        # CASH VELOCITY: prices end up equal either way, but cash sold early
        # compounds into hands/seeds/animals. Measured: the leader banks 3.6x
        # our gross income over days 0-16 with the same farm.
        # INTRA-DAY SPREAD. Town shops consume every 4 turns and the centre
        # every 12, so the price recovers through the day. Dumping the whole
        # overnight harvest at h0-2 crashes our own quote; the leader spreads
        # 51% of its selling into h12-23 and realises a higher average.
        if _k("SPREAD", 1) and not liquidation and day < LIQ_DAY and hour < 22:
            left = max(1, 22 - hour)
            per = max(1, -(-n // left))
            n = min(n, per)
        if _k("SELL_ALL", 0) and not (item == "WHEAT" and active_animals > 0):
            market.append(["SELL", item, n])
            continue
        if liquidation or day == LAST_DAY or shed_total > 80:
            market.append(["SELL", item, n])
            continue
        p = MARKET_PARAMS[item]
        above = p["above_func"]
        if above in ("log", "log10"):
            market.append(["SELL", item, n])        # glut barely moves price
            continue
        inv0 = inv_mkt.get(item, I0)
        # will this glut ever clear? remaining town drain vs current excess
        excess = max(0, inv0 - I0) + n
        remaining_drain = drain.get(item, 0) * max(0, 28 - day)
        # visible opponent supply for this item (their farm is public)
        opp_supply = _opp_capacity(farms, player).get(item, 0)
        race = excess > remaining_drain * _k("RACE_DRAIN", 0.8) or opp_supply >= _k("RACE_OPP", 4)
        if race:
            # glut won't clear / opponent will flood: take today's price now
            market.append(["SELL", item, n])
            st_sold[item] = st_sold.get(item, 0) + n
            continue
        # monopoly-ish: pace to town drain, hold above a decaying floor
        allowance = drain.get(item, 0) * 1.3 + 4
        already = st_sold.get(item, 0)
        k = int(max(0, allowance - already))
        floor = MARKET_PARAMS[item]["base"] * max(0.15, _k("FLOOR_A", 0.55) - _k("FLOOR_B", 0.015) * day)
        kk = 0
        while kk < min(n, k) and _price(item, inv0 + kk) >= floor:
            kk += 1
        if kk > 0:
            market.append(["SELL", item, kk])
            st_sold[item] = already + kk
    # ORDER-SLOT PRIORITY. The engine walks market slots in lockstep across
    # both players: slot 0 for both, then slot 1... A sell at slot 6 is quoted
    # AFTER the rival's slot-0 sells have already pushed inventory up. Our
    # sells were built last (after hires/land/seeds/animals), so we were
    # systematically quoted worse every single turn. Put sells first.
    if _k("SELL_FIRST", 1):
        sells = [m for m in market if m and m[0] == "SELL"]
        others = [m for m in market if not (m and m[0] == "SELL")]
        market = sells + others
    return {"farmer": actions[0], "hands": actions[1:], "market": market[:10]}

