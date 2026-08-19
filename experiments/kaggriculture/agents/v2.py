"""Kaggriculture agent v2 — full economy: animals + strawberries + fertilizer.

Blueprint (from instrumenting a ~$174k expert replay):
- Day 0: all-in opening — hires, cows+sheep, melon+wheat seeds, feed wheat.
- Days 1-12: strawberry wave, more animals, land (NE, SW), scale to ~12 hands.
- Steady state: water everything daily, feed+care every animal, collect
  fertilizer and apply to producing strawberries / melon windows, harvest at
  peak, sell with marginal-price awareness.
- Days 28-29: liquidation, return-to-shed logistics.
"""
import math

# ---- engine constants (mirrored from kaggriculture.py) ----
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
HARVEST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}
LAST_PLANT_DAY = {"WHEAT": 25, "CARROT": 26, "MELON": 18, "STRAWBERRY": 15, "TOMATO": 18}
TURNS_PER_DAY = 24
LAST_DAY = 29
SHED_CAP = 100

# ---- policy knobs ----
MAX_HANDS = 12
HIRE_COST_CAP = 90
LIQUIDATION_DAY = 28
MAX_ANIMALS = 14
ANIMAL_BUY_LAST_DAY = 14
STRAWBERRY_CAP = 40
MELON_CAP = 12
SELL_FLOOR = {  # hold-below fraction of base while producing
    "WHEAT": 0.0, "CARROT": 0.45, "TOMATO": 0.45, "STRAWBERRY": 0.50,
    "MELON": 0.40, "EGG": 0.0, "MILK": 0.50, "WOOL": 0.45, "FERTILIZER": 0.55,
}
FEED_STOP = {"GOOSE": 28, "COW": 27, "SHEEP": 26}  # last day feeding pays off

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


def _units_sellable(item, inv, floor_price):
    """How many units can be sold before price drops below floor_price."""
    n = 0
    while n < 200 and _price(item, inv + n) >= floor_price:
        n += 1
    return n


def _shed_tiles(board):
    h = board // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


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


def agent(obs):
    player = obs.get("player", 0)
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private", {}) or {}
    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
    inv_mkt = (obs.get("market", {}) or {}).get("inventory", {}) or {}
    tiles = farm["tiles"]
    board = len(tiles)
    money = farm["money"]
    seeds = dict(private.get("seeds", {}) or {})
    shed = dict(private.get("shed", {}) or {})
    inventories = [dict(i) for i in private.get("inventories", [{}])]
    shed_access = _shed_tiles(board)
    st = _STATE.setdefault(player, {"assign": {}})

    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
    n_units = len(units)
    while len(inventories) < n_units:
        inventories.append({})

    # ---- survey ----
    plants, animals, structures_empty, empty_tiles, weeds = [], [], [], [], []
    for y in range(board):
        for x in range(board):
            t = tiles[y][x]
            if t is None:
                empty_tiles.append((x, y))
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    plants.append((x, y, t))
                elif k == "WEED":
                    weeds.append((x, y))
                elif k in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        animals.append((x, y, t))
                    else:
                        structures_empty.append((x, y, t))

    n_animals = len(animals)
    n_straw = sum(1 for _, _, t in plants if t["crop"] == "STRAWBERRY")
    n_melon = sum(1 for _, _, t in plants if t["crop"] == "MELON")

    # animals waiting in shed or unit inventories to be housed
    shed_animals = {a: shed.get(a, 0) for a in ANIMALS}
    carried_animals = {a: sum(inv.get(a, 0) for inv in inventories) for a in ANIMALS}
    unhoused = {a: shed_animals[a] + carried_animals[a] for a in ANIMALS}
    total_unhoused = sum(unhoused.values())

    # empty tiles sorted by distance to shed centre (for placement decisions)
    def shed_d(p):
        return min(_dist(p, s) for s in shed_access)
    empty_by_center = sorted(empty_tiles, key=shed_d)

    # ---- task list: (priority, x, y, action, meta) ----
    tasks = []
    liquidation = day >= LIQUIDATION_DAY

    for x, y, t in plants:
        crop = t["crop"]
        c = CROPS.get(crop)
        if c is None:
            continue
        age = day - t["planted_day"]
        watered = t["watered_today"]
        streak = t["consecutive_unwatered"]
        ongoing = c["ongoing"]
        if ongoing:
            harvest_ready = t["yield_units"] >= 2 or (t["yield_units"] > 0 and (
                liquidation or hour >= 18))
            dying_soon = False
        else:
            ready_age = HARVEST_AGE.get(crop, c["max_day"])
            harvest_ready = t["yield_units"] > 0 and age >= c["first"] and (
                age > c["max_day"] or (age >= ready_age and (watered or day == LAST_DAY)))
            dying_soon = t["max_lifespan_step"] >= 0 and (
                day * TURNS_PER_DAY + hour) >= t["max_lifespan_step"] - 4
        if t["yield_units"] > 0 and (dying_soon or (day == LAST_DAY and age >= c["first"])):
            tasks.append((1, x, y, ["HARVEST"], None))
        elif harvest_ready:
            tasks.append((1 if not ongoing else 3, x, y, ["HARVEST"], None))

        if not watered and not (day == LAST_DAY and hour > 16):
            if streak >= 1:
                tasks.append((0, x, y, ["WATER"], None))
            else:
                # worth watering daily: bonus windows + ongoing production
                tasks.append((2, x, y, ["WATER"], None))

        # fertilize: ongoing crops in production, or one-time in bonus window
        if t.get("fertilized_until_day", -1) < day and day < LIQUIDATION_DAY:
            if ongoing and age >= c["first"] - 2:
                tasks.append((4, x, y, ["FERTILIZE"], "FERT"))
            elif crop == "MELON" and 4 <= age <= 9:
                tasks.append((4, x, y, ["FERTILIZE"], "FERT"))

    for x, y, t in animals:
        a = ANIMALS[t["animal"]]
        fed = t["fed_today"]
        unfed_streak = t["consecutive_unfed"]
        if not fed and day <= FEED_STOP.get(t["animal"], 27):
            prio = 0 if unfed_streak >= 1 else 2
            tasks.append((prio, x, y, ["FEED"], "WHEAT"))
        if not t["cared_today"] and day <= FEED_STOP.get(t["animal"], 27) and fed is not None:
            tasks.append((3, x, y, ["CARE"], None))
        if t["yield_units"] > 0:
            full = t["yield_units"] >= a["max_held"] - 1
            tasks.append((2 if full or liquidation else 3, x, y, ["HARVEST"], None))
        if t.get("fertilizer_available"):
            tasks.append((4, x, y, ["COLLECT_FERTILIZER"], None))

    # place unhoused animals / build structures for them
    if total_unhoused > 0:
        # place on empty matching structures first
        struct_slots = {"COOP": [], "PASTURE": []}
        for x, y, t in structures_empty:
            struct_slots[t["kind"]].append((x, y))
        need_pasture = unhoused["COW"] + unhoused["SHEEP"] - len(struct_slots["PASTURE"])
        need_coop = unhoused["GOOSE"] - len(struct_slots["COOP"])
        for kind, need in (("PASTURE", need_pasture), ("COOP", need_coop)):
            for (x, y) in empty_by_center:
                if need <= 0:
                    break
                if shed_d((x, y)) <= 3:
                    tasks.append((3, x, y, ["BUILD_%s" % ("COOP" if kind == "COOP" else "PASTURE")], None))
                    need -= 1
        for a_kind in ("COW", "SHEEP", "GOOSE"):
            slots = struct_slots[ANIMALS[a_kind]["structure"]]
            for i in range(min(unhoused[a_kind], len(slots))):
                x, y = slots[i]
                tasks.append((2, x, y, ["PLACE", a_kind], a_kind))

    # ---- planting plan ----
    budget = {c: n for c, n in seeds.items() if n > 0 and c in CROPS}
    planted_now = {"STRAWBERRY": n_straw, "MELON": n_melon}
    if day < LIQUIDATION_DAY - 1 and hour <= 21:
        # reserve center tiles for structures
        for (x, y) in empty_by_center:
            if shed_d((x, y)) <= 1 and n_animals + total_unhoused < MAX_ANIMALS and day <= ANIMAL_BUY_LAST_DAY:
                continue  # keep inner ring free for pastures
            crop = _pick_crop(budget, day, planted_now)
            if crop is None:
                break
            budget[crop] -= 1
            planted_now[crop] = planted_now.get(crop, 0) + 1
            tasks.append((3, x, y, ["PLANT", crop], None))

    for (x, y) in weeds:
        if day <= LAST_DAY - 4:
            tasks.append((6, x, y, ["DIG"], None))

    # ---- assignment ----
    tasks.sort(key=lambda t: t[0])
    actions = [None] * n_units
    claimed = set()
    prev = st["assign"]
    new_assign = {}
    carrying = [sum(inv.values()) for inv in inventories]
    force_home = day == LAST_DAY
    home_deadline = TURNS_PER_DAY - 2

    wheat_carried = [inventories[i].get("WHEAT", 0) for i in range(n_units)]
    fert_carried = [inventories[i].get("FERTILIZER", 0) for i in range(n_units)]

    for prio, x, y, act, need in tasks:
        key = (x, y, act[0])
        if key in claimed:
            continue
        best, best_cost, best_via = None, None, None
        for ui, pos in enumerate(units):
            if actions[ui] is not None:
                continue
            via = None
            # resource-dependent tasks
            if need == "WHEAT" and wheat_carried[ui] <= 0:
                if shed.get("WHEAT", 0) <= 0:
                    continue
                via = min(shed_access, key=lambda s: _dist(pos, s))
            elif need == "FERT" and fert_carried[ui] <= 0:
                continue  # only units already carrying fertilizer fertilize
            elif need in ANIMALS and inventories[ui].get(need, 0) <= 0:
                if shed.get(need, 0) <= 0:
                    continue
                via = min(shed_access, key=lambda s: _dist(pos, s))
            cost = (_dist(pos, via) + _dist(via, (x, y))) if via else _dist(pos, (x, y))
            if force_home:
                d_home_after = min(_dist((x, y), s) for s in shed_access)
                if hour + cost + 1 + d_home_after > home_deadline and (
                        carrying[ui] > 0 or act[0] == "HARVEST"):
                    continue
            if prev.get(ui) == (x, y):
                cost -= 0.5
            if best is None or cost < best_cost:
                best, best_cost, best_via = ui, cost, via
        if best is None:
            continue
        pos = units[best]
        if best_via is not None and pos != best_via and not (
                need == "WHEAT" and wheat_carried[best] > 0):
            # go fetch the resource first
            if pos == best_via:
                pass
            mv = _step_toward(pos, best_via)
            actions[best] = [mv] if mv else ["PASS"]
        elif best_via is not None and pos == best_via:
            # standing at shed: pick up the resource
            if need == "WHEAT":
                k = min(8, shed.get("WHEAT", 0))
                actions[best] = ["PICKUP", "WHEAT", k]
                wheat_carried[best] += k
                shed["WHEAT"] = shed.get("WHEAT", 0) - k
            else:
                actions[best] = ["PICKUP", need, 1]
                shed[need] = shed.get(need, 0) - 1
        elif pos == (x, y):
            actions[best] = act
            if act[0] == "FEED":
                wheat_carried[best] -= 1
            elif act[0] == "FERTILIZE":
                fert_carried[best] -= 1
            elif act[0] == "COLLECT_FERTILIZER":
                fert_carried[best] += 1
        else:
            mv = _step_toward(pos, (x, y))
            actions[best] = [mv] if mv else ["PASS"]
        new_assign[best] = (x, y)
        claimed.add(key)

    # idle: courier duty / drop inventory
    for ui, pos in enumerate(units):
        if actions[ui] is not None:
            continue
        d_home = min(_dist(pos, s) for s in shed_access)
        if carrying[ui] > 0 and (force_home or hour >= TURNS_PER_DAY - d_home - 3
                                 or carrying[ui] >= 15):
            target = min(shed_access, key=lambda s: _dist(pos, s))
            if pos in shed_access:
                actions[ui] = ["DROP"]
            else:
                mv = _step_toward(pos, target)
                actions[ui] = [mv] if mv else ["PASS"]
        else:
            actions[ui] = ["PASS"]
    st["assign"] = new_assign

    # ---- market ----
    market = []

    # opening: day 0 hour 0 — all-in
    if day == 0 and hour == 0:
        market = [["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                  ["BUY_ANIMAL", "COW", 3], ["BUY_ANIMAL", "SHEEP", 1],
                  ["BUY_SEED", "MELON", 7], ["BUY_SEED", "WHEAT", 8],
                  ["BUY_PRODUCT", "WHEAT", 6]]
        return {"farmer": actions[0], "hands": actions[1:], "market": market}

    # hires: workload-scaled
    if day < LAST_DAY:
        n_work = sum(1 for t in tasks if t[3][0] in
                     ("WATER", "FEED", "CARE", "HARVEST", "PLANT", "COLLECT_FERTILIZER"))
        demand = 1.8 * n_work
        capacity = (TURNS_PER_DAY - hour) * n_units
        hires_today = farm.get("hires_today", 0)
        reserve = 200 if day > 2 else 20
        while capacity < demand and hires_today < MAX_HANDS and len(market) < 5:
            cost = _fib(hires_today)
            if cost > HIRE_COST_CAP or money - cost < reserve:
                break
            market.append(["HIRE"])
            money -= cost
            hires_today += 1
            capacity += (TURNS_PER_DAY - hour - 1)

    # land: NE then SW (skip SE unless rich mid-game)
    n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if hour <= 2 and day >= 1:
        land_prices = [1000, 2000, 4000]
        if n_extra < 2 and day <= 16 and money - land_prices[n_extra] > 300:
            market.append(["BUY_LAND"])
            money -= land_prices[n_extra]
        elif n_extra == 2 and day <= 13 and money > 6000:
            market.append(["BUY_LAND"])
            money -= 4000

    # animals: keep buying while early and prices healthy
    if (day >= 1 and day <= ANIMAL_BUY_LAST_DAY and hour <= 4
            and n_animals + total_unhoused < MAX_ANIMALS and len(market) < 8):
        choice = None
        if _price("MILK", inv_mkt.get("MILK", I0)) >= 110:
            choice = "COW"
        elif _price("WOOL", inv_mkt.get("WOOL", I0)) >= 130:
            choice = "SHEEP"
        elif _price("EGG", inv_mkt.get("EGG", I0)) >= 45:
            choice = "GOOSE"
        if choice and money - ANIMALS[choice]["cost"] > 400:
            market.append(["BUY_ANIMAL", choice, 1])
            money -= ANIMALS[choice]["cost"]

    # seeds: strawberries + rolling melon + wheat
    if day < LIQUIDATION_DAY - 2 and len(market) < 8:
        want = {}
        room = len(empty_tiles)
        held = sum(seeds.get(c, 0) for c in CROPS)
        room = max(0, room - held)
        if room > 0:
            if day <= LAST_PLANT_DAY["STRAWBERRY"] and n_straw + seeds.get("STRAWBERRY", 0) < STRAWBERRY_CAP:
                want["STRAWBERRY"] = min(room, 6)
                room -= want["STRAWBERRY"]
            if room > 0 and day <= LAST_PLANT_DAY["MELON"] and n_melon + seeds.get("MELON", 0) < MELON_CAP:
                want["MELON"] = min(room, 4)
                room -= want["MELON"]
            if room > 0 and day <= LAST_PLANT_DAY["WHEAT"]:
                want["WHEAT"] = min(room, 6)
        for crop, n in want.items():
            cost = CROPS[crop]["seed"] * n
            reserve = 150
            while n > 0 and money - CROPS[crop]["seed"] * n <= reserve:
                n -= 1
            if n > 0 and len(market) < 9:
                market.append(["BUY_SEED", crop, n])
                money -= CROPS[crop]["seed"] * n

    # feed wheat: keep shed stocked for animals
    feed_need = sum(1 for _, _, t in animals if day <= FEED_STOP.get(t["animal"], 27))
    wheat_stock = shed.get("WHEAT", 0) + sum(wc for wc in wheat_carried)
    if feed_need > 0 and wheat_stock < feed_need * 2 and len(market) < 9:
        wheat_price = _price("WHEAT", inv_mkt.get("WHEAT", I0) - 1)
        if wheat_price <= 45 and money > 150:
            n_buy = min(feed_need * 2 - wheat_stock, int((money - 100) // wheat_price))
            if n_buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", n_buy])
                money -= n_buy * wheat_price

    # sells: marginal-price aware
    sellable_items = [i for i in shed if i in MARKET_PARAMS and shed.get(i, 0) > 0]
    for item in sorted(sellable_items,
                       key=lambda i: -_price(i, inv_mkt.get(i, I0)) * shed.get(i, 0)):
        n = shed.get(item, 0)
        if len(market) >= 10:
            break
        if item == "WHEAT" and not liquidation:
            keep = feed_need * 2
            n = max(0, n - keep)
        if n <= 0:
            continue
        if liquidation:
            market.append(["SELL", item, n])
            continue
        floor = SELL_FLOOR.get(item, 0.4) * MARKET_PARAMS[item]["base"]
        sellable = _units_sellable(item, inv_mkt.get(item, I0), max(2, floor))
        shed_total = sum(shed.get(k, 0) for k in shed)
        if shed_total > SHED_CAP - 20:
            sellable = n  # shed pressure: dump
        n = min(n, sellable)
        if n > 0:
            market.append(["SELL", item, n])

    return {"farmer": actions[0], "hands": actions[1:], "market": market[:10]}


def _pick_crop(budget, day, planted_now):
    if (budget.get("STRAWBERRY", 0) > 0 and day <= LAST_PLANT_DAY["STRAWBERRY"]
            and planted_now.get("STRAWBERRY", 0) < STRAWBERRY_CAP):
        return "STRAWBERRY"
    if (budget.get("MELON", 0) > 0 and day <= LAST_PLANT_DAY["MELON"]
            and planted_now.get("MELON", 0) < MELON_CAP):
        return "MELON"
    for c in ("WHEAT", "CARROT", "TOMATO"):
        if budget.get(c, 0) > 0 and day <= LAST_PLANT_DAY.get(c, 0):
            return c
    return None


# kaggle_environments uses the last callable in the file
def _agent_entry(obs):
    return agent(obs)
