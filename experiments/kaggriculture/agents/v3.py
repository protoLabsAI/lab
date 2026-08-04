"""Kaggriculture agent v3 — role-based scheduler.

Fixes over v2: dedicated feeder units with pickup->feed->care->harvest->collect
circuits (animals never starve), zoned crop workers with persistent task
commitment (no routing thrash), feed-wheat reserve never sold, simpler
sell-everything-with-low-floor market policy, real final-day harvest push.
"""
import math

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
LAST_PLANT = {"WHEAT": 24, "CARROT": 25, "MELON": 18, "STRAWBERRY": 15, "TOMATO": 18}
TPD = 24
LAST_DAY = 29
SHED_CAP = 100

MAX_HANDS = 12
HIRE_COST_CAP = 90
LIQ_DAY = 28
MAX_ANIMALS = 13
ANIMAL_LAST_BUY = 12
STRAW_CAP = 45
MELON_CAP = 12
WHEAT_ROLLING = 6
SELL_FLOOR_FRAC = 0.30
FEED_STOP = {"GOOSE": 28, "COW": 27, "SHEEP": 26}

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


def _go(pos, target, do_action):
    """Move toward target, or perform do_action if already there."""
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
    shed_access = _shed_tiles(board)
    liquidation = day >= LIQ_DAY

    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
    n_units = len(units)
    while len(inventories) < n_units:
        inventories.append({})

    # shed-access tiles in LOCKED quadrants silently no-op PICKUP/DROP
    open_shed = [s for s in shed_access if tiles[s[1]][s[0]] != "LOCKED"]
    if not open_shed:
        open_shed = [shed_access[0]]

    def shed_d(p):
        return min(_dist(p, s) for s in open_shed)

    def nearest_shed(p):
        return min(open_shed, key=lambda s: _dist(p, s))

    # ---- survey ----
    plants, animals, empty_structs, empty_tiles, weeds = [], [], [], [], []
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
                    (animals if t.get("animal") else empty_structs).append((x, y, t))

    n_animals = len(animals)
    n_straw = sum(1 for _, _, t in plants if t["crop"] == "STRAWBERRY")
    n_melon = sum(1 for _, _, t in plants if t["crop"] == "MELON")
    unhoused = {a: shed.get(a, 0) + sum(inv.get(a, 0) for inv in inventories)
                for a in ANIMALS}
    total_unhoused = sum(unhoused.values())

    feed_need_today = sum(1 for _, _, t in animals
                          if not t["fed_today"] and day <= FEED_STOP.get(t["animal"], 27))
    active_animals = sum(1 for _, _, t in animals if day <= FEED_STOP.get(t["animal"], 27))
    feed_reserve = active_animals * 2

    # ---- roles ----
    n_feeders = 0
    if n_animals + total_unhoused > 0 and day < LAST_DAY:
        n_feeders = max((n_animals + 3) // 4, min(2, total_unhoused))
        n_feeders = min(n_feeders, n_units)
    feeder_ids = list(range(1, 1 + n_feeders))
    if n_feeders and n_units <= n_feeders:
        feeder_ids = list(range(n_units))
    worker_ids = [i for i in range(n_units) if i not in feeder_ids]

    actions = [None] * n_units

    # any unit carrying an animal must go place it (highest value action)
    for ui in range(n_units):
        if any(inventories[ui].get(a, 0) > 0 for a in ANIMALS):
            actions[ui] = _feeder_place_animal(units[ui], inventories[ui], shed,
                                               empty_structs, empty_tiles,
                                               open_shed, unhoused, board, tiles)

    # =====================================================
    # FEEDERS: wheat pickup -> animal circuit -> fertilize -> drop
    # =====================================================
    # split animals among feeders by index
    animal_list = sorted(animals, key=lambda a: (a[1], a[0]))
    for fi, ui in enumerate(feeder_ids):
        if actions[ui] is not None:
            continue
        pos = units[ui]
        inv = inventories[ui]
        mine = [a for j, a in enumerate(animal_list) if j % max(1, n_feeders) == fi]

        # place unhoused animals first (highest value: get production started)
        act = None
        if total_unhoused > 0:
            act = _feeder_place_animal(pos, inv, shed, empty_structs, empty_tiles,
                                       open_shed, unhoused, board, tiles)
        if act is None:
            act = _feeder_circuit(pos, inv, shed, mine, day, hour, plants,
                                  nearest_shed(pos), feed_need_today, liquidation)
        actions[ui] = act

    # =====================================================
    # WORKERS: zoned crop work
    # =====================================================
    # build crop tasks
    tasks = []  # (prio, x, y, action)
    for x, y, t in plants:
        crop = t["crop"]
        c = CROPS.get(crop)
        if c is None:
            continue
        age = day - t["planted_day"]
        watered = t["watered_today"]
        streak = t["consecutive_unwatered"]
        ongoing = c["ongoing"]
        # water: everything daily, critical first
        if not watered and not (day == LAST_DAY and age < c["first"]):
            tasks.append((0 if streak >= 1 else 2, x, y, ["WATER"]))
        # harvest
        if t["yield_units"] > 0 and age >= c["first"]:
            if ongoing:
                if t["yield_units"] >= 2 or liquidation or t["yield_units"] >= 1 and hour >= 16:
                    tasks.append((3, x, y, ["HARVEST"]))
            else:
                ready_age = HARVEST_AGE.get(crop, c["max_day"])
                step_now = day * TPD + hour
                decaying = t["max_lifespan_step"] >= 0 and step_now >= t["max_lifespan_step"] - 6
                if decaying or day == LAST_DAY:
                    tasks.append((1, x, y, ["HARVEST"]))
                elif age >= ready_age and (watered or age > c["max_day"]):
                    tasks.append((2, x, y, ["HARVEST"]))

    # plant tasks
    budget = {c: n for c, n in seeds.items() if n > 0 and c in CROPS}
    planted = {"STRAWBERRY": n_straw, "MELON": n_melon,
               "WHEAT": sum(1 for _, _, t in plants if t["crop"] == "WHEAT")}
    if day < LIQ_DAY - 1 and hour <= 20:
        reserve_inner = n_animals + total_unhoused < MAX_ANIMALS and day <= ANIMAL_LAST_BUY
        for (x, y) in sorted(empty_tiles, key=shed_d):
            if reserve_inner and shed_d((x, y)) <= 1:
                continue
            crop = _pick_crop(budget, day, planted)
            if crop is None:
                break
            budget[crop] -= 1
            planted[crop] = planted.get(crop, 0) + 1
            tasks.append((3, x, y, ["PLANT", crop]))

    for (x, y) in weeds:
        if day <= LAST_DAY - 3:
            tasks.append((5, x, y, ["DIG"]))

    _assign_workers(actions, units, worker_ids, tasks, inventories, open_shed,
                    day, hour, board, _STATE.setdefault(player, {}))

    # idle/courier
    for ui in range(n_units):
        if actions[ui] is not None:
            continue
        pos = units[ui]
        carrying = sum(v for k, v in inventories[ui].items())
        d_home = shed_d(pos)
        if carrying > 0 and (day == LAST_DAY or hour >= TPD - d_home - 3 or carrying >= 12):
            actions[ui] = _go(pos, nearest_shed(pos), ["DROP"])
        else:
            actions[ui] = ["PASS"]

    # =====================================================
    # MARKET
    # =====================================================
    market = []

    if day == 0 and hour == 0:
        market = [["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                  ["BUY_ANIMAL", "COW", 3], ["BUY_ANIMAL", "SHEEP", 1],
                  ["BUY_SEED", "MELON", 7], ["BUY_SEED", "WHEAT", 10],
                  ["BUY_PRODUCT", "WHEAT", 6]]
        return {"farmer": actions[0], "hands": actions[1:], "market": market}

    # hires: crew sized to the farm we WANT to run (plants + plantable), not
    # just current tasks — otherwise few plants -> few hands -> few plants.
    if day < LAST_DAY - 1:
        n_plantable = len(empty_tiles) if day < LIQ_DAY - 2 else 0
        workload = (len(plants) + n_plantable) * 1.3 + active_animals * 4.5
        target_units = 1 + int(workload // 15)
        if day >= 1 and len(plants) + len(empty_tiles) >= 15:
            target_units = max(target_units, 6)
        target_units = min(target_units, MAX_HANDS + 1)
        hires_today = farm.get("hires_today", 0)
        while n_units + len([m for m in market if m == ["HIRE"]]) < target_units and len(market) < 6:
            cost = _fib(hires_today)
            if cost > HIRE_COST_CAP or money - cost < 30 or hour > 20:
                break
            market.append(["HIRE"])
            money -= cost
            hires_today += 1

    # land
    n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if hour <= 2 and 1 <= day:
        land_prices = [1000, 2000, 4000]
        if n_extra < 2 and day <= 16 and money - land_prices[n_extra] > 400:
            market.append(["BUY_LAND"])
            money -= land_prices[n_extra]
        elif n_extra == 2 and 6 <= day <= 12 and money > 7000:
            market.append(["BUY_LAND"])
            money -= 4000

    # seeds
    if day < LIQ_DAY - 2 and len(market) < 8:
        want = {}
        room = max(0, len(empty_tiles) - sum(seeds.get(c, 0) for c in CROPS))
        spendable = max(0, money - 120 - feed_reserve * 5)
        if room > 0:
            if day <= LAST_PLANT["STRAWBERRY"] and n_straw + seeds.get("STRAWBERRY", 0) < STRAW_CAP:
                n = min(room, STRAW_CAP - n_straw - seeds.get("STRAWBERRY", 0), 12,
                        int(spendable * 0.6) // 100)
                want["STRAWBERRY"] = n
                room -= n
                spendable -= n * 100
            if room > 0 and day <= LAST_PLANT["MELON"] and n_melon + seeds.get("MELON", 0) < MELON_CAP:
                n = min(room, MELON_CAP - n_melon - seeds.get("MELON", 0), 6,
                        int(spendable * 0.6) // 80)
                want["MELON"] = n
                room -= n
                spendable -= n * 80
            # wheat fills every remaining tile — cheap, fast, and feeds animals
            if room > 0 and day <= LAST_PLANT["WHEAT"]:
                n = min(room, spendable // 10)
                want["WHEAT"] = n
                room -= n
                spendable -= n * 10
            if room > 0 and day <= LAST_PLANT["CARROT"] and spendable >= 20:
                want["CARROT"] = min(room, spendable // 20)
        for crop, n in want.items():
            while n > 0 and money - CROPS[crop]["seed"] * n <= 150:
                n -= 1
            if n > 0 and len(market) < 9:
                market.append(["BUY_SEED", crop, n])
                money -= CROPS[crop]["seed"] * n

    # feed wheat
    wheat_stock = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
    if active_animals > 0 and wheat_stock < feed_reserve and len(market) < 9:
        wp = _price("WHEAT", inv_mkt.get("WHEAT", I0) - 1)
        if wp <= 50 and money > 120:
            n_buy = min(feed_reserve + 2 - wheat_stock, int(max(0, money - 100) // wp))
            if n_buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", n_buy])
                money -= n_buy * wp

    # animals (up to 2/day while cash allows)
    if (1 <= day <= ANIMAL_LAST_BUY and hour <= 6
            and n_animals + total_unhoused < MAX_ANIMALS and len(market) < 8):
        n_sheep = sum(1 for _, _, t in animals if t["animal"] == "SHEEP") + unhoused.get("SHEEP", 0)
        for _ in range(2):
            if n_animals + total_unhoused >= MAX_ANIMALS:
                break
            choice = None
            if _price("WOOL", inv_mkt.get("WOOL", I0)) >= 130 and n_sheep < 5 and day <= 12:
                choice = "SHEEP"
            elif _price("MILK", inv_mkt.get("MILK", I0)) >= 100 and day <= 12:
                choice = "COW"
            elif _price("EGG", inv_mkt.get("EGG", I0)) >= 45 and day <= 16:
                choice = "GOOSE"
            if choice and money - ANIMALS[choice]["cost"] > 500:
                market.append(["BUY_ANIMAL", choice, 1])
                money -= ANIMALS[choice]["cost"]
                total_unhoused += 1
                if choice == "SHEEP":
                    n_sheep += 1
            else:
                break

    # sells
    sellable = [i for i in shed if i in MARKET_PARAMS and shed.get(i, 0) > 0]
    for item in sorted(sellable, key=lambda i: -_price(i, inv_mkt.get(i, I0)) * shed[i]):
        if len(market) >= 10:
            break
        n = shed[item]
        if item == "WHEAT" and not liquidation:
            n = max(0, n - feed_reserve)
        if n <= 0:
            continue
        if liquidation or sum(shed.get(k, 0) for k in shed) > SHED_CAP - 20:
            market.append(["SELL", item, n])
            continue
        floor = max(2, SELL_FLOOR_FRAC * MARKET_PARAMS[item]["base"])
        k = 0
        inv0 = inv_mkt.get(item, I0)
        while k < n and _price(item, inv0 + k) >= floor:
            k += 1
        if k > 0:
            market.append(["SELL", item, k])

    return {"farmer": actions[0], "hands": actions[1:], "market": market[:10]}


def _feeder_place_animal(pos, inv, shed, empty_structs, empty_tiles, open_shed,
                         unhoused, board, tiles):
    """Get unhoused animals onto structures: build, pickup, place."""
    def shed_d(p):
        return min(_dist(p, s) for s in open_shed)

    # carrying an animal? place it
    for a_kind in ("COW", "SHEEP", "GOOSE"):
        if inv.get(a_kind, 0) > 0:
            want_kind = ANIMALS[a_kind]["structure"]
            slots = [(x, y) for x, y, t in empty_structs if t["kind"] == want_kind]
            if slots:
                target = min(slots, key=lambda s: _dist(pos, s))
                return _go(pos, target, ["PLACE", a_kind])
            # need a structure: build on nearest empty tile to shed
            spots = [p for p in empty_tiles if shed_d(p) <= 3]
            if spots:
                target = min(spots, key=lambda s: _dist(pos, s))
                build = "BUILD_COOP" if want_kind == "COOP" else "BUILD_PASTURE"
                return _go(pos, target, [build])
            return None
    # animal in shed? go pick it up
    for a_kind in ("COW", "SHEEP", "GOOSE"):
        if shed.get(a_kind, 0) > 0:
            target = min(open_shed, key=lambda s: _dist(pos, s))
            return _go(pos, target, ["PICKUP", a_kind, 1])
    return None


def _feeder_circuit(pos, inv, shed, mine, day, hour, plants, home, feed_need, liquidation):
    """Service assigned animals: feed -> care -> harvest -> collect fertilizer."""
    wheat = inv.get("WHEAT", 0)
    # do we need wheat first?
    my_unfed = [a for a in mine if not a[2]["fed_today"]
                and day <= FEED_STOP.get(a[2]["animal"], 27)]
    if my_unfed and wheat == 0:
        if shed.get("WHEAT", 0) > 0:
            k = min(max(len(my_unfed), 4), shed.get("WHEAT", 0), 10)
            return _go(pos, home, ["PICKUP", "WHEAT", k])
        # no wheat anywhere: wait for the buy order to land
        my_unfed = []

    # priority pass over my animals
    best = None
    for x, y, t in mine:
        a = ANIMALS[t["animal"]]
        acts = []
        active = day <= FEED_STOP.get(t["animal"], 27)
        if active and not t["fed_today"] and wheat > 0:
            acts.append((0 if t["consecutive_unfed"] >= 1 else 1, ["FEED"]))
        if t["yield_units"] >= a["max_held"] - 1 or (t["yield_units"] > 0 and (liquidation or hour >= 15)):
            acts.append((2, ["HARVEST"]))
        if active and not t["cared_today"] and t["fed_today"]:
            acts.append((3, ["CARE"]))
        elif active and not t["cared_today"] and wheat == 0:
            acts.append((3, ["CARE"]))
        if t.get("fertilizer_available") and not liquidation:
            acts.append((4, ["COLLECT_FERTILIZER"]))
        for prio, act in acts:
            cost = prio * 100 + _dist(pos, (x, y))
            if best is None or cost < best[0]:
                best = (cost, (x, y), act)
    if best is not None:
        return _go(pos, best[1], best[2])

    # fertilize nearby producing plants with carried fertilizer
    if inv.get("FERTILIZER", 0) > 0 and not liquidation:
        targets = []
        for x, y, t in plants:
            c = CROPS.get(t["crop"])
            if c is None or t.get("fertilized_until_day", -1) >= day:
                continue
            age = day - t["planted_day"]
            if c["ongoing"] and age >= c["first"] - 2:
                targets.append((x, y))
            elif t["crop"] == "MELON" and 4 <= age <= 9:
                targets.append((x, y))
        if targets:
            target = min(targets, key=lambda p: _dist(pos, p))
            return _go(pos, target, ["FERTILIZE"])

    # drop off whatever we're carrying
    if sum(inv.values()) > 0:
        return _go(pos, home, ["DROP"])
    return None  # fall through to idle logic


def _assign_workers(actions, units, worker_ids, tasks, inventories, shed_access,
                    day, hour, board, st):
    """Priority-greedy with commitment: units keep their previous target."""
    free = [ui for ui in worker_ids if actions[ui] is None]
    if not free:
        return
    task_by_tile = {}
    for prio, x, y, act in sorted(tasks, key=lambda t: t[0]):
        task_by_tile.setdefault((x, y), []).append((prio, act))

    prev = st.get("commit", {})
    commit = {}
    claimed = set()

    # 1) honor previous commitments still valid
    for ui in list(free):
        tgt = prev.get(ui)
        if tgt and tgt in task_by_tile and tgt not in claimed:
            prio, act = task_by_tile[tgt][0]
            pos = units[ui]
            actions[ui] = _go(pos, tgt, act)
            if pos != tgt:
                commit[ui] = tgt   # still walking; keep the commitment
            claimed.add(tgt)
            free.remove(ui)

    # 2) assign remaining tasks by priority, nearest unit
    flat = []
    for (x, y), lst in task_by_tile.items():
        if (x, y) in claimed:
            continue
        prio, act = lst[0]
        flat.append((prio, x, y, act))
    flat.sort(key=lambda t: t[0])
    for prio, x, y, act in flat:
        if not free:
            break
        if day == LAST_DAY:
            # only take if we can act and return home in time
            pass
        best = min(free, key=lambda ui: _dist(units[ui], (x, y)))
        pos = units[best]
        if day == LAST_DAY:
            d_task = _dist(pos, (x, y))
            d_home = min(_dist((x, y), s) for s in shed_access)
            carrying = sum(inventories[best].values())
            if hour + d_task + 1 + d_home > TPD - 2 and (carrying > 0 or act[0] == "HARVEST"):
                continue
        actions[best] = _go(pos, (x, y), act)
        if pos != (x, y):
            commit[best] = (x, y)
        free.remove(best)
    st["commit"] = commit


def _pick_crop(budget, day, planted):
    if (budget.get("STRAWBERRY", 0) > 0 and day <= LAST_PLANT["STRAWBERRY"]
            and planted.get("STRAWBERRY", 0) < STRAW_CAP):
        return "STRAWBERRY"
    if (budget.get("MELON", 0) > 0 and day <= LAST_PLANT["MELON"]
            and planted.get("MELON", 0) < MELON_CAP):
        return "MELON"
    for c in ("WHEAT", "CARROT", "TOMATO"):
        if budget.get(c, 0) > 0 and day <= LAST_PLANT.get(c, 0):
            return c
    return None


def _entry(obs):
    return agent(obs)
