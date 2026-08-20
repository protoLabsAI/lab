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

MAX_HANDS = 12
HIRE_COST_CAP = 90
LIQ_DAY = 28
MAX_ANIMALS = 13
ANIMAL_LAST_BUY = 12
STRAW_CAP = 45
MELON_CAP = 12
SELL_FLOOR_FRAC = 0.30
FEED_STOP = {"GOOSE": 28, "COW": 28, "SHEEP": 28}

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
            jobs.append((0 if t["consecutive_unwatered"] >= 1 else 1, (x, y), ("WATER", None)))
        if c["ongoing"]:
            if t["yield_units"] >= 2 or (t["yield_units"] > 0 and (
                    liquidation or day >= LAST_DAY - 1)):
                jobs.append((2, (x, y), ("HARVEST", None)))
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
        if t["yield_units"] > 0:
            prio = 1 if t["yield_units"] >= a["max_held"] - 1 or day >= LAST_DAY - 1 else 2
            jobs.append((prio, (x, y), ("HARVEST", None)))

    for x, y in sv["weeds"]:
        if day <= LAST_DAY - 4:
            jobs.append((4, (x, y), ("DIG", None)))
    return jobs


def _plan_planting(sv, day, seeds, n_animals_total):
    """(tile, crop) planting assignments, nearest tiles first."""
    if day >= LIQ_DAY - 1:
        return []
    budget = {c: seeds.get(c, 0) for c in CROPS}
    counts = {
        "STRAWBERRY": sum(1 for _, _, t in sv["plants"] if t["crop"] == "STRAWBERRY"),
        "MELON": sum(1 for _, _, t in sv["plants"] if t["crop"] == "MELON"),
    }
    out = []
    reserve_inner = n_animals_total < MAX_ANIMALS and day <= ANIMAL_LAST_BUY
    for (x, y) in sorted(sv["empty"], key=lambda p: _dist(p, (4.5, 4.5))):
        if reserve_inner and _dist((x, y), (4.5, 4.5)) <= 1.5:
            continue
        crop = None
        if (budget.get("STRAWBERRY", 0) > 0 and day <= LAST_PLANT["STRAWBERRY"]
                and counts.get("STRAWBERRY", 0) < STRAW_CAP):
            crop = "STRAWBERRY"
        elif (budget.get("MELON", 0) > 0 and day <= LAST_PLANT["MELON"]
                and counts.get("MELON", 0) < MELON_CAP):
            crop = "MELON"
        elif budget.get("WHEAT", 0) > 0 and day <= LAST_PLANT["WHEAT"]:
            crop = "WHEAT"
        elif budget.get("CARROT", 0) > 0 and day <= LAST_PLANT["CARROT"]:
            crop = "CARROT"
        if crop is None:
            break
        budget[crop] -= 1
        counts[crop] = counts.get(crop, 0) + 1
        out.append(((x, y), crop))
    return out


def _fert_chains(sv, day, liquidation):
    """Pair each collectable fertilizer with a nearby fertilizable plant."""
    if liquidation:
        return []
    sources = [(x, y) for x, y, t in sv["animals"] if t.get("fertilizer_available")]
    targets = []
    for x, y, t in sv["plants"]:
        c = CROPS.get(t["crop"])
        if c is None or t.get("fertilized_until_day", -1) >= day:
            continue
        age = day - t["planted_day"]
        remaining = LAST_DAY - day
        if c["ongoing"] and age >= c["first"] - 3 and remaining >= 2:
            targets.append((x, y))
        elif t["crop"] == "MELON" and 2 <= age <= 9:
            targets.append((x, y))
    chains = []
    used = set()
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
            chains.append([(s, ("COLLECT_FERTILIZER", None)),
                           (best[1], ("FERTILIZE", None))])
        else:
            chains.append([(s, ("COLLECT_FERTILIZER", None))])
    return chains


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


def _route_units(units, hour, jobs, plant_jobs, chains, fert_chains=()):
    """Greedy nearest-neighbor routing. Returns routes[ui] = [(tile, (op, arg)), ...]."""
    budget_turns = max(1, TPD - hour - 1)
    routes = [[] for _ in units]
    loads = [0.0] * len(units)
    positions = [tuple(p) for p in units]

    def assign_chain(chain):
        best, best_cost = None, None
        for ui in range(len(units)):
            d = _dist(positions[ui], chain[0][0])
            cost = loads[ui] + d
            if loads[ui] > budget_turns - len(chain) - d:
                continue
            if best is None or cost < best_cost:
                best, best_cost = ui, cost
        if best is None:
            return False
        for tile, opa in chain:
            routes[best].append((tile, opa))
            loads[best] += _dist(positions[best], tile) + 1
            positions[best] = tile
        return True

    # placement chains first (time-critical), assigned whole
    for chain in chains:
        if not assign_chain(chain):
            best = min(range(len(units)), key=lambda ui: loads[ui])
            for tile, opa in chain:
                routes[best].append((tile, opa))
                loads[best] += _dist(positions[best], tile) + 1
                positions[best] = tile

    # urgent singles first (watering/feeding/harvest beat new planting)
    singles = sorted(jobs, key=lambda j: j[0])
    urgent = [j for j in singles if j[0] <= 2]
    rest = [j for j in singles if j[0] > 2]

    def assign_single(prio, tile, opa):
        best, best_cost = None, None
        for ui in range(len(units)):
            if loads[ui] >= budget_turns:
                continue
            d = _dist(positions[ui], tile)
            cost = loads[ui] + d
            if best is None or cost < best_cost:
                best, best_cost = ui, cost
        if best is None:
            return
        routes[best].append((tile, opa))
        loads[best] += _dist(positions[best], tile) + 1
        positions[best] = tile

    for prio, tile, opa in urgent:
        assign_single(prio, tile, opa)

    # fertilizer chains after survival work, before new planting
    for chain in fert_chains:
        assign_chain(chain)

    # plant compounds: PLANT then WATER at same tile
    for tile, crop in plant_jobs:
        best, best_cost = None, None
        for ui in range(len(units)):
            if loads[ui] >= budget_turns - 1:
                continue
            d = _dist(positions[ui], tile)
            cost = loads[ui] + d
            if best is None or cost < best_cost:
                best, best_cost = ui, cost
        if best is None:
            continue
        routes[best].append((tile, ("PLANT", crop)))
        routes[best].append((tile, ("WATER", None)))
        loads[best] += _dist(positions[best], tile) + 2
        positions[best] = tile

    for prio, tile, opa in rest:
        assign_single(prio, tile, opa)
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
    feed_reserve = active_animals * 2 + 4 if active_animals else 0

    st = _STATE.setdefault(player, {})
    if st.get("day") != day:
        st.clear()
        st["day"] = day
        st["routes"] = {}

    # ---- (re)build routes when crew size changes or at day start ----
    if st.get("crew") != n_units:
        st["crew"] = n_units
        jobs = _build_day_jobs(sv, day, liquidation)
        plant_jobs = _plan_planting(sv, day, seeds, n_animals + total_unhoused)
        chains = _placement_chains(sv, shed, open_shed)
        fchains = _fert_chains(sv, day, liquidation)
        routes = _route_units(units, hour, jobs, plant_jobs, chains, fchains)
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
        if carrying > 0 and (day == LAST_DAY or hour >= TPD - d_home - 3 or carrying >= 10):
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
        market = [["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                  ["BUY_LAND"],
                  ["BUY_ANIMAL", "COW", 2],
                  ["BUY_SEED", "MELON", 6], ["BUY_SEED", "WHEAT", 20],
                  ["BUY_PRODUCT", "WHEAT", 5]]
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
        if day >= LAST_DAY - 1:
            target_units = min(target_units, 7)
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
    if active_animals > 0 and wheat_stock < feed_reserve:
        wp = _price("WHEAT", inv_mkt.get("WHEAT", I0) - 1)
        if wp <= 55:
            keep = 100 if day <= 3 else 40
            n_buy = min(feed_reserve + 4 - wheat_stock, int(max(0, money - keep) // wp))
            if n_buy > 0:
                market.append(["BUY_PRODUCT", "WHEAT", n_buy])
                money -= n_buy * wp

    # 3) land (needs crew to work it)
    n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
    if hour <= 2 and day >= 1 and n_units >= 5:
        land_prices = [1000, 2000, 4000]
        if n_extra < 2 and day <= 16 and money - land_prices[n_extra] > (200 if n_extra == 0 else 800):
            market.append(["BUY_LAND"])
            money -= land_prices[n_extra]
        elif n_extra == 2 and 6 <= day <= 12 and money > 8000:
            market.append(["BUY_LAND"])
            money -= 4000

    # 4) seeds
    if day < LIQ_DAY - 2 and len(market) < 8:
        want = {}
        room = max(0, len(sv["empty"]) - sum(seeds.get(c, 0) for c in CROPS))
        spendable = max(0, money - 150)
        n_straw = sum(1 for _, _, t in sv["plants"] if t["crop"] == "STRAWBERRY")
        n_melon = sum(1 for _, _, t in sv["plants"] if t["crop"] == "MELON")
        if room > 0 and day <= LAST_PLANT["STRAWBERRY"] and n_straw + seeds.get("STRAWBERRY", 0) < STRAW_CAP:
            n = min(room, STRAW_CAP - n_straw - seeds.get("STRAWBERRY", 0), 12,
                    int(spendable * 0.6) // 100)
            if n > 0:
                want["STRAWBERRY"] = n
                room -= n
                spendable -= n * 100
        if room > 0 and day <= LAST_PLANT["MELON"] and n_melon + seeds.get("MELON", 0) < MELON_CAP:
            n = min(room, MELON_CAP - n_melon - seeds.get("MELON", 0), 6,
                    int(spendable * 0.6) // 80)
            if n > 0:
                want["MELON"] = n
                room -= n
                spendable -= n * 80
        if room > 0 and day <= LAST_PLANT["WHEAT"]:
            n = min(room, spendable // 10)
            if n > 0:
                want["WHEAT"] = n
                room -= n
                spendable -= n * 10
        for crop, n in want.items():
            if n > 0 and len(market) < 9:
                market.append(["BUY_SEED", crop, n])
                money -= CROPS[crop]["seed"] * n

    # 5) animals last (only truly spare cash)
    if (1 <= day <= ANIMAL_LAST_BUY and hour <= 6
            and n_animals + total_unhoused < MAX_ANIMALS and len(market) < 9):
        n_sheep = sum(1 for _, _, t in sv["animals"] if t["animal"] == "SHEEP") + unhoused.get("SHEEP", 0)
        for _ in range(2):
            if n_animals + total_unhoused >= MAX_ANIMALS or len(market) >= 9:
                break
            choice = None
            if _price("WOOL", inv_mkt.get("WOOL", I0)) >= 130 and n_sheep < 5 and day <= 12:
                choice = "SHEEP"
            elif _price("MILK", inv_mkt.get("MILK", I0)) >= 100 and day <= 12:
                choice = "COW"
            elif _price("EGG", inv_mkt.get("EGG", I0)) >= 45 and day <= 16:
                choice = "GOOSE"
            if choice and money - ANIMALS[choice]["cost"] > 800:
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
        if item == "WHEAT" and active_animals > 0 and day < LAST_DAY:
            n = max(0, n - (feed_reserve if not liquidation else active_animals))
        if n <= 0:
            continue
        if liquidation or sum(shed.get(k, 0) for k in shed) > 80:
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

