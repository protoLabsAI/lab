"""Kaggriculture agent v1 — crop planner.

Architecture:
- Day-level plan: hires (workload-scaled, Fibonacci-aware), land purchases,
  seed purchases (exact, no stranded capital), crop mix (melon core with a
  concurrency cap + wheat/carrot staples chosen by marginal price).
- Turn-level scheduler: priority tasks (critical water > harvest-at-peak >
  bonus-window water > plant > maintenance water > dig weeds), greedy
  nearest-unit assignment, Manhattan movement.
- Market: sell shed contents every turn with a price floor guard, full
  liquidation from day 28; return-to-shed logistics on the final day.

No animals, no fertilizer, no ongoing crops in v1.
"""

# ---- engine constants (mirrored from kaggriculture.py) ----
CROPS = {
    "WHEAT":  {"seed": 10, "first": 2,  "max_day": 4,  "max_yield": 6},
    "CARROT": {"seed": 20, "first": 2,  "max_day": 3,  "max_yield": 4},
    "MELON":  {"seed": 80, "first": 10, "max_day": 12, "max_yield": 6},
}
# age at which daily-watered unfertilized yield stops growing -> harvest then
HARVEST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10}
# latest planting day such that harvest lands by day 29
LAST_PLANT_DAY = {"WHEAT": 25, "CARROT": 26, "MELON": 18}

BASE_PRICE = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

TURNS_PER_DAY = 24
LAST_DAY = 29
SHED_CAP = 100

MELON_TILE_CAP = 14        # max concurrent melon plants
MELON_INV_STOP = 10150     # stop planting melons if market inventory above this
SELL_FLOOR_FRAC = 0.40     # hold produce priced below this fraction of base
LIQUIDATION_DAY = 28       # sell everything from this day on
MAX_HANDS_PER_DAY = 10
HIRE_COST_CAP = 60         # never pay more than this for one hand

_STATE = {}  # per-player sticky assignment memory


def _shed_tiles(board):
    h = board // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(pos, target):
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
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


def _pick_crop_for_tile(budget, day, melon_count, inv_mkt, prices):
    """Choose a crop to plant from available seed budget."""
    order = []
    if (budget.get("MELON", 0) > 0 and day <= LAST_PLANT_DAY["MELON"]
            and melon_count < MELON_TILE_CAP
            and inv_mkt.get("MELON", 10000) < MELON_INV_STOP):
        order.append("MELON")
    # staples by current price attractiveness
    staples = []
    for c in ("CARROT", "WHEAT"):
        if budget.get(c, 0) > 0 and day <= LAST_PLANT_DAY[c]:
            staples.append((prices.get(c, BASE_PRICE[c]) / BASE_PRICE[c], c))
    staples.sort(reverse=True)
    order.extend(c for _, c in staples)
    return order[0] if order else None


def _seed_wishlist(room, day, melon_count, seeds, inv_mkt, prices):
    """How many seeds to buy this turn (planted next turns)."""
    want = {"MELON": 0, "CARROT": 0, "WHEAT": 0}
    room = max(0, room - sum(seeds.get(c, 0) for c in CROPS))
    if room <= 0:
        return want
    if (day <= LAST_PLANT_DAY["MELON"]
            and inv_mkt.get("MELON", 10000) < MELON_INV_STOP):
        m = min(room, max(0, MELON_TILE_CAP - melon_count - seeds.get("MELON", 0)))
        want["MELON"] = m
        room -= m
    if room > 0:
        carrot_ok = day <= LAST_PLANT_DAY["CARROT"]
        wheat_ok = day <= LAST_PLANT_DAY["WHEAT"]
        c_score = prices.get("CARROT", 35) / 35 if carrot_ok else -1
        w_score = prices.get("WHEAT", 25) / 25 if wheat_ok else -1
        if c_score < 0 and w_score < 0:
            return want
        if c_score >= w_score:
            want["CARROT"] = (room + 1) // 2
            want["WHEAT"] = room - want["CARROT"] if wheat_ok else 0
        else:
            want["WHEAT"] = (room + 1) // 2
            want["CARROT"] = room - want["WHEAT"] if carrot_ok else 0
    return want


def agent(obs):
    player = obs.get("player", 0)
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    farms = obs.get("farms", [])
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private", {}) or {}
    market_obs = obs.get("market", {}) or {}
    prices = market_obs.get("prices", {}) or {}
    inv_mkt = market_obs.get("inventory", {}) or {}
    tiles = farm["tiles"]
    board = len(tiles)
    money = farm["money"]
    seeds = dict(private.get("seeds", {}) or {})
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", [{}])
    shed_access = _shed_tiles(board)

    st = _STATE.setdefault(player, {"assign": {}})

    units = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", [])]
    n_units = len(units)

    # ---- survey the farm ----
    my_plants = []       # (x, y, tile)
    empty_tiles = []     # unlocked empties
    weed_tiles = []
    for y in range(board):
        for x in range(board):
            t = tiles[y][x]
            if t is None:
                empty_tiles.append((x, y))
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    my_plants.append((x, y, t))
                elif k == "WEED":
                    weed_tiles.append((x, y))

    melon_count = sum(1 for _, _, t in my_plants if t["crop"] == "MELON")

    # ---- build task list: (priority, x, y, action) lower prio = more urgent ----
    tasks = []
    for x, y, t in my_plants:
        crop = t["crop"]
        c = CROPS.get(crop)
        if c is None:
            continue  # shouldn't happen in v1 (we only plant known crops)
        age = day - t["planted_day"]
        watered = t["watered_today"]
        unwatered_streak = t["consecutive_unwatered"]
        window_start = (c["max_day"] + 1) // 2
        in_window = window_start <= age <= c["max_day"]
        will_harvest_today = age >= HARVEST_AGE[crop] or day == LAST_DAY

        # watering
        if not watered:
            if unwatered_streak >= 1 and not (will_harvest_today and age >= c["first"]):
                tasks.append((0, x, y, ["WATER"]))          # dies tonight otherwise
            elif in_window and not will_harvest_today:
                tasks.append((2, x, y, ["WATER"]))          # +1 yield
            elif in_window and will_harvest_today:
                tasks.append((2, x, y, ["WATER"]))          # water then harvest later
            # else: maintenance water skippable today (streak == 0), leave it

        # harvesting
        if t["yield_units"] > 0 and age >= c["first"]:
            past_peak = age > c["max_day"]
            ready = age >= HARVEST_AGE[crop]
            if past_peak:
                tasks.append((1, x, y, ["HARVEST"]))        # decaying — grab now
            elif ready and (watered or not in_window):
                tasks.append((1, x, y, ["HARVEST"]))
            elif day == LAST_DAY and hour >= 4:
                tasks.append((1, x, y, ["HARVEST"]))

    # ---- planting plan ----
    plantable = {c: n for c, n in seeds.items() if n > 0 and c in CROPS}
    plant_tasks_budget = dict(plantable)
    if day <= LAST_DAY - 3 and hour <= 21:
        for (x, y) in empty_tiles:
            crop = _pick_crop_for_tile(plant_tasks_budget, day, melon_count, inv_mkt, prices)
            if crop is None:
                break
            plant_tasks_budget[crop] -= 1
            if crop == "MELON":
                melon_count += 1
            tasks.append((3, x, y, ["PLANT", crop]))

    for (x, y) in weed_tiles:
        if day <= LAST_DAY - 4:
            tasks.append((5, x, y, ["DIG"]))

    # ---- final-day logistics: get produce home ----
    carrying = [sum(inv.values()) for inv in inventories]
    force_home = day == LAST_DAY
    home_deadline = TURNS_PER_DAY - 2  # be shed-adjacent by hour 22 on day 29

    # ---- assign units to tasks ----
    tasks.sort(key=lambda t: t[0])
    actions = [None] * n_units
    claimed = set()
    prev = st["assign"]
    new_assign = {}

    for prio, x, y, act in tasks:
        if (x, y, act[0]) in claimed:
            continue
        best, best_cost = None, None
        for ui, pos in enumerate(units):
            if actions[ui] is not None:
                continue
            if force_home:
                # only take the task if we can do it and still reach the shed
                d_task = _dist(pos, (x, y))
                d_home = min(_dist((x, y), s) for s in shed_access)
                if hour + d_task + 1 + d_home > home_deadline and carrying[ui] + (
                    1 if act[0] == "HARVEST" else 0) > 0:
                    continue
            cost = _dist(pos, (x, y))
            if prev.get(ui) == (x, y):
                cost -= 0.5  # stickiness: keep walking to the same target
            if best is None or cost < best_cost:
                best, best_cost = ui, cost
        if best is None:
            continue
        pos = units[best]
        if pos == (x, y):
            actions[best] = act
        else:
            mv = _step_toward(pos, (x, y))
            actions[best] = [mv] if mv else ["PASS"]
        new_assign[best] = (x, y)
        claimed.add((x, y, act[0]))

    # idle units: bring inventory home late in the day, else pass
    for ui, pos in enumerate(units):
        if actions[ui] is not None:
            continue
        if carrying[ui] > 0 and (force_home or hour >= TURNS_PER_DAY - 1 - min(
                _dist(pos, s) for s in shed_access) - 2 or carrying[ui] >= 20):
            target = min(shed_access, key=lambda s: _dist(pos, s))
            if pos == target or pos in shed_access:
                actions[ui] = ["DROP"]
            else:
                mv = _step_toward(pos, target)
                actions[ui] = [mv] if mv else ["PASS"]
        else:
            actions[ui] = ["PASS"]

    st["assign"] = new_assign

    # ---- market orders ----
    market = []

    # 1) hire hands (morning, workload-scaled)
    if day < LAST_DAY:
        n_water = sum(1 for p in tasks if p[3][0] == "WATER")
        n_harv = sum(1 for p in tasks if p[3][0] == "HARVEST")
        n_plant = sum(1 for p in tasks if p[3][0] == "PLANT")
        demand = 1.7 * (n_water + n_harv + n_plant) + 0.3 * len(my_plants)
        capacity = (TURNS_PER_DAY - hour) * n_units
        hires_today = farm.get("hires_today", 0)
        reserve = 150
        n_new = 0
        while (capacity < demand
               and hires_today + n_new < MAX_HANDS_PER_DAY
               and len(market) < 4):
            cost = _fib(hires_today + n_new)
            if cost > HIRE_COST_CAP or money - cost < reserve:
                break
            market.append(["HIRE"])
            money -= cost
            n_new += 1
            capacity += (TURNS_PER_DAY - hour - 1)

    # 2) land purchase (morning only)
    if hour == 0 and day <= 20:
        n_extra = len(farm.get("unlocked_quadrants", ["NW"])) - 1
        land_prices = [1000, 2000, 4000]
        if n_extra < 3:
            cost = land_prices[n_extra]
            if len(empty_tiles) < 6 and money - cost > 400:
                market.append(["BUY_LAND"])
                money -= cost

    # 3) seed purchases: buy what we can plant soon, minus what we hold
    if day <= LAST_DAY - 3:
        room = len(empty_tiles) + len(weed_tiles)
        want = _seed_wishlist(room, day, melon_count, seeds, inv_mkt, prices)
        for crop, n in want.items():
            cost = CROPS[crop]["seed"] * n
            if n > 0 and money - cost > 100:
                market.append(["BUY_SEED", crop, n])
                money -= cost

    # 4) sells
    liquidate = day >= LIQUIDATION_DAY
    for item, n in sorted(shed.items()):
        if n <= 0 or len(market) >= 10:
            continue
        price = prices.get(item, 0)
        base = BASE_PRICE.get(item, 1)
        if liquidate or price >= SELL_FLOOR_FRAC * base or sum(shed.values()) > SHED_CAP - 15:
            market.append(["SELL", item, n])

    farmer_action = actions[0] if actions else ["PASS"]
    hands_actions = actions[1:]
    return {"farmer": farmer_action, "hands": hands_actions, "market": market[:10]}
