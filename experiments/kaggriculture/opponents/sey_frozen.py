"""Deterministic, versioned local opponents for paired candidate evaluation.

These policies deliberately use no module-level episode state.  Calling either
agent with the same observation always returns the same action, which makes
them useful regression targets across processes and seat swaps.
"""

from __future__ import annotations

from typing import Any


CROP_DATA = {
    "WHEAT": {"seed": 10, "harvest_day": 4},
    "CARROT": {"seed": 20, "harvest_day": 3},
    "TOMATO": {"seed": 50, "harvest_day": 8},
}
PRODUCTS = ("WOOL", "MILK", "EGG", "TOMATO", "CARROT", "WHEAT")
SHED_CAPACITY = 100
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _step_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = position
    tx, ty = target
    dx, dy = tx - x, ty - y
    if abs(dx) >= abs(dy) and dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    return ["PASS"]


def _shed_access(farm: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    half = len(farm["tiles"]) // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _assign_tasks(
    obs: dict[str, Any],
    tasks: list[tuple[int, int, int, list[Any]]],
) -> tuple[list[Any], list[list[Any]]]:
    """Greedily match unique tasks to units with deterministic tie breaks."""
    player = int(obs["player"])
    farm = obs["farms"][player]
    positions = [tuple(farm["farmer"]), *(tuple(pos) for pos in farm.get("hands", []))]
    inventories = obs.get("private", {}).get("inventories", [])
    access = _shed_access(farm)
    assignments: dict[int, tuple[tuple[int, int], list[Any]]] = {}
    available = set(range(len(positions)))

    # Inventory routing comes before farm work.  Production goes home, while
    # purchased geese and feed move from the shed to their destinations.
    animal_tiles: list[tuple[int, int, dict[str, Any]]] = []
    empty_structures: dict[str, list[tuple[int, int]]] = {"COOP": [], "PASTURE": []}
    fertilizable: list[tuple[int, int, dict[str, Any]]] = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("animal"):
                animal_tiles.append((x, y, tile))
            elif isinstance(tile, dict) and tile.get("kind") in empty_structures:
                empty_structures[str(tile["kind"])].append((x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                fertilizable.append((x, y, tile))

    reserved_structures: set[tuple[int, int]] = set()
    for unit in sorted(list(available)):
        inventory = inventories[unit] if unit < len(inventories) else {}
        position = positions[unit]
        wheat_count = int(inventory.get("WHEAT", 0))
        fertilizer_count = int(inventory.get("FERTILIZER", 0))
        carried_product = sum(int(inventory.get(item, 0)) for item in PRODUCTS)
        carried_animal = next(
            (animal for animal in ("GOOSE", "COW", "SHEEP") if int(inventory.get(animal, 0))),
            None,
        )
        if carried_animal:
            candidates = [
                tile
                for tile in empty_structures[ANIMAL_STRUCTURE[carried_animal]]
                if tile not in reserved_structures
            ]
            if candidates:
                target = min(candidates, key=lambda tile: (_distance(position, tile), tile))
                reserved_structures.add(target)
                action = ["PLACE", carried_animal] if position == target else _step_toward(position, target)
                assignments[unit] = (target, action)
                available.remove(unit)
                continue
        if wheat_count:
            hungry = [
                (x, y)
                for x, y, tile in animal_tiles
                if not bool(tile.get("fed_today", False))
            ]
            if hungry:
                target = min(hungry, key=lambda tile: (_distance(position, tile), tile))
                action = ["FEED"] if position == target else _step_toward(position, target)
                assignments[unit] = (target, action)
                available.remove(unit)
                continue
        if fertilizer_count:
            targets = [
                (x, y, tile)
                for x, y, tile in fertilizable
                if int(tile.get("fertilized_until_day", -1)) < int(obs.get("day", 0))
            ]
            if targets:
                x, y, _tile = min(
                    targets,
                    key=lambda item: (
                        0 if item[2].get("crop") == "TOMATO" else 1,
                        _distance(position, (item[0], item[1])),
                        item[0],
                        item[1],
                    ),
                )
                target = (x, y)
                action = ["FERTILIZE"] if position == target else _step_toward(position, target)
                assignments[unit] = (target, action)
                available.remove(unit)
                continue
        if carried_product:
            target = min(access, key=lambda tile: (_distance(position, tile), tile))
            action = ["DROP"] if position == target else _step_toward(position, target)
            assignments[unit] = (target, action)
            available.remove(unit)

    pending = list(tasks)
    while pending and available:
        _, unit, task_index = min(
            (
                (priority, _distance(positions[unit], (x, y)), unit, x, y),
                unit,
                index,
            )
            for index, (priority, x, y, _action) in enumerate(pending)
            for unit in available
        )
        priority, x, y, operation = pending.pop(task_index)
        del priority
        position = positions[unit]
        assignments[unit] = ((x, y), operation if position == (x, y) else _step_toward(position, (x, y)))
        available.remove(unit)

    actions = [assignments.get(index, (position, ["PASS"]))[1] for index, position in enumerate(positions)]
    return actions[0], actions[1:]


def _crop_tasks(
    obs: dict[str, Any],
    crop_for_tile,
    reserved: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int, int, list[Any]]]:
    player = int(obs["player"])
    farm = obs["farms"][player]
    seeds = {name: int(obs["private"].get("seeds", {}).get(name, 0)) for name in CROP_DATA}
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    tasks: list[tuple[int, int, int, list[Any]]] = []
    empty: list[tuple[int, int]] = []
    reserved = reserved or set()

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if (x, y) in reserved:
                continue
            if tile is None:
                empty.append((x, y))
            elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                tasks.append((2, x, y, ["DIG"]))
            elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = str(tile.get("crop"))
                age = day - int(tile.get("planted_day", day))
                ready = int(tile.get("yield_units", 0)) > 0 and age >= CROP_DATA.get(crop, {}).get("harvest_day", 99)
                if not bool(tile.get("watered_today", False)) and day < 29:
                    priority = -1 if int(tile.get("consecutive_unwatered", 0)) else 0
                    tasks.append((priority, x, y, ["WATER"]))
                elif ready and (day < 29 or hour < 15):
                    tasks.append((1, x, y, ["HARVEST"]))

    if hour >= 15 or day >= 25:
        return tasks
    for x, y in empty:
        crop = crop_for_tile(x, y)
        if seeds.get(crop, 0) <= 0:
            continue
        tasks.append((4, x, y, ["PLANT", crop]))
        seeds[crop] -= 1
    return tasks


def _market_sales(obs: dict[str, Any]) -> list[list[Any]]:
    shed = obs["private"].get("shed", {})
    prices = obs["market"].get("prices", {})
    orders: list[list[Any]] = []
    for item in sorted(PRODUCTS, key=lambda name: (-int(prices.get(name, 0)), name)):
        count = int(shed.get(item, 0))
        if count:
            orders.append(["SELL", item, count])
    return orders


def _hire_orders(obs: dict[str, Any], target: int) -> list[list[str]]:
    farm = obs["farms"][int(obs["player"])]
    if int(obs.get("hour", 0)) > 1:
        return []
    missing = max(0, target - len(farm.get("hands", [])))
    return [["HIRE"] for _ in range(missing)]


def crop_specialist(obs: dict[str, Any]) -> dict[str, Any]:
    """A stable 25-tile carrot rotation that is stronger than ``starter``."""
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs["private"]
    day = int(obs.get("day", 0))
    active = sum(
        1
        for row in farm["tiles"]
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )
    seeds = int(private.get("seeds", {}).get("CARROT", 0))
    market = _market_sales(obs) + _hire_orders(obs, 9)
    if day < 24 and len(market) < 10:
        gap = max(0, 25 - active - seeds)
        affordable = max(0, (int(farm["money"]) - 250) // CROP_DATA["CARROT"]["seed"])
        count = min(8, gap, affordable)
        if count:
            market.append(["BUY_SEED", "CARROT", count])
    tasks = _crop_tasks(obs, lambda _x, _y: "CARROT")
    farmer, hands = _assign_tasks(obs, tasks)
    return {"farmer": farmer, "hands": hands, "market": market[:10]}


def diversified_baseline(obs: dict[str, Any]) -> dict[str, Any]:
    """A fixed crop mix plus three cared-for geese and daily feed logistics."""
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs["private"]
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    coop_targets = {(3, 3), (3, 4), (4, 3)}
    tasks: list[tuple[int, int, int, list[Any]]] = []
    occupied_geese = 0
    empty_coops = 0
    hungry = 0
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if (x, y) in coop_targets and tile is None:
                tasks.append((-3, x, y, ["BUILD_COOP"]))
            elif isinstance(tile, dict) and tile.get("kind") == "COOP":
                if tile.get("animal") == "GOOSE":
                    occupied_geese += 1
                    if not bool(tile.get("fed_today", False)):
                        hungry += 1
                        tasks.append((-2, x, y, ["FEED"]))
                    elif int(tile.get("yield_units", 0)) > 0:
                        tasks.append((-1, x, y, ["HARVEST"]))
                    elif not bool(tile.get("cared_today", False)):
                        tasks.append((1, x, y, ["CARE"]))
                else:
                    empty_coops += 1

    shed = private.get("shed", {})
    inventories = private.get("inventories", [])
    carried_geese = sum(int(inv.get("GOOSE", 0)) for inv in inventories)
    carried_wheat = sum(int(inv.get("WHEAT", 0)) for inv in inventories)
    access = set(_shed_access(farm))
    positions = [tuple(farm["farmer"]), *(tuple(pos) for pos in farm.get("hands", []))]
    if empty_coops and int(shed.get("GOOSE", 0)) > 0 and not carried_geese:
        for x, y in sorted(access):
            if (x, y) in positions:
                tasks.append((-4, x, y, ["PICKUP", "GOOSE", 1]))
                break
    if hungry and int(shed.get("WHEAT", 0)) > 0 and not carried_wheat:
        for x, y in sorted(access):
            if (x, y) in positions:
                tasks.append((-4, x, y, ["PICKUP", "WHEAT", hungry]))
                break

    def crop_for_tile(x: int, y: int) -> str:
        return "TOMATO" if (x + 2 * y) % 5 == 0 else "CARROT"

    tasks.extend(_crop_tasks(obs, crop_for_tile, coop_targets))
    market = _market_sales(obs) + _hire_orders(obs, 10)
    owned_geese = occupied_geese + empty_coops + int(shed.get("GOOSE", 0)) + carried_geese
    if day == 0 and owned_geese < 3 and len(market) < 10:
        market.append(["BUY_ANIMAL", "GOOSE", 3 - owned_geese])
    wheat_available = int(shed.get("WHEAT", 0)) + carried_wheat
    if occupied_geese and wheat_available < occupied_geese and hour <= 1 and len(market) < 10:
        market.append(["BUY_PRODUCT", "WHEAT", occupied_geese - wheat_available])

    if day < 23 and len(market) < 10:
        active_by_crop = {crop: 0 for crop in CROP_DATA}
        for row in farm["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop = tile.get("crop")
                    if crop in active_by_crop:
                        active_by_crop[crop] += 1
        seed_stock = private.get("seeds", {})
        targets = {"CARROT": 18, "TOMATO": 4}
        for crop in ("CARROT", "TOMATO"):
            gap = max(0, targets[crop] - active_by_crop[crop] - int(seed_stock.get(crop, 0)))
            reserve = 500
            affordable = max(0, (int(farm["money"]) - reserve) // CROP_DATA[crop]["seed"])
            count = min(6, gap, affordable)
            if count and len(market) < 10:
                market.append(["BUY_SEED", crop, count])

    farmer, hands = _assign_tasks(obs, tasks)
    return {"farmer": farmer, "hands": hands, "market": market[:10]}


def animal_specialist(obs: dict[str, Any]) -> dict[str, Any]:
    """Mixed livestock policy that closes the feed/fertilizer production loop.

    The policy starts with two geese, a cow, and a sheep, then scales to seven
    animals after cash flow stabilizes.  Animal manure is carried directly to
    tomatoes (then other crops), avoiding two otherwise-wasted shed trips.
    """
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs["private"]
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    money = int(farm["money"])
    structure_targets = {
        (3, 3): "COOP",
        (3, 4): "COOP",
        (4, 3): "COOP",
        (2, 2): "PASTURE",
        (2, 3): "PASTURE",
        (2, 4): "PASTURE",
        (3, 2): "PASTURE",
    }
    reserved = set(structure_targets)
    tasks: list[tuple[int, int, int, list[Any]]] = []
    occupied: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        "GOOSE": [],
        "COW": [],
        "SHEEP": [],
    }
    empty_structures = {"COOP": 0, "PASTURE": 0}

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            target_structure = structure_targets.get((x, y))
            if target_structure and tile is None:
                tasks.append((-6, x, y, [f"BUILD_{target_structure}"]))
                continue
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal in occupied:
                occupied[str(animal)].append((x, y, tile))
                if not bool(tile.get("fed_today", False)):
                    # FEED is deliberately handled only by inventory routing;
                    # assigning this task to a wheat-less unit would be a
                    # silent no-op that steals capacity from crop care.
                    continue
                elif not bool(tile.get("cared_today", False)):
                    tasks.append((-2, x, y, ["CARE"]))
                elif int(tile.get("yield_units", 0)) > 0:
                    tasks.append((-1, x, y, ["HARVEST"]))
                elif day <= 21 and bool(tile.get("fertilizer_available", False)):
                    tasks.append((0, x, y, ["COLLECT_FERTILIZER"]))
            elif tile.get("kind") in empty_structures:
                empty_structures[str(tile["kind"])] += 1

    shed = private.get("shed", {})
    inventories = private.get("inventories", [])
    positions = [tuple(farm["farmer"]), *(tuple(pos) for pos in farm.get("hands", []))]
    access_positions = [tile for tile in sorted(_shed_access(farm)) if tile in positions]
    carried = {
        item: sum(int(inventory.get(item, 0)) for inventory in inventories)
        for item in ("GOOSE", "COW", "SHEEP", "WHEAT", "FERTILIZER")
    }

    # Pull each animal type out of the shed only when a matching structure is
    # already ready.  Distinct access positions let several carriers depart in
    # parallel without relying on shared mutable state.
    pickup_slot = 0
    for animal in ("GOOSE", "COW", "SHEEP"):
        structure = ANIMAL_STRUCTURE[animal]
        if (
            empty_structures[structure] > 0
            and int(shed.get(animal, 0)) > 0
            and carried[animal] == 0
            and access_positions
        ):
            x, y = access_positions[pickup_slot % len(access_positions)]
            tasks.append((-5, x, y, ["PICKUP", animal, 1]))
            pickup_slot += 1

    all_animals = [entry for entries in occupied.values() for entry in entries]
    hungry = sum(1 for _x, _y, tile in all_animals if not bool(tile.get("fed_today", False)))
    if hungry and int(shed.get("WHEAT", 0)) > 0 and carried["WHEAT"] == 0 and access_positions:
        x, y = access_positions[pickup_slot % len(access_positions)]
        tasks.append((-5, x, y, ["PICKUP", "WHEAT", min(hungry, int(shed["WHEAT"]))]))
        pickup_slot += 1

    needs_fertilizer = sum(
        1
        for row in farm["tiles"]
        for tile in row
        if isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and int(tile.get("fertilized_until_day", -1)) < day
    )
    if (
        needs_fertilizer
        and int(shed.get("FERTILIZER", 0)) > 0
        and carried["FERTILIZER"] == 0
        and access_positions
    ):
        x, y = access_positions[pickup_slot % len(access_positions)]
        tasks.append((-4, x, y, ["PICKUP", "FERTILIZER", min(6, needs_fertilizer, int(shed["FERTILIZER"]))]))

    def crop_for_tile(x: int, y: int) -> str:
        selector = (3 * x + 5 * y) % 10
        if selector < 5:
            return "CARROT"
        if selector < 8:
            return "TOMATO"
        return "WHEAT"

    tasks.extend(_crop_tasks(obs, crop_for_tile, reserved))

    # Keep feed wheat instead of reflexively selling it.  Other production is
    # liquidated continuously to avoid the shed cap and premium-product gluts.
    market = [order for order in _market_sales(obs) if order[1] != "WHEAT"]
    # Keep a small, demand-backed fertilizer reserve for the pickup task above
    # and sell everything beyond it.  Unit actions execute before market
    # orders, so a successful pickup removes the reserve before this sale is
    # processed and the quantities remain consistent.
    fertilizer_reserve = min(6, needs_fertilizer)
    fertilizer_surplus = max(0, int(shed.get("FERTILIZER", 0)) - fertilizer_reserve)
    if fertilizer_surplus:
        market.append(["SELL", "FERTILIZER", fertilizer_surplus])
    unlocked = len(farm.get("unlocked_quadrants", ["NW"]))
    workload = len(all_animals) * 3 + sum(
        1
        for row in farm["tiles"]
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )
    hand_target = 10 if workload < 30 else 12
    market.extend(_hire_orders(obs, hand_target))

    budget = money
    if day >= 7 and day <= 15 and unlocked == 1 and money >= 5_000 and len(market) < 10:
        market.append(["BUY_LAND"])
        budget -= 1_000

    desired_animals = {"GOOSE": 2, "COW": 1, "SHEEP": 1}
    if day >= 7 and money >= 4_000:
        desired_animals = {"GOOSE": 3, "COW": 2, "SHEEP": 2}
    animal_cost = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
    if day <= 15:
        for animal in ("GOOSE", "COW", "SHEEP"):
            owned = (
                len(occupied[animal])
                + int(shed.get(animal, 0))
                + carried[animal]
            )
            gap = max(0, desired_animals[animal] - owned)
            affordable = max(0, (budget - 700) // animal_cost[animal])
            count = min(gap, affordable)
            if count and len(market) < 10:
                market.append(["BUY_ANIMAL", animal, count])
                budget -= count * animal_cost[animal]

    wheat_available = int(shed.get("WHEAT", 0)) + carried["WHEAT"]
    feed_target = max(4, len(all_animals) * (2 if day < 27 else 1))
    if all_animals and wheat_available < feed_target and hour <= 2 and len(market) < 10:
        count = feed_target - wheat_available
        if budget >= count * int(obs["market"]["prices"].get("WHEAT", 25)):
            market.append(["BUY_PRODUCT", "WHEAT", count])

    empty_demand = {crop: 0 for crop in CROP_DATA}
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if tile is None and (x, y) not in reserved:
                empty_demand[crop_for_tile(x, y)] += 1
    # The cutoff includes transport slack as well as biological maturity.  It
    # intentionally leaves some late empty tiles instead of buying seeds that
    # cannot be planted and monetized before the terminal step.
    crop_cutoff = {"CARROT": 20, "TOMATO": 14, "WHEAT": 19}
    seed_stock = private.get("seeds", {})
    for crop in ("CARROT", "TOMATO", "WHEAT"):
        if day > crop_cutoff[crop] or len(market) >= 10:
            continue
        stock = int(seed_stock.get(crop, 0))
        gap = max(0, empty_demand[crop] - stock)
        affordable = max(0, (budget - 400) // CROP_DATA[crop]["seed"])
        # A two-seed rolling buffer is enough for parallel planting while
        # bounding terminal waste even if livestock temporarily consumes the
        # available labor.
        count = min(2, gap, affordable, max(0, 2 - stock))
        if count:
            market.append(["BUY_SEED", crop, count])
            budget -= count * CROP_DATA[crop]["seed"]

    farmer, hands = _assign_tasks(obs, tasks)
    return {"farmer": farmer, "hands": hands, "market": market[:10]}


FROZEN_OPPONENTS = {
    "crop-specialist": crop_specialist,
    "diversified-baseline": diversified_baseline,
    "animal-specialist": animal_specialist,
}
