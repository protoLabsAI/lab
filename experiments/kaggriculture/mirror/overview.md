# Kaggriculture - Kaggle Simulation Competition Overview

> **Kaggle Featured Simulation Competition**  
> **Host:** Kaggle  
> **Prizes & Awards:** $50,000 | Awards Points & Medals  
> **Participation:** 3,998 Entrants | 996 Participants | 968 Teams | 1,749 Submissions  
> **Tags:** Simulations, Custom Metric  

---

## Overview

In this competition, you will design, build, and deploy an autonomous AI agent to manage a virtual farm, navigate a dynamic economy, and compete head-to-head against other agents on a live leaderboard.

### Description

This simulation competition is a turn-based farming game where two players compete on separate farms to see who can earn the most profit by the end of a 30-day season (720 turns).

Your agent acts as the main farmer and can strategically hire farm hands to scale up operations. To succeed, your agent must:
* **Plant, water, fertilize, and harvest** a variety of crops.
* **Buy, feed, and care** for animals to produce eggs, milk, and wool.
* **Collect and utilize fertilizer** to boost crop yields.
* **Buy neighboring quadrants** of land to expand your farm's footprint.
* **Trade smart** on a dynamic market where prices react to your sales and town demand.

Kaggriculture represents a highly complex environment that models the exact same dynamics found in real-world supply chains, dynamic market pricing, and industrial resource allocation under uncertainty. Underlying mechanics like scheduling resources, optimizing labor, adjusting to supply/demand price changes, and making long-horizon capital investments, serve as a high-fidelity sandbox for training AI to solve complex enterprise operations.

---

## Timeline

* **July 29, 2026** - Start Date.
* **September 23, 2026** - Entry Deadline. You must accept the competition rules before this date in order to compete.
* **September 23, 2026** - Team Merger Deadline. This is the last day participants may join or merge teams.
* **September 30, 2026** - Final Submission Deadline.
* **October 1, 2026 to (approx) October 15, 2026** - We will continue to run games, or until the leaderboard has reached convergence. At the conclusion of this period, the leaderboard is final.

*All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.*

---

## Evaluation

Each day your team is able to submit up to 5 agents (bots) to the competition. Each submission will play Episodes (games) against other bots on the ladder that have a similar skill rating. Over time, skill ratings will go up with wins or down with losses, and even out with ties. To reduce the number of bots playing and ensure high-quality matching, only the latest 2 submissions are tracked. The latest 2 submissions are also used for final leaderboard evaluation.

Every bot submitted will continue to play episodes until the end of the competition, with newer bots playing a much more frequent number of episodes. On the leaderboard, only your best-scoring bot will be shown, but you can track the progress of all of your submissions on your Submissions page.

When you upload a submission, a **Validation Episode** is run where your agent plays against a copy of itself to ensure it runs without error. If the episode fails, the submission is marked as `Error`, and you can download the agent logs to debug. Otherwise, the submission is initialized with a default rating and joins the matchmaking pool.

### Ranking System

Each submission is assigned a skill rating. When your agent plays an episode against an opponent:
* **Winning the match** (having the most coins in the bank at the end of 720 turns) increases your skill rating, while losing decreases it.
* The amount your rating changes depends on the rating difference between you and your opponent. Beating a highly-rated agent will boost your rating more than beating a lower-rated one.
* Ties will generally pull ratings closer together.
* The actual coin difference in a match does not affect the rating change—only the win, loss, or tie outcome matters.

### Final Evaluation

At the submission deadline, additional submissions will be locked. Games will continue to run for approximately two weeks to continue to reduce uncertainty, especially for new agents. A final Bradley-Terry tournament will be run on those episodes to produce the final leaderboard.

---

## Prizes

* **1st Place** - $5,000
* **2nd Place** - $5,000
* **3rd Place** - $5,000
* **4th Place** - $5,000
* **5th Place** - $5,000
* **6th Place** - $5,000
* **7th Place** - $5,000
* **8th Place** - $5,000
* **9th Place** - $5,000
* **10th Place** - $5,000

---

## How to Play

### Overview

Each player starts with an empty farm and a small amount of income (seed money, if you will). Each turn, they can perform actions such as moving around the board, purchasing seeds or livestock, planting seeds, watering plants, harvesting produce or animal products, and selling that produce at the market. The game runs for a fixed amount of time representing one season, and the winner is determined by who has the most money in the bank at the end.

### Object Types

| Type | Yield Type | Seed Cost | Base Market Price | Time to First Yield | Time to Max Yield | Subsequent Yields | Max Yield | Action Cost | Max yield / tile / DAY |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wheat** | One-time | 10 | 25 | 2 days | 4 days | none | 6 | 1 | 1.5 |
| **Carrot** | One-time | 20 | 35 | 2 days | 3 days | none | 4 | 1 | 1.333 |
| **Tomato** | Ongoing | 50 | 60 | 7 days | NA | every day | 4 | 1 | 4 |
| **Strawberry** | Ongoing | 100 | 120 | 10 days | NA | every other day | 4 | 1 | 2 |
| **Melon** | One-time | 80 | 250 | 10 days | 12 days | none | 6 | 1 | .5 |
| **Goose/Egg** | Ongoing | 300 | 50 | 4 days | NA | every day | 4 | 1 + 1 (build coop) | 2 |
| **Cow/Milk** | Ongoing | 400 | 160 | 8 days | NA | every two days | 6 | 1 + 1 (build pasture) | 1 |
| **Sheep/Wool** | Ongoing | 500 | 200 | 6 days | NA | every three days | 6 | 1 + 1 (build pasture) | .67 |
| **Fertilizer** | NA | 100 | X | X | X | X | X | 1 | - |

> [!IMPORTANT]
> All plants must be watered every day. They will turn into weeds if they are not watered for two successive days. All animals must be fed every day using wheat. They will escape and be unrecoverable if they are not fed for two successive days. Wheat is also available to buy at the market and can be purchased at the current market price.

### Actions

Each turn, the player may take one action. There are 24 turns per day, and 30 days in the season — 720 total turns.

#### Farmer / Farm Hand Action
Each Farmer / Farm Hand can be given an action every turn. Farmer/Farm Hand CAN occupy the same space.

##### Movement
* `NORTH`, `SOUTH`, `EAST`, `WEST` — Move one cell in that direction.

##### Shed
Picks up an item from the shed (must be orthogonally adjacent) into the inventory:
* `PICKUP <item> [n]` — move up to `n` of `<item>` (default 1) from the shed into the active farmer/hand's inventory. Any item present in the shed is valid (animals, fertilizer, harvested produce, etc.). Seeds live in a separate slot and are never picked up — `PLANT` consumes them directly.
* `DROP` — orthogonally adjacent to the shed, dump the active farmer/hand's entire current inventory into the shed. Overflow past `shedCapacity` is discarded. No-op if not shed-adjacent.

##### Plants
* `PLANT` — Plant a seed purchased from the market.
  * Seeds are automatically available to all Farmers / Farm Hands.
  * If you try to plant too many in a specific turn, none are planted (i.e. if you have 1 melon seed, but two units do the `PLANT MELON` command).
* `WATER` — Water a plant. This only needs to be done once per day, and subsequent waterings on the same day are a no-op.
* `HARVEST` — Gather produce from a plant. If the plant does not have subsequent yields, it will be removed from the map. Each harvest action will yield at least one unit of the crop, with the potential of additional yield depending on watering and fertilizer (the formula differs by crop type — see harvest yields below). Harvested items are added to the inventory.
* `FERTILIZE` — Fertilize a plant to increase its potential yield.
  * Doubles the per-day yield bonus for the next 3 days. The bonus only applies on days the plant is also watered (basic needs first).

##### Animals
* `PLACE <item> [n]` — Drop items from the active farmer/hand inventory into either a tile or the shed:
  * **Animal placement:** standing on a matching unoccupied structure (`GOOSE` on a coop, `SHEEP` / `COW` on a pasture) places one animal from inventory onto the tile. The `n` argument is ignored.
  * **Shed drop:** standing orthogonally adjacent to the shed moves up to `n` (default 1) of `<item>` from inventory into the shed. Capped by `shedCapacity`; excess stays in inventory.
* `FEED` — Feed an animal using wheat (only needs to be done once per day).
* `HARVEST` — Collect the eggs/milk/wool produced by the animal.
* `COLLECT_FERTILIZER` — Collect 1 fertilizer from the animal. Each surviving animal makes 1 fertilizer available at the end of every day; collecting consumes that day's stock and the next becomes available after the next end-of-day refresh.
* `CARE` — Care for an animal (once per day, no-op if already cared for). See animal care below.

##### Animal Care
`CARE` banks a yield bonus that is paid out on the animal's next scheduled production:
* At end of day, if the animal was both fed AND cared for that day, `pending_care_bonus` increments by 2. Days where the animal was unfed do not bank a bonus (basic needs first).
* On a scheduled production day, if the animal is fed, the entire banked bonus is added to that production's yield (in addition to the base 1) and the bank resets to 0.
* If the animal is unfed on the production day, no yield is produced that day and the bank is also reset.
* `pending_care_bonus` is capped indirectly by the per-animal `max_held` cap on `yield_units`.

##### Terrain
* `BUILD_COOP` — adds a coop to an unoccupied tile.
* `BUILD_PASTURE` — add pasture to an unoccupied tile.
* `DIG` — Remove a plant from a square to free up space OR remove a weed from a square (does not yield any produce) OR remove a goose coop / pasture.

##### Other
* `PASS` — Default if there is nothing to do (optional).

#### Market Action
Each turn you can submit up to `maxMarketOrdersPerTurn` (default 10) market actions; any orders past that limit are silently dropped. This is an ordered list and market orders will be processed in order simultaneously (one from each player) while both players have orders.
* `BUY_SEED` — Purchase N units of a single item from the market (e.g. `BUY_SEED WHEAT 1`).
* `BUY_ANIMAL` — Purchase N units of livestock (e.g. `BUY_ANIMAL GOOSE 1`).
* `BUY_PRODUCT` — Purchase N units of product (e.g. `BUY_PRODUCT WHEAT 1`, `BUY_PRODUCT FERTILIZER 1`).
* `SELL` — Sell N units of a single item to the market (e.g. `SELL WHEAT 1`).
* `HIRE` — Hire a farm hand for the day. Cost increases for each extra hand hired on the same day.
* `BUY_LAND` — Unlock a new 5x5 segment of land to plant on. Costs are: $1k, $2k, $4k.

---

### Watering / Animal Feed

Plants (and animals) must be watered/fed a minimum of every other day. Watering only needs to be done once per day, and subsequent watering actions are a no-op. In the case of plants not watered for two consecutive days, at the end of the day they turn into a `WEED`. In the case of animals they escape (unrecoverable).

Note that watering one-time yield plants during their yield window results in a higher yield. This is NOT true for ongoing yield plants/animals.

### Harvest Yields

Plants will potentially have higher yields based on how well they have been cared for:
* **One-time crops (wheat, carrot, melon):** Starting at half the plant's `max_yield_day` (Time to Max Yield) rounded up, watering during the bonus window will add one unit per day to the total harvestable yield. Fertilized plants add 2 per day instead. Once a plant has hit its maximum lifespan, the total yield available on the plant will reduce by 1 every other turn until it hits 0, at which point the plant becomes a weed. One-time crops reach max lifespan one day after `max_yield_day`.
* **Ongoing crops (tomato, strawberry):** Scheduled production happens at fixed intervals. The base yield is 1 per scheduled production. If the plant is fertilized AND watered that day, yield is doubled to 2. Ongoing crops start decay one day after their cumulative production count reaches `max_yield` (i.e. they've fired enough scheduled productions to hit the cap, regardless of whether the produce has been harvested).

---

## Map Features

Each player has their own farm with a set number of squares. Players are unable to see the state of the other's shed, but can see the state of their opponent's farm.

### Farm Space
* The land near your farm is a `boardSize x boardSize` grid (default 10x10), divided into four 5x5 quadrants. At first, your farm covers one quadrant (25% of the squares). For an increasingly large fee, you can buy the neighboring quadrants and eventually cover 100% of the squares.
* Each plant or animal occupies one square on the farm.
* Players can allocate these squares however they choose between crops and livestock. There are no specific limits per type.
* Weeds have a chance of spawning on any empty cells on the farm, and must be cleared before the land can be used for other purposes.
* Squares on the farm can be either a plant, a coop/pasture, a weed, or empty.

### Shed (Inventory)
* Functions as an inventory for items that are harvested but not yet sold, or for seeds that have not yet been planted.
* Farmer and hired farm hands will spawn at the shed at the start of each day.
* Farmer and hired farm hands drop their inventory at the end of the day in the shed (if there is room).
* Limited to 100 items, excluding seeds. Once the shed is full, any further items added (via `PLACE` mid-day or end-of-day inventory drop) are discarded — there is no overflow holding area, so stockpiling on farmer/hand inventories does not bypass the cap.

### Farmer / Farm Hand Hiring
* Hiring is a market order (`HIRE`). It costs more every time you want to hire an additional hand each day. At the end of the day, all hands drop inventory at the farm and disappear (need to be re-hired each day).
* Cost is `farmHandCostMult * fib(n)` where `n` is the number of hires already made today (`fib` starts 1, 1, 2, 3, 5, 8, 13, ...).
  * With the default `farmHandCostMult = 1`: 1, 1, 2, 3, 5, 8, 13, 21, etc... (resets at the start of each day).
* A hired hand appears orthogonally adjacent to the shed in a free space following NWSE. If there are no open spaces, it looks for the one with the least occupants, breaking ties by NWSE preference.

### Inventory
* When harvesting or picking items up, they are added to inventory.
* Can drop items in the shed.
* At the end of the day, all items in all inventory will be added to shed inventory (if there is room). Anything that doesn't fit is discarded — overflow is lost.

---

## Town Buildings

As the season progresses, new shops unlock at regular intervals (every `townShopUnlockInterval` days, default 3). Each unlock is randomly selected from the shops that have not yet been added; once unlocked, a shop stays active for the rest of the game. Total demand grows monotonically as more shops unlock.

Each unlocked shop consumes one of every product it demands every `townShopSellInterval` turns (default 4). So with the default interval, a shop demanding wheat removes 6 wheat from the market per day. Single-product shops consume 2x.

In addition, the town center consumes one of every product (excluding fertilizer) every `townCenterSellInterval` turns (default 12). After day 10 this is increased to 2 of each, and after day 20 it is increased to 4 of each.

| Shop Type | Increases Demand For |
| :--- | :--- |
| **Bakery** | eggs, wheat |
| **Pizza Shop** | milk, tomatoes, wheat |
| **Brunch Spot** | eggs, wheat, strawberries |
| **Yarn Store** | wool (2x) |
| **Ice Cream Shop** | strawberries, milk, wheat |
| **Pet Cafe** | carrots (2x) |
| **Smoothie Shop** | strawberries, milk |
| **Farmers Market** | wheat, carrots, tomatoes, strawberries |

---

## Market Mechanics

The market has an unlimited supply of seeds and animals at fixed prices. Sell prices, however, move dynamically per resource and persist across days.

Every product (and fertilizer) starts the game with a market inventory of $I_0 = 10,000$ units, far above any single game's realistic production volume so that inventory is essentially guaranteed to stay positive. The sell price for a product is `base` at $I_0$, rises as inventory falls (players buying or town consumption draining supply), and falls as inventory grows (players selling).

### Selling inventory to the market
Players can queue any number of sell or buy orders (for any quantity) in the market action list. Orders are processed concurrently across players, one unit at a time. For example, when both players issue `SELL CARROT 10` first, we take the current carrot price, give both players that price for their first carrot, then add 2 carrots to the market (1 from each player) — which may shift the price — and repeat until both orders complete.

If the sell price has been driven down to $1 (the price floor), the unit is still purchased but is not added to market inventory, so the floor remains responsive to subsequent buys.

### Buying inventory from the market
Only WHEAT and FERTILIZER can be bought from the market via `BUY_PRODUCT` (other products are sold at the market but not bought back). Two things drain market inventory: town buildings (town center and shops, which consume products for free) and player `BUY_PRODUCT` orders. Buy orders follow the same one-unit-at-a-time concurrent procedure as sell orders. If a player runs out of money mid-order, the order is stopped.

The buy price is quoted at the post-buy inventory and the sell price is quoted at the pre-sell inventory, so an immediate buy followed by a sell of the same item against an otherwise-unchanged market nets exactly zero.

### The Price Function

For each resource the curve is defined by a base price, an anchor throughput $T$, and an independent shape function + target move for each side of the equilibrium:

$$\text{price}(\text{inv}) = \text{base} + \text{sign} \cdot \text{amp} \cdot f(|\text{inv} - I_0|)$$

* $\text{sign} = +1$ if $\text{inv} < I_0$ (scarcity $\rightarrow$ price up)
* $\text{sign} = -1$ if $\text{inv} > I_0$ (glut $\rightarrow$ price down)
* $\text{amp} = \frac{\text{target} \cdot \text{base}}{f(T)}$ (derived; not stored)
* $f \in \{ \text{linear}, \text{sq}, \text{sqrt}, \text{log}, \text{log10} \}$ ($\text{log}$ uses $\ln(1+x)$, so $f(0)=0$)

*Floored at $1 and rounded to the nearest dollar.*

$T$ is the production capacity of a single 5x5 field over a 24-day game at optimal watering with no fertilizer (animal totals are pre-discounted by 30% to account for wheat-feed overhead). `target` says "moving $T$ units past $I_0$ shifts the price by $\text{target} \times \text{base}$." Picking different $f$ and `target` on each side lets resources with similar production profiles play very differently strategically — wheat panics on scarcity but absorbs gluts, carrot is the opposite; melon barely reacts to scarcity but crashes hard on overproduction; wool mirrors melon at a smaller scale. Premium resources ($\text{base} > \$100$: strawberry, melon, milk, wool) use $\text{above\_target} > 1$, so even modest gluts drive them straight to the $1 floor — bundling and timing sales matters more for these than for staples.

| Resource | Base | $I_0$ | $T$ | Below func | Below target | Above func | Above target | $P(I_0 - T)$ | $P(I_0 + T)$ | $P(I_0 + 2T)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wheat** | 25 | 10,000 | 400 | sq | 0.80 | log | 0.20 | $45 | $20 | $17 |
| **Carrot** | 35 | 10,000 | 450 | log | 0.20 | sqrt | 0.70 | $42 | $10 | $1 |
| **Tomato** | 60 | 10,000 | 200 | linear | 0.40 | sqrt | 0.60 | $84 | $24 | $9 |
| **Strawberry** | 120 | 10,000 | 100 | sqrt | 0.70 | linear | 1.60 | $204 | $1 | $1 |
| **Melon** | 250 | 10,000 | 300 | log | 0.20 | sq | 3.60 | $300 | $1 | $1 |
| **Egg** | 50 | 10,000 | 332 | linear | 0.40 | log | 0.20 | $70 | $40 | $34 |
| **Milk** | 160 | 10,000 | 122 | sqrt | 0.60 | linear | 1.60 | $256 | $1 | $1 |
| **Wool** | 200 | 10,000 | 105 | log | 0.20 | sq | 3.20 | $240 | $1 | $1 |
| **Fertilizer** | 100 | 10,000 | 200 | linear | 0.40 | linear | 0.40 | $140 | $60 | $20 |

The defaults live in `MARKET_PARAMS` in `kaggriculture.py`. Per-resource overrides (sparse: any subset of base, $I_0$, $T$, `below_func`, `below_target`, `above_func`, `above_target`) can be supplied at episode creation via `env.configuration["marketParams"]` without touching code, e.g. `{"WOOL": {"above_target": 0.95}}`.

---

## Turn Processing Order

1. **Action validation** — verify action legality
2. **Player actions** — record the actions taken by each player (happening simultaneously)
3. **Market actions** — process market queue in order by player (described above)
4. **Town buy actions** — town center and shops reduce inventory
5. **Update observations:**
   * **Day refresh** — if applicable, update the condition of plants and animals for a new day, and reset their fed/watered condition to false
   * **Market refresh** — modify the price of items on the market based on sells from previous turn
   * **Income update** — update the player's bank based on any buys or sells
   * **Farm update** — clear plants that have been harvested, items from the inventory that have been used or sold, add new plants/animals to the farm, etc.

---

## Win Conditions & Reward

* **Win Condition:** Whoever has the greatest number of coins at the end of the season (720 turns) is the winner. It is also possible that the two players will tie.
* **Reward:** The player who has the most money in the bank at the end of the game wins. Unsold items in the inventory do not count towards that total.

---

## Observation Format

The top-level observation passed to each agent:

```json
{
  "player": 0, // 0 or 1
  "day": 0,    // 0-indexed in-game day
  "hour": 0,   // 0-indexed turn within the day
  "farms": [farm, farm], // public per-player state, indexed by player id (shared)
  "market": {  // shared
    "inventory": { "WHEAT": 10000, "CARROT": 10000 },
    "prices": { "WHEAT": 25, "CARROT": 35 }
  },
  "town": {    // shared
    "unlocked_shops": ["BAKERY"]
  },
  "private": { // this player only; opponent's private state is not visible
    "shed": { "WHEAT": 0, "GOOSE": 0, "FERTILIZER": 0 },
    "seeds": { "WHEAT": 0, "CARROT": 0 },
    "inventories": [farmer_inv, hand_inv] // [0] is the main farmer
  }
}
```

Each farm dict (public, visible to both players):

```json
{
  "money": 3000.0,
  "tiles": [[tile]], // tiles[y][x]
  "farmer": [x, y],
  "hands": [[x, y]], // hired hands for the current day
  "unlocked_quadrants": ["NW"], // subset of {"NW", "NE", "SW", "SE"}
  "hires_today": 0 // used to price the next HIRE
}
```

A tile is one of:
* `null` — empty unlocked tile
* `"LOCKED"` — tile in a quadrant the player has not yet bought
* **Plant dict:**
  ```json
  {
    "kind": "PLANT",
    "crop": "WHEAT", // "WHEAT" | "CARROT" | "TOMATO" | "STRAWBERRY" | "MELON"
    "planted_day": 0,
    "watered_today": false, // reset to False each end-of-day
    "consecutive_unwatered": 0, // 2+ -> tile turns to a weed
    "yield_units": 0, // units currently harvestable
    "max_lifespan_step": 0, // step at which decay begins; -1 for ongoing crops
    "fertilized_until_day": -1 // last day fertilizer bonus applies; -1 if none
  }
  ```
* **Weed dict:** `{"kind": "WEED"}`
* **Animal structure dict** (coop/pasture, optionally occupied):
  ```json
  {
    "kind": "COOP", // "COOP" | "PASTURE"
    "animal": "GOOSE", // "GOOSE" | "COW" | "SHEEP" | null
    "placed_day": 0,
    "yield_units": 0,
    "fed_today": false,
    "consecutive_unfed": 0, // 2+ -> animal escapes
    "cared_today": false,
    "fertilizer_available": false, // set after CARE; cleared by COLLECT_FERTILIZER
    "pending_care_bonus": 0 // banked CARE bonus, applied on the next yield tick
  }
  ```

---

## Configuration Defaults

Per-crop seed costs and per-product base prices are not configurable; they are documented in the Object Types and Price Function tables above. The configurable knobs are:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `episodeSteps` | `720` | Total turns in the season (24 turns x 30 days) |
| `boardSize` | `10` | Width and height (in tiles) of each player's square farm. Advanced uses 10 = four 5x5 quadrants |
| `startingMoney` | `3000` | Coins each player starts with |
| `maxMarketOrdersPerTurn` | `10` | Maximum number of market orders processed per player per turn; extras are silently dropped |
| `turnsPerDay` | `24` | Number of turns that make up one in-game day |
| `shedCapacity` | `100` | Max non-seed items the shed can hold; overflow at end-of-day drop is discarded |
| `weedSpawnChance` | `0.005` | Per-tile probability of a weed spawning on an empty unlocked tile during end-of-day refresh |
| `townShopUnlockInterval` | `3` | Days between successive town shop unlocks |
| `townShopSellInterval` | `4` | Turns between consumption ticks by every unlocked town shop |
| `townCenterSellInterval` | `12` | Turns between consumption ticks by the town center |
| `seed` | `null` | Optional input seed for deterministic episode generation; cleared from config after read so it stays out of agent observations |

---

## Getting Started: Test Locally & Submit

### Test Locally

Install the environment from PyPI:
```bash
pip install -U kaggle-environments
```

Run a game from Python or a notebook:
```python
from kaggle_environments import make

env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run([agent, "random"]) # or env.run(["main.py", "random"]) to load from a file

# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Render in a notebook
env.render(mode="ipython", width=1200, height=800)

# Or dump a replay JSON for the visualizer / offline analysis
import json
with open("replay.json", "w") as f:
    json.dump(env.toJSON(), f)
```

Three built-in agents are available by name: `"pass"`, `"random"`, and `"starter"` (a deterministic baseline).

### Set Up the Kaggle CLI

Install the CLI:
```bash
pip install kaggle
```

You'll need a Kaggle account — sign up at [https://www.kaggle.com](https://www.kaggle.com) if you don't have one. Then download your API credentials at [https://www.kaggle.com/settings/api](https://www.kaggle.com/settings/api) by clicking "Generate New Token" under the "API" section.

**Recommended: API token file.** Save the token string to `~/.kaggle/access_token`:
```bash
mkdir -p ~/.kaggle
# Paste the token from the Kaggle settings UI into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

**Alternative auth methods:**
* OAuth (browser flow): `kaggle auth login`
* Environment variable: `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxxx`

### Find the Competition

```bash
kaggle competitions list -s "kaggriculture"
kaggle competitions pages kaggriculture
kaggle competitions pages kaggriculture --content
```

### Accept the Competition Rules

Before submitting, you must accept the rules on the Kaggle website. Navigate to [https://www.kaggle.com/competitions/kaggriculture](https://www.kaggle.com/competitions/kaggriculture) and click **"Join Competition"**.

Verify you've joined:
```bash
kaggle competitions list --group entered
```

### Download Competition Data

```bash
kaggle competitions download kaggriculture -p kaggriculture-data
```

### Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**
```bash
kaggle competitions submit kaggriculture -f main.py -m "Wheat loop v1"
```

**Multi-file agent — bundle into a tar.gz with main.py at the root:**
```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit kaggriculture -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**
```bash
kaggle competitions submit kaggriculture -k YOUR_USERNAME/kaggriculture-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

### Monitor Your Submission

Check submission status:
```bash
kaggle competitions submissions kaggriculture
```

### List Episodes

```bash
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions episodes <SUBMISSION_ID> -v
```

### Download Replays and Logs

```bash
# Download replay JSON for an episode
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays

# Download agent logs (index 0 for player 1, index 1 for player 2)
kaggle competitions logs <EPISODE_ID> 0
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

### Check the Leaderboard

```bash
kaggle competitions leaderboard kaggriculture -s
```

### Typical Workflow

```python
# Test locally
python -c "
from kaggle_environments import make
env = make('kaggriculture', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit kaggriculture -f main.py -m "v1"

# Check status
kaggle competitions submissions kaggriculture

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard kaggriculture -s
```

---

## Quick Start Agent

Below is a simple starter agent implementing a basic wheat loop:

```python
def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market = []

    # Buy a wheat seed if we have none and have enough money
    if private["seeds"].get("WHEAT", 0) == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])

    # Sell any wheat sitting in the shed
    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])

    # If standing on an empty tile, plant wheat
    if tile is None and private["seeds"].get("WHEAT", 0) > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}

    # If standing on a plant, manage watering and harvesting
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = obs["day"] - tile["planted_day"]
        if crop_age >= 2:  # Wheat first_yield_day = 2
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market}

    return {"farmer": ["PASS"], "hands": [], "market": market}
```

---

## Frequently Asked Questions

### Submissions
* Submissions must be at most **100 MiB**.
* Daily submission limit: **5**.
* Only your **most recent 2** are active.
* Your files will be located in `/kaggle_simulations/agent/`. Ensure all your file imports are set appropriately.

### Submission Resources
* **HDD Space:** 8 GiB
* **RAM:** 6.5 GiB
* **vCPUs:** 1.6
* **Submission Size Limit:** 100 MiB

---

## Citation

```bibtex
Bovard Doerschuk-Tiberi, Domino Weir, and Maria Cruz. Kaggriculture. https://kaggle.com/competitions/kaggriculture, 2026. Kaggle.
```
