# Competition details

Verified on 2026-08-04 from the live Kaggle pages, authenticated competition API/CLI, and the organizer download.

## Identity

- Name/slug: `Kaggriculture` / `kaggriculture`
- Competition ID: `147734`
- Type: Featured simulation competition with points and medals
- Sponsor: Google LLC; platform host/organization: Kaggle
- Objective: build an autonomous agent that finishes a 30-day season with more banked coins than its opponent
- Prize pool: $50,000, split into ten equal $5,000 prizes for places 1–10
- Account state at verification: rules accepted, submissions enabled, 0 submissions made, 5 daily slots available
- Teams at verification: 1,290

## Timeline

All deadlines are 23:59 UTC.

| Milestone | Date |
|---|---|
| Page-stated start | 2026-07-29 |
| Entry deadline | 2026-09-23 |
| Team merger deadline | 2026-09-23 |
| Final submission deadline | 2026-09-30 |
| Continued games/convergence | approximately 2026-10-01 through 2026-10-15 |

The authenticated API reports an enable time on July 30, one day after the overview's stated start. Kaggle may revise the timeline.

## Evaluation

- Up to 5 agent submissions per team per UTC day.
- Only the latest 2 submissions remain active and are used for final evaluation.
- A new upload first runs a self-play validation episode. A runtime failure produces `Error` status and downloadable logs.
- Valid bots play similarly rated ladder opponents. Wins increase rating, losses decrease it, and ties pull ratings together. Rating changes depend on opponent rating, not coin margin.
- The leaderboard shows the best active submission, while the submissions page tracks both active bots.
- Uploads lock at the final deadline; games continue for about two weeks to reduce uncertainty.
- Final standings use a Bradley–Terry tournament over episodes. Simulation competitions have no private leaderboard.

## Game contract

- Two separate farms; the opponent's public farm and bank are visible, but its shed, seeds, and carried inventories are hidden.
- 30 days × 24 turns = 720 turns by default.
- Each player starts with $3,000, one farmer, the NW 5×5 quadrant of a 10×10 farm, and an empty shed.
- Additional quadrants cost $1,000, $2,000, and $4,000 in NE, SW, SE order.
- One action per farmer/hand per turn, plus up to 10 ordered market orders.
- Products are wheat, carrot, tomato, strawberry, melon, eggs, milk, wool, and fertilizer.
- Crops require watering; animals require wheat feed. Two missed daily refreshes destroy the plant or lose the animal.
- The shared market has fixed seed/animal acquisition costs and supply-sensitive product prices. Town demand drains supply and can support prices.
- The winner has more banked coins after the final turn. Unsold inventory has zero terminal value.

## Submission contract

- Artifact root must contain `main.py` exposing `agent(obs)`.
- A single Python file can be submitted directly; multi-file entries use a `.tar.gz` with `main.py` at its root.
- Runtime files are located under `/kaggle_simulations/agent/`, so imports must resolve from there.
- Episodes have no network ingress or egress.
- Replays and agent actions may be public.
- The rendered FAQ showed 100 MiB upload, 8 GiB disk, 6.5 GiB RAM, and 1.6 vCPU values on the verification date. Raw CLI page content still contained unresolved template tokens, so these resource figures should be rechecked before relying on them.

The competition download contains only `README.md` (21,917 bytes) and `AGENTS.md` (13,057 bytes). There is no train/test table or sample submission CSV; the executable simulator is distributed through `kaggle-environments`.

## Primary sources

- [Overview](https://www.kaggle.com/competitions/kaggriculture/overview)
- [Data](https://www.kaggle.com/competitions/kaggriculture/data)
- [Rules](https://www.kaggle.com/competitions/kaggriculture/rules)
- [Evaluation](https://www.kaggle.com/competitions/kaggriculture/overview/evaluation)
- [Final-evaluation announcement](https://www.kaggle.com/competitions/kaggriculture/discussion/731587)

