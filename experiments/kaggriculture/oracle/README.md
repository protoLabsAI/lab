# oracle/ — ANALYSIS ONLY. NEVER SUBMIT.

These agents replay recorded public episodes of other competitors. Josh's
standing decision (2026-08-05): **we ship our own agents only.** Replaying
another player's move list is not what we're building.

They stay for two legitimate uses:

1. **Upper-bound oracle.** v11 shows what near-perfect execution of the
   convergent top-cluster plan scores ($174k solo). Diffing our original
   agent against it localises our execution deficit in dollars.
2. **Sparring partner.** A stronger opponent than `opponents/sey_v7.py`
   for measuring our own agents.

Ship path is `agents/` only. Current champion: `agents/v10.py`.
