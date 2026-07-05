"""TRL-compatible reward function wrapping the hardened `code_reward` (Gate 1).

TRL's `GRPOTrainer(reward_funcs=...)` calls `fn(prompts, completions, **cols) -> list[float]`,
where `cols` are the aux dataset columns (`tests`, `entry`, `setup`, `timeout`) aligned to
the batch (see `rl_dataset.to_columns`). Rewards are sparse binary (all hidden tests pass
→ 1.0, else 0.0) — never partial credit (the field converged here; see RESEARCH.md).

Zero-and-exclude (Ornith layer-2). A reward *function* can only return floats, so TRL cannot
natively drop a gamed trajectory from the advantage estimate. Two ways to honor the exclude
signal, selected by `exclude_policy`:

  * "penalize" (default, TRL-native): gamed/excluded → a strong negative reward (`penalty`).
    Simple and needs no trainer changes. Weaker than true exclusion — it still teaches
    "avoid this exact pattern" and perturbs the group baseline — but ships today.
  * "mask": gamed/excluded → reward 0.0 AND recorded in `fn.last.exclude_mask`. A custom
    GRPOTrainer subclass reads that mask and zeroes the per-sample loss (true advantage
    exclusion). Use this once the masking trainer is wired.

Either way the batch detail is stashed on `fn.last` (a `RewardBatch`) so the trainer / logger
can inspect gaming rate, pass counts, and per-sample reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graders.code_reward import RewardResult, score


@dataclass
class RewardBatch:
    rewards: list[float]
    exclude_mask: list[bool]
    results: list[RewardResult] = field(default_factory=list)

    @property
    def gaming_rate(self) -> float:
        n = len(self.results) or 1
        return sum(1 for r in self.results if r.gamed) / n

    @property
    def solve_rate(self) -> float:
        n = len(self.results) or 1
        return sum(1 for r in self.results if r.reward >= 1.0) / n


def _completion_text(c) -> str:
    """Accept a raw string or a chat-format list of {role, content} messages."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        last = c[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
        return str(last)
    if isinstance(c, dict):
        return c.get("content", "") or ""
    return str(c or "")


def _align(col, n: int, default):
    """A TRL aux column is a per-sample list; tolerate None / scalar broadcast."""
    if col is None:
        return [default] * n
    if isinstance(col, list):
        return col
    return [col] * n


def make_trl_reward_fn(exclude_policy: str = "penalize", penalty: float = -1.0):
    """Build the reward callable. Reads `fn.last` for the most recent RewardBatch."""
    if exclude_policy not in ("penalize", "mask"):
        raise ValueError(f"exclude_policy must be 'penalize' or 'mask', got {exclude_policy!r}")

    def reward_fn(prompts=None, completions=None, tests=None, entry=None,
                  setup=None, timeout=None, **_ignored):
        comps = completions or []
        n = len(comps)
        tests_col = _align(tests, n, [])
        entry_col = _align(entry, n, None)
        setup_col = _align(setup, n, "")
        timeout_col = _align(timeout, n, 10)

        rewards: list[float] = []
        mask: list[bool] = []
        results: list[RewardResult] = []
        for i, c in enumerate(comps):
            r = score(_completion_text(c), tests_col[i] or [], entry=entry_col[i],
                      setup=setup_col[i] or "", timeout=int(timeout_col[i] or 10))
            results.append(r)
            excluded = r.exclude
            mask.append(excluded)
            if excluded and exclude_policy == "penalize":
                rewards.append(penalty)
            else:
                # "mask" policy keeps the true 0.0 and lets the trainer drop the sample;
                # non-excluded samples always use the sparse-binary reward.
                rewards.append(r.reward)

        reward_fn.last = RewardBatch(rewards=rewards, exclude_mask=mask, results=results)
        return rewards

    reward_fn.last = RewardBatch([], [], [])
    reward_fn.exclude_policy = exclude_policy
    reward_fn.__name__ = "code_exec_reward"  # TRL uses __name__ for logging columns
    return reward_fn
