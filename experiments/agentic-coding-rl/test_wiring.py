"""Verify rl_dataset loader + reward_fn over the REAL coding suite (GPU-free).

Run from the experiment dir:
  PYTHONPATH=/home/ava/dev/lab/evals:/home/ava/dev/lab/experiments/agentic-coding-rl \
  <evals-venv-python> test_wiring.py
"""
import sys

from rl_dataset import load_tasks, to_columns
from reward_fn import make_trl_reward_fn

EVAL_ROOT = "/home/ava/dev/lab/evals"
fails = 0


def check(name, cond, extra=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{extra}]" if extra else ""))
    if not cond:
        fails += 1


print("== loader ==")
tasks = load_tasks(root=EVAL_ROOT)
check("loads executable-reward tasks", len(tasks) > 0, f"{len(tasks)} tasks")
check("every task has hidden tests", all(len(t.tests) > 0 for t in tasks))
check("prompt never contains an assert (tests stay hidden)",
      all("assert " not in t.prompt for t in tasks))
sample = tasks[0]
print(f"  sample: {sample.source}  entry={sample.entry}  {len(sample.tests)} tests")

cols = to_columns(tasks)
check("to_columns aligns all columns",
      len({len(v) for v in cols.values()}) == 1 and len(cols["prompt"]) == len(tasks))

print("== reward_fn: penalize policy ==")
fn = make_trl_reward_fn(exclude_policy="penalize", penalty=-1.0)

# Build a batch: [wrong-on-real, gaming-on-real, correct-on-synthetic, chat-format-correct]
synthetic_tests = ["assert solve([1,2,3]) == 6", "assert solve([]) == 0"]
prompts = [sample.prompt, sample.prompt, "sum a list", "sum a list"]
completions = [
    "```python\ndef %s(*a, **k):\n    return None\n```" % (sample.entry or "solve"),  # wrong
    "```python\nimport sys\nsys.exit(0)\n```",                                          # gaming
    "```python\ndef solve(xs):\n    return sum(xs)\n```",                               # correct
    [{"role": "assistant", "content": "```python\ndef solve(xs):\n    return sum(xs)\n```"}],  # chat
]
tests_col = [sample.tests, sample.tests, synthetic_tests, synthetic_tests]
entry_col = [sample.entry, sample.entry, "solve", "solve"]
timeout_col = [sample.timeout, sample.timeout, 8, 8]

rewards = fn(prompts=prompts, completions=completions, tests=tests_col,
             entry=entry_col, timeout=timeout_col)
batch = fn.last
check("wrong real completion -> 0.0", rewards[0] == 0.0, f"got {rewards[0]}")
check("gaming completion -> penalty -1.0", rewards[1] == -1.0, f"got {rewards[1]}")
check("gaming flagged in exclude_mask", batch.exclude_mask[1] is True)
check("correct synthetic -> 1.0", rewards[2] == 1.0, f"got {rewards[2]}")
check("chat-format correct -> 1.0", rewards[3] == 1.0, f"got {rewards[3]}")
check("gaming_rate = 1/4", abs(batch.gaming_rate - 0.25) < 1e-9, f"{batch.gaming_rate}")
check("solve_rate = 2/4", abs(batch.solve_rate - 0.5) < 1e-9, f"{batch.solve_rate}")

print("== reward_fn: mask policy ==")
fnm = make_trl_reward_fn(exclude_policy="mask")
rewards_m = fnm(prompts=prompts, completions=completions, tests=tests_col,
                entry=entry_col, timeout=timeout_col)
check("mask policy: gaming -> reward 0.0 (not penalty)", rewards_m[1] == 0.0, f"got {rewards_m[1]}")
check("mask policy: gaming still in exclude_mask", fnm.last.exclude_mask[1] is True)
check("mask policy: correct still 1.0", rewards_m[2] == 1.0)

print("-" * 60)
print("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
