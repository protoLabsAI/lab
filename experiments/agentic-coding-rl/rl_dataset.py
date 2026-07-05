"""Load coding tasks with hidden tests for the RLVR loop.

Reads the same YAML suites the eval harness uses (`evals/tasks/coding/*.yaml`) and
yields flat task records the GRPO loop can consume. The crucial property: the model
sees only `prompt`; the `tests` (assert battery) travel in a separate field and are
NEVER concatenated into the prompt — that is the data-pipeline half of Gate 1's
"hidden held-out tests" requirement (`code_reward` enforces the execution half).

Only `code_exec`-graded tasks are usable as an RL reward (they carry an executable
test battery). Tasks graded only by an LLM judge are skipped — they have no verifiable
signal and would need the strict-judge path instead.

A record:
    {"id", "prompt", "tests": [assert...], "entry": str|None,
     "setup": str, "timeout": int, "source": "<file>#<task_id>"}
"""

from __future__ import annotations

import glob
from dataclasses import asdict, dataclass

import yaml

# Default: the execution-graded coding suites. hard_v2 is the Phase-0 baseline set.
DEFAULT_GLOBS = ("tasks/coding/hard_v2.yaml", "tasks/coding/hard.yaml",
                 "tasks/coding/generation.yaml")


@dataclass
class RLTask:
    id: str
    prompt: str
    tests: list[str]
    entry: str | None
    setup: str
    timeout: int
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


def _code_exec_grader(graders: list[dict]) -> dict | None:
    for g in graders or []:
        if g.get("type") == "code_exec" and g.get("tests"):
            return g
    return None


def load_file(path: str) -> list[RLTask]:
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    suite_id = doc.get("id", path)
    out: list[RLTask] = []
    for t in doc.get("tests", []):
        g = _code_exec_grader(t.get("graders", []))
        if not g:
            continue  # no executable reward -> not usable for RLVR
        out.append(RLTask(
            id=t["id"],
            prompt=t["prompt"],
            tests=list(g["tests"]),
            entry=g.get("entry"),
            setup=g.get("setup", "") or "",
            timeout=int(g.get("timeout", 10)),
            source=f"{suite_id}#{t['id']}",
        ))
    return out


def load_tasks(globs: tuple[str, ...] = DEFAULT_GLOBS, root: str = ".") -> list[RLTask]:
    """Load and de-duplicate RL tasks from the given suite globs (relative to `root`)."""
    seen: set[str] = set()
    tasks: list[RLTask] = []
    for pat in globs:
        for path in sorted(glob.glob(f"{root}/{pat}")):
            for task in load_file(path):
                if task.id in seen:
                    continue
                seen.add(task.id)
                tasks.append(task)
    return tasks


def to_columns(tasks: list[RLTask]) -> dict[str, list]:
    """Column-oriented dict for building a HF Dataset (TRL GRPOTrainer input).

    `prompt` is the trainable column; the rest ride along as aux columns and are
    handed back to the reward function as aligned kwargs — never shown to the model.
    """
    keys = ("id", "prompt", "tests", "entry", "setup", "timeout", "source")
    cols: dict[str, list] = {k: [] for k in keys}
    for t in tasks:
        d = t.as_dict()
        for k in keys:
            cols[k].append(d[k])
    return cols


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ts = load_tasks(root=root)
    print(f"loaded {len(ts)} executable-reward tasks")
    for t in ts[:5]:
        print(f"  {t.source:<40} {len(t.tests)} tests  entry={t.entry} timeout={t.timeout}s")
