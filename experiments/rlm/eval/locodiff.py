"""LoCoDiff loader + scorer.

LoCoDiff (Mentat AI) prompts have a fixed shape:

  # Instructions
  ...prose...
  # Required Response Format
  ...prose...
  # File History

  > git log -p --cc --topo-order --reverse -- <PATH>

  commit <sha>
  ...diff...
  commit <sha>
  ...

We split on the first `commit ` line. Everything before is "instructions" (we
discard — RLM gives the planner its own framing). Everything from the first
`commit ` is the git log we hand to the planner as a REPL variable.

Scoring is exact string match against `<task>_expectedoutput.txt`.
Repo: https://github.com/AbanteAI/LoCoDiff-bench (MIT)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_GIT_CMD_RE = re.compile(r"^>\s*git log[^\n]*--\s*([^\s]+)", re.MULTILINE)


@dataclass
class LoCoDiffTask:
    name: str
    target_path: str
    git_log: str
    expected: str
    prompt_bytes: int

    @property
    def expected_bytes(self) -> int:
        return len(self.expected)


def _split_prompt(prompt_text: str) -> tuple[str, str]:
    """Return (instructions_block, git_log_text).

    The git log starts at the first line beginning with `commit `.
    """
    idx = prompt_text.find("\ncommit ")
    if idx == -1:
        # Fallback: no commit found, treat whole thing as git_log
        return "", prompt_text
    return prompt_text[:idx], prompt_text[idx + 1 :]


def _extract_target_path(instructions: str, fallback_name: str) -> str:
    m = _GIT_CMD_RE.search(instructions)
    if m:
        return m.group(1)
    # Fallback: derive from filename like aider_aider_args.py_prompt.txt
    return fallback_name


def load_task(prompt_path: Path | str) -> LoCoDiffTask:
    prompt_path = Path(prompt_path)
    if not prompt_path.name.endswith("_prompt.txt"):
        raise ValueError(f"expected *_prompt.txt, got {prompt_path.name}")
    name = prompt_path.name.removesuffix("_prompt.txt")
    expected_path = prompt_path.parent / f"{name}_expectedoutput.txt"

    prompt_text = prompt_path.read_text(encoding="utf-8")
    expected_text = expected_path.read_text(encoding="utf-8")

    instructions, git_log = _split_prompt(prompt_text)
    target = _extract_target_path(instructions, name)

    return LoCoDiffTask(
        name=name,
        target_path=target,
        git_log=git_log,
        expected=expected_text,
        prompt_bytes=len(prompt_text),
    )


def list_tasks(prompts_dir: Path | str) -> list[Path]:
    return sorted(Path(prompts_dir).glob("*_prompt.txt"))


def score(predicted: str | None, expected: str) -> bool:
    """Exact match. LoCoDiff has no partial credit.

    We try the prediction verbatim first; if it's wrapped in a single ```...```
    fence (chat-model habit), strip the fence and try again. We do NOT strip
    whitespace globally — trailing newlines are part of the file's true state.
    """
    if predicted is None:
        return False
    if predicted == expected:
        return True
    p = predicted.strip()
    if p.startswith("```") and p.endswith("```"):
        first_nl = p.find("\n")
        if first_nl != -1:
            inner = p[first_nl + 1 : -3]
            # Files almost always end in a newline; preserve the inner content.
            if inner == expected:
                return True
            if inner.rstrip("\n") + "\n" == expected:
                return True
    return False


def bucket(prompt_bytes: int) -> str:
    """Approximate the paper's quartiles. Tokens ≈ bytes / 3.5 for English+code."""
    tokens = prompt_bytes / 3.5
    if tokens <= 21_000:
        return "Q1"
    if tokens <= 36_000:
        return "Q2"
    if tokens <= 60_000:
        return "Q3"
    return "Q4"
