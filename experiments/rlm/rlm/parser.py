"""Parse planner output into actionable units.

The planner produces free text that may contain:
- Fenced ```python ... ``` code blocks (we execute the FIRST one per turn)
- A FINAL(...) sentinel to terminate with a literal answer
- A FINAL_VAR(name) sentinel to terminate with a REPL variable's repr
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# Detects an unclosed code fence — happens when the planner gets truncated by
# max_tokens mid-block. We surface this so the orchestrator can ask for a redo
# rather than treat a truncated turn as "neither code nor FINAL".
_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n", re.DOTALL)
# FINAL(...) where the body does NOT contain ')'. Simple, common case; for nested
# parens use FINAL_VAR(name). We find ALL matches and take the last.
_FINAL_RE = re.compile(r"FINAL\(([^)]*)\)")
_FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\)")
_THINK_CLOSE_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Drop everything up to and including the last </think>.

    Handles both `<think>...</think>` pairs and the dangling-`</think>` case
    (vLLM reasoning-parser quirk noted in CLAUDE.md).
    """
    if "</think>" not in text:
        return text
    return text.rsplit("</think>", 1)[1].lstrip()


@dataclass
class ParsedPlannerOutput:
    code: str | None
    final: str | None
    final_var: str | None
    truncated_fence: bool = False

    @property
    def is_final(self) -> bool:
        return self.final is not None or self.final_var is not None


def parse(output: str) -> ParsedPlannerOutput:
    """Extract code block and any FINAL sentinel from planner output.

    Code and FINAL_VAR can co-exist: the model often binds the final variable
    inside the same fenced block. In that case, both fields are populated and
    the orchestrator runs the code BEFORE resolving the var.

    A literal FINAL("...") together with code is ambiguous (which is the
    answer — what's already there or what the code computes?). We resolve in
    favor of the code: if code is present, ignore FINAL("...") this turn; the
    model can re-emit FINAL on the next turn after seeing exec output.
    """
    cleaned = _strip_thinking(output)

    fence_m = _FENCE_RE.search(cleaned)
    code = fence_m.group(1) if fence_m else None

    # Detect an OPEN-but-not-closed fence: an opening ``` exists past where the
    # last closed fence ends. Indicates the planner was truncated mid-block.
    truncated = False
    if code is None:
        truncated = _OPEN_FENCE_RE.search(cleaned) is not None
    else:
        # Closed fence found — but maybe a SECOND fence was opened after it and
        # never closed. Check the tail.
        tail = cleaned[fence_m.end() :]
        truncated = _OPEN_FENCE_RE.search(tail) is not None

    final_var_m = _FINAL_VAR_RE.search(cleaned)
    final_var = final_var_m.group(1) if final_var_m else None

    final: str | None = None
    if final_var is None and code is None and not truncated:
        finals = _FINAL_RE.findall(cleaned)
        if finals:
            raw = finals[-1].strip()
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
                raw = raw[1:-1]
            final = raw

    return ParsedPlannerOutput(
        code=code, final=final, final_var=final_var, truncated_fence=truncated
    )
