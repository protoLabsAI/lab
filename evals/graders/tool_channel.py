"""Deterministic grader for tool-call channel correctness.

channel_correctness asks: did the model call the required tool(s) via the
*structured* tool_calls field (good), emit them as text in content (leak), or
not call them at all? This is inherently a deterministic check on the trace —
an LLM judge reading serialized output cannot distinguish a structured call
from text and tends to mislabel correct tool-sourced answers as hallucinations.

Expects task["expected"] shaped like:
    expected:
      tools: [calculator, current_time]   # required tools (structured)
      no_tool: false                       # true => no tool should be called
      forbidden_tools: [gmail_draft]       # using these caps the score at 0.0
      args:                                # optional substring checks per tool
        gmail_send_message:
          to: "sarah@company.com"
          body: ["27B", "35B"]            # list => all substrings must appear

And reads from task_output (produced by run_agent):
    _tools_called: [names...]              # structured calls
    _tool_calls_detail: [{name, arguments}]
    _text_leaked_tool: bool                # tool syntax found in content text
    output: final assistant text
"""

from __future__ import annotations

from graders.base import Grader, GradeResult


def _arg_ok(call_args: dict, want: dict) -> bool:
    """All wanted substrings appear in the corresponding argument value."""
    if not isinstance(call_args, dict):
        return False
    for k, v in want.items():
        actual = str(call_args.get(k, "")).lower()
        needles = v if isinstance(v, list) else [v]
        if not all(str(n).lower() in actual for n in needles):
            return False
    return True


class ToolChannelGrader(Grader):
    def __init__(self, dimension: str = "channel_correctness", threshold: float = 0.75):
        self.dimension = dimension
        self.threshold = threshold

    def grade(self, task_input: dict, task_output: dict, expected: dict | None = None) -> GradeResult:
        expected = expected or {}
        called = list(task_output.get("_tools_called", task_output.get("tools_called", [])))
        details = task_output.get("_tool_calls_detail", [])
        leaked = bool(task_output.get("_text_leaked_tool", False))

        req = expected.get("tools", [])
        no_tool = expected.get("no_tool", False)
        forbidden = expected.get("forbidden_tools", [])
        arg_specs = expected.get("args", {})

        # Case 1: no tool should be called (farewell etc.)
        if no_tool:
            if not called and not leaked:
                return GradeResult.from_threshold(self.dimension, 1.0, self.threshold,
                    reasoning="No tool called, as required.")
            if leaked and not called:
                return GradeResult.from_threshold(self.dimension, 0.5, self.threshold,
                    reasoning="Emitted tool-call text though none was needed.")
            return GradeResult.from_threshold(self.dimension, 0.5, self.threshold,
                reasoning=f"Unnecessarily called tool(s): {called}")

        # Forbidden tool used => hard fail (e.g. gmail_draft instead of send)
        if any(f in called for f in forbidden):
            return GradeResult.from_threshold(self.dimension, 0.0, self.threshold,
                reasoning=f"Used forbidden tool {[f for f in forbidden if f in called]} instead of required {req}.")

        # Fraction of required tools called via the STRUCTURED channel
        hits = sum(1 for t in req if t in called)
        frac = hits / len(req) if req else (1.0 if called else 0.0)

        # Argument correctness for the structured calls (averaged into the score)
        if arg_specs:
            ok = 0
            for tname, want in arg_specs.items():
                d = next((c for c in details if c.get("name") == tname), None)
                if d and _arg_ok(d.get("arguments"), want):
                    ok += 1
            arg_frac = ok / len(arg_specs)
            score = round(0.5 * frac + 0.5 * arg_frac, 3)
            reason = f"structured {hits}/{len(req)} required tools; args {ok}/{len(arg_specs)} correct"
        else:
            score = round(frac, 3)
            reason = f"structured {hits}/{len(req)} required tools called"

        # If nothing structured but content leaked tool syntax, give partial credit
        if score == 0.0 and leaked:
            return GradeResult.from_threshold(self.dimension, 0.5, self.threshold,
                reasoning="Tool call emitted as text in content, not structured tool_calls.")

        return GradeResult.from_threshold(self.dimension, score, self.threshold, reasoning=reason)
