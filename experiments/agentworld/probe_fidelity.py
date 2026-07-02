#!/usr/bin/env python3
"""Fidelity probe: how far does Qwen-AgentWorld's *simulated* sandbox diverge from the real one?

Reward-trust counterpart to the game-rlvr proposal. A deterministic game engine gives a
*verifiable* reward; a language world model gives a *plausible-but-not-verifiable* observation.
This measures exactly how plausible, on our own tasks.

Framing decision (see README + repo prompts/): claw's `sandbox_shell_exec` is a one-shot,
structured tool call — command in, `{exit_code, stdout, stderr}` out. That maps onto AgentWorld's
**MCP domain** (tool call -> predicted tool result), NOT the Terminal domain (a two-phase tmux
*screen* simulator with prompt/echo/wait turns). MCP framing diffs apples-to-apples and matches
our actual use case: mocking a sandbox tool's response inside an eval loop.

Ground truth is free: yesterday's Ornith-9B claw run left real (command -> result) tapes in the
`tool_dispatch` records of every T100-T104 trace. We teacher-force on the REAL prior observations
so simulation errors don't compound, and diff each predicted result against the recorded one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # only needed for the live run; parsing/diffing works without it.

HERE = Path(__file__).resolve().parent
MCP_SYSTEM_PATH = HERE / "prompts" / "mcp_system_prompt.txt"

RESPONSE_TAG = "predicted_observation"
RESPONSE_MARKER = "**Environment Observation:**"

# The one tool we simulate. Mirrors claw's sandbox_shell_exec, with an explicit return schema so
# the world model emits our exact envelope. The MCP system prompt interpolates this at
# {tool_definitions}; an explicit output schema is priority-1 per that prompt's format rules.
TOOL_DEFINITIONS = json.dumps([{
    "name": "sandbox_shell_exec",
    "description": "Execute a shell command inside a Linux container sandbox (one-shot, "
                   "non-interactive) and return its result.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The shell command to run."}},
        "required": ["command"],
    },
    "returns": {
        "type": "object",
        "description": "Captured result after the command finishes.",
        "properties": {
            "exit_code": {"type": "integer", "description": "Process exit status (0 = success)."},
            "stdout": {"type": "string", "description": "Captured standard output, verbatim."},
            "stderr": {"type": "string", "description": "Captured standard error, verbatim."},
        },
        "required": ["exit_code", "stdout", "stderr"],
    },
}], indent=2)


@dataclass
class Step:
    idx: int
    command: str
    real_stdout: str
    real_stderr: str
    real_exit: int

    @property
    def real_envelope(self) -> dict:
        return {"exit_code": self.real_exit, "stdout": self.real_stdout, "stderr": self.real_stderr}


@dataclass
class StepResult:
    idx: int
    command: str
    parsed_ok: bool         # did the model emit a JSON envelope we could parse?
    exit_match: bool
    exact_stdout: bool
    stdout_seq: float       # char-level similarity on stdout
    stdout_lines: float     # Jaccard over stripped non-empty stdout lines
    real_len: int
    sim_len: int


# ----- official parser (ported from repo eval/lwm_eval_utils/output_parser.py) -----

def _strip_think(text: str) -> str:
    # qwen3 reasoning-parser keeps think out of `content`; this is belt-and-suspenders for the
    # right-unclosed case, bounded by the response tag so we never eat the real answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in text.lower():
        m = re.search(rf"<{RESPONSE_TAG}>", text, re.IGNORECASE)
        cut = m.start() if m else len(text)
        text = re.sub(r"<think>.*", "", text[:cut], flags=re.DOTALL | re.IGNORECASE) + text[cut:]
    return text


def parse_prediction(raw: str) -> str:
    """Extract the predicted observation body: strip think, take last response tag, strip marker."""
    cleaned = _strip_think(raw or "")
    starts = list(re.finditer(rf"<{RESPONSE_TAG}>", cleaned, re.IGNORECASE))
    if starts:
        s = starts[-1].end()
        close = re.search(rf"</{RESPONSE_TAG}>", cleaned[s:], re.IGNORECASE)
        body = cleaned[s:s + close.start()] if close else cleaned[s:]
    else:
        body = cleaned
    body = body.strip()
    if body.startswith(RESPONSE_MARKER):
        body = body[len(RESPONSE_MARKER):].strip()
    else:
        body = body.replace(RESPONSE_MARKER, "").strip()
    return body


def predicted_envelope(body: str) -> dict | None:
    """Pull the predicted {exit_code, stdout, stderr} object out of the observation.

    Takes the LAST valid envelope: this model reasons in prose and often quotes prior turns'
    outputs before emitting its final prediction, so the first {...} with a stdout key is usually
    a quoted history entry, not the answer.
    """
    body = body.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?|```$", "", body).strip()
    candidates = [body] + _all_json_objects(body)
    last = None
    for candidate in candidates:
        try:
            # strict=False: terminal stdout routinely contains literal newlines/tabs inside the
            # JSON string values, which strict JSON rejects.
            obj = json.loads(candidate, strict=False)
            if isinstance(obj, dict) and "stdout" in obj:
                last = obj  # keep scanning; the final envelope wins
        except json.JSONDecodeError:
            continue
    return last


def _all_json_objects(text: str) -> list[str]:
    """Every balanced top-level {...} span, in order."""
    spans, depth, start = [], 0, -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start:i + 1])
    return spans


# ----- trace parsing + prompt construction -----

def parse_trace(path: Path) -> list[Step]:
    steps: list[Step] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("type") != "tool_dispatch" or rec.get("tool_name") != "sandbox_shell_exec":
            continue
        cmd = (rec.get("request_body") or {}).get("command", "")
        resp = rec.get("response_body") or {}
        steps.append(Step(
            idx=len(steps), command=cmd,
            real_stdout=resp.get("stdout", "") or "",
            real_stderr=resp.get("stderr", "") or "",
            real_exit=resp.get("exit_code", 0),
        ))
    return steps


def _tool_call_str(command: str) -> str:
    return json.dumps({"name": "sandbox_shell_exec", "arguments": {"command": command}})


def build_messages(steps: list[Step], i: int, n_demo: int) -> list[dict]:
    """MCP framing. n_demo leading steps become {demonstrations} (format grounding); steps
    [n_demo, i) become teacher-forced Historical Context; step i is the current call."""
    system = MCP_SYSTEM_PATH.read_text()
    demos = ""
    if n_demo:
        demo_turns = [
            f"### Turn {s.idx + 1}\n**Tool Call:**\n{_tool_call_str(s.command)}\n"
            f"{RESPONSE_MARKER}\n<{RESPONSE_TAG}>{json.dumps(s.real_envelope)}</{RESPONSE_TAG}>"
            for s in steps[:n_demo]
        ]
        demos = "\n\n# Demonstrations\n\n" + "\n\n".join(demo_turns) + "\n"
    system = system.replace("{tool_definitions}", TOOL_DEFINITIONS).replace("{demonstrations}", demos)

    history = [
        f"### Turn {s.idx + 1}\n**Tool Call:**\n{_tool_call_str(s.command)}\n"
        f"{RESPONSE_MARKER}\n{json.dumps(s.real_envelope)}"
        for s in steps[n_demo:i]
    ]
    user = ""
    if history:
        user += "# Historical Context\n\n" + "\n\n".join(history) + "\n\n"
    user += (
        f"# Current Tool Call\n\n### Turn {steps[i].idx + 1}\n**Tool Call:**\n"
        f"{_tool_call_str(steps[i].command)}\n\n"
        f"Predict the tool result. Respond with {RESPONSE_MARKER} followed by "
        f"<{RESPONSE_TAG}>...</{RESPONSE_TAG}> containing a JSON object "
        f'{{"exit_code", "stdout", "stderr"}}.'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ----- scoring -----

def _lines(s: str) -> set[str]:
    return {ln.strip() for ln in s.splitlines() if ln.strip()}


def score_step(step: Step, env: dict | None) -> StepResult:
    real = step.real_stdout
    if env is None:
        return StepResult(step.idx, step.command[:70], False, False, False, 0.0, 0.0,
                          len(real), 0)
    sim = str(env.get("stdout", ""))
    rl, sl = _lines(real), _lines(sim)
    overlap = (len(rl & sl) / len(rl | sl)) if (rl or sl) else 1.0
    return StepResult(
        idx=step.idx, command=step.command[:70], parsed_ok=True,
        exit_match=(env.get("exit_code") == step.real_exit),
        exact_stdout=(real.rstrip() == sim.rstrip()),
        stdout_seq=round(SequenceMatcher(None, real.rstrip(), sim.rstrip()).ratio(), 3),
        stdout_lines=round(overlap, 3),
        real_len=len(real), sim_len=len(sim),
    )


def run(trace: Path, endpoint: str, model: str, max_steps: int, n_demo: int,
        temperature: float, out: Path | None, max_tokens: int, no_think: bool) -> None:
    steps = parse_trace(trace)
    if not steps:
        print(f"No sandbox_shell_exec steps in {trace}", file=sys.stderr)
        sys.exit(1)
    if max_steps:
        steps = steps[:max_steps]
    probe_steps = list(range(n_demo, len(steps)))
    print(f"Loaded {len(steps)} real steps from {trace.name}; "
          f"{n_demo} as format demo, probing {len(probe_steps)}.\n")

    if OpenAI is None:
        print("openai not installed — dry run (parsed tape only):", file=sys.stderr)
        for s in steps:
            print(f"[{s.idx}] $ {s.command[:70]}  -> exit {s.real_exit}, "
                  f"{len(s.real_stdout)}B out / {len(s.real_stderr)}B err")
        return

    client = OpenAI(base_url=endpoint, api_key="local")
    results: list[StepResult] = []
    for i in probe_steps:
        msgs = build_messages(steps, i, n_demo)
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if no_think else {}
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens,
            extra_body=extra)
        msg = resp.choices[0].message
        # qwen3 reasoning parser splits think into reasoning_content; on an unclosed </think>
        # the whole answer lands there with content empty (the documented claw-eval fallback).
        raw = msg.content or getattr(msg, "reasoning_content", None) or ""
        body = parse_prediction(raw)
        env = predicted_envelope(body)
        r = score_step(steps[i], env)
        results.append(r)
        if not r.parsed_ok:
            mark = "NO-PARSE"
        elif r.exact_stdout:
            mark = "EXACT"
        else:
            mark = f"seq={r.stdout_seq:.2f} lines={r.stdout_lines:.2f} exit={'Y' if r.exit_match else 'N'}"
        print(f"[{steps[i].idx:>2}] {mark:<34} $ {steps[i].command[:60]}")

    n = len(results)
    summary = {
        "trace": str(trace), "model": model, "domain": "mcp", "n_probed": n,
        "parse_rate": round(sum(r.parsed_ok for r in results) / n, 3),
        "exit_match_rate": round(sum(r.exit_match for r in results) / n, 3),
        "exact_stdout_rate": round(sum(r.exact_stdout for r in results) / n, 3),
        "mean_stdout_seq": round(sum(r.stdout_seq for r in results) / n, 3),
        "mean_stdout_lines": round(sum(r.stdout_lines for r in results) / n, 3),
    }
    print("\n=== FIDELITY SUMMARY (sim vs real sandbox) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if out:
        out.write_text(json.dumps({"summary": summary, "steps": [asdict(r) for r in results]}, indent=2))
        print(f"\nWrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--endpoint", default="http://localhost:8010/v1")
    ap.add_argument("--model", default="Qwen/Qwen-AgentWorld-35B-A3B")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = all steps")
    ap.add_argument("--n-demo", type=int, default=1,
                    help="leading steps used as format demonstrations (default 1)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--think", action="store_true",
                    help="leave <think> on (default off: keeps full raw output in content so an "
                         "unclosed/truncated </think> can't blank the response)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.trace, args.endpoint, args.model, args.max_steps, args.n_demo,
        args.temperature, args.out, args.max_tokens, no_think=not args.think)


if __name__ == "__main__":
    main()
