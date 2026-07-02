#!/usr/bin/env python3
"""Phase 1 of the scaffold-transfer probe: practice a claw task inside AgentWorld, author a scaffold.

The policy (Ornith via gateway `protolabs/smart`) runs the task as a normal tool-calling agent, but
the sandbox tool results are *simulated by AgentWorld* (:8010) instead of a real Docker container.
After the episode the policy reflects and writes a reusable **scaffold** — a task-category harness
(decomposition, tool-call workflow, common errors + recovery, verification steps). That scaffold is
then injected as `system_prompt_prefix` in the real-sandbox arm (run_transfer.py) to measure transfer.

The point is deliberately NOT that AgentWorld's simulated outputs are correct — our fidelity probe
showed it reproduces a sandbox's *shape*, not its *state*. The question is whether a workflow learned
against a hallucinated environment still raises real-sandbox pass-rate. Sim shapes the scaffold;
reality (run_transfer.py) grades it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_fidelity import MCP_SYSTEM_PATH, RESPONSE_MARKER, RESPONSE_TAG, parse_prediction  # noqa: E402

# The three sandbox tools claw injects for terminal/SWE tasks (mirrors claw sandbox_tools.py).
SANDBOX_TOOLS = [
    {"type": "function", "function": {
        "name": "sandbox_shell_exec",
        "description": "Execute a shell command in the sandbox and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The shell command to execute."}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "sandbox_file_read",
        "description": "Read a file from the sandbox filesystem.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path to read."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "sandbox_file_write",
        "description": "Write content to a file in the sandbox filesystem.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
]
TOOL_DEFS_JSON = json.dumps([t["function"] for t in SANDBOX_TOOLS], indent=2)

SCAFFOLD_REFLECT = """You just attempted the task above by practicing in a simulated sandbox.

Write a concise, reusable **scaffold** for this *category* of task — the orchestration a strong agent
should follow next time, BEFORE seeing the specific files. Cover:
1. Decomposition — the ordered sub-goals.
2. Tool-call workflow — what to inspect first, in what order, which tools.
3. Failure modes + recovery — the errors this task class tends to hit and how to react.
4. Verification — how to confirm the solution is actually correct before finishing.

Be specific to the task category but do NOT hardcode any specific value, filename, or expected output
you saw (those may differ in the real environment). Output ONLY the scaffold, as terse imperative
guidance (no preamble). Target 120-220 words."""


COLD_REFLECT = """Write a concise, reusable **scaffold** for this *category* of task — the orchestration
a strong agent should follow, BEFORE seeing the specific files. Cover:
1. Decomposition — the ordered sub-goals.
2. Tool-call workflow — what to inspect first, in what order, which tools.
3. Failure modes + recovery — the errors this task class tends to hit and how to react.
4. Verification — how to confirm the solution is actually correct before finishing.

Be specific to the task category but do NOT hardcode any specific value, filename, or expected output.
Output ONLY the scaffold, as terse imperative guidance (no preamble). Target 120-220 words."""


def author_cold(task_text: str, policy: OpenAI, policy_model: str, temperature: float) -> str:
    """Placebo control: the policy authors a scaffold from the task prompt alone — NO AgentWorld
    practice. Isolates the contribution of sim practice vs. the model just reasoning about the task."""
    msgs = [
        {"role": "user", "content": task_text},
        {"role": "user", "content": COLD_REFLECT},
    ]
    r = policy.chat.completions.create(
        model=policy_model, messages=msgs, temperature=temperature, max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    m = r.choices[0].message
    return (m.content or getattr(m, "reasoning_content", None) or "").strip()


def load_task(task_dir: Path) -> tuple[str, str]:
    y = yaml.safe_load((task_dir / "task.yaml").read_text())
    return y["task_id"], y["prompt"]["text"]


def simulate_tool(aw: OpenAI, aw_model: str, history: list[dict], name: str,
                  arguments: dict, temperature: float) -> str:
    """Ask AgentWorld to predict the result of one tool call, given the running history."""
    system = (MCP_SYSTEM_PATH.read_text()
              .replace("{tool_definitions}", TOOL_DEFS_JSON)
              .replace("{demonstrations}", ""))
    turns = []
    for h in history:
        turns.append(f"**Tool Call:**\n{json.dumps({'name': h['name'], 'arguments': h['arguments']})}\n"
                     f"{RESPONSE_MARKER}\n{h['observation']}")
    user = ""
    if turns:
        user += "# Historical Context\n\n" + "\n\n".join(turns) + "\n\n"
    user += (f"# Current Tool Call\n\n**Tool Call:**\n{json.dumps({'name': name, 'arguments': arguments})}\n\n"
             f"Predict the tool result. Respond with {RESPONSE_MARKER} then <{RESPONSE_TAG}>...</{RESPONSE_TAG}>.")
    r = aw.chat.completions.create(
        model=aw_model, temperature=temperature, max_tokens=4096,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    m = r.choices[0].message
    raw = m.content or getattr(m, "reasoning_content", None) or ""
    obs = parse_prediction(raw)
    # strip a wrapping predicted_observation tag remnant / fences for clean feed-back
    obs = re.sub(r"</?%s>" % RESPONSE_TAG, "", obs).strip()
    return obs[:6000] or "(no output)"


def run_episode(task_text: str, policy: OpenAI, policy_model: str, aw: OpenAI, aw_model: str,
                max_turns: int, temperature: float, log) -> list[dict]:
    messages = [
        {"role": "system", "content":
            "You are a coding agent working in a Linux sandbox at /workspace. Use the provided tools "
            "to complete the task. Inspect before acting; verify your work before finishing."},
        {"role": "user", "content": task_text},
    ]
    history: list[dict] = []
    for turn in range(max_turns):
        r = policy.chat.completions.create(
            model=policy_model, messages=messages, tools=SANDBOX_TOOLS, temperature=temperature,
            max_tokens=4096, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        log(f"[turn {turn+1}/{max_turns}] {len(calls)} tool_call(s)"
            + (f" | {(msg.content or '')[:70]!r}" if msg.content else ""))
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in calls] if calls else None})
        if not calls:
            break  # policy decided it's done
        for tc in calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            obs = simulate_tool(aw, aw_model, history, tc.function.name, args, temperature)
            history.append({"name": tc.function.name, "arguments": args, "observation": obs})
            log(f"    {tc.function.name}({json.dumps(args)[:60]}) -> {obs[:60]!r}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
    # reflection -> scaffold
    messages.append({"role": "user", "content": SCAFFOLD_REFLECT})
    r = policy.chat.completions.create(
        model=policy_model, messages=messages, temperature=temperature, max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    m = r.choices[0].message
    scaffold = (m.content or getattr(m, "reasoning_content", None) or "").strip()
    return scaffold, history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-dir", required=True, type=Path)
    ap.add_argument("--policy-endpoint", default="http://ava:4000/v1")
    ap.add_argument("--policy-model", default="protolabs/smart")
    ap.add_argument("--policy-key", default="dummy")
    ap.add_argument("--aw-endpoint", default="http://localhost:8010/v1")
    ap.add_argument("--aw-model", default="Qwen/Qwen-AgentWorld-35B-A3B")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out-dir", type=Path, default=Path("scaffolds"))
    ap.add_argument("--cold", action="store_true",
                    help="placebo control: author a scaffold from the task prompt alone, no AgentWorld practice")
    args = ap.parse_args()

    task_id, task_text = load_task(args.task_dir)
    policy = OpenAI(base_url=args.policy_endpoint, api_key=args.policy_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def log(s): print(s, flush=True)

    if args.cold:
        log(f"=== {task_id} — COLD scaffold (no practice, placebo control) ===")
        scaffold = author_cold(task_text, policy, args.policy_model, args.temperature)
        out = args.out_dir / f"{task_id}_cold.md"
        out.write_text(scaffold)
        log(f"Wrote cold scaffold ({len(scaffold)} chars) -> {out}\n")
        log(scaffold)
        return

    aw = OpenAI(base_url=args.aw_endpoint, api_key="local")
    best = None
    for ep in range(args.episodes):
        log(f"\n=== {task_id} — practice episode {ep+1}/{args.episodes} (AgentWorld sandbox) ===")
        scaffold, history = run_episode(task_text, policy, args.policy_model, aw, args.aw_model,
                                        args.max_turns, args.temperature, log)
        out = args.out_dir / f"{task_id}_ep{ep+1}.md"
        out.write_text(scaffold)
        log(f"\n--- scaffold ep{ep+1} ({len(scaffold)} chars, {len(history)} sim tool calls) -> {out} ---")
        log(scaffold)
        best = out
    # canonical pointer = last episode's scaffold
    (args.out_dir / f"{task_id}.md").write_text(best.read_text())
    log(f"\nWrote canonical scaffold: {args.out_dir / f'{task_id}.md'}")


if __name__ == "__main__":
    main()
