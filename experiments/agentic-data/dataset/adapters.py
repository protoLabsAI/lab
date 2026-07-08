"""Per-source normalizers: raw dataset row -> canonical Trajectory.

Column mappings below were verified by streaming a live row from each source
(2026-07-06). Adding a new source = register an adapter + verify its columns first;
unverified sources raise by design (grounding rule — no unseen mapping ships).
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterator

from schema import Message, ToolCall, Trajectory

Adapter = Callable[[dict, dict], Iterator[Trajectory]]  # (row, source_cfg) -> trajectories
_REGISTRY: dict[str, Adapter] = {}


def register(*keys: str):
    def deco(fn: Adapter) -> Adapter:
        for k in keys:
            _REGISTRY[k] = fn
        return fn
    return deco


def get(key: str) -> Adapter:
    if key not in _REGISTRY:
        raise KeyError(f"no adapter for '{key}' — add one in adapters.py")
    return _REGISTRY[key]


# --- shared helpers ---------------------------------------------------------

_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system",
             "tool": "tool", "function": "tool", "observation": "tool",
             "user": "user", "assistant": "assistant"}


def _parse_tools(raw: Any) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        out = []
        for t in raw:
            if isinstance(t, str):
                try:
                    t = json.loads(t)
                except Exception:
                    continue
            if isinstance(t, dict):
                out.append(t)
        return out
    return []


def _sharegpt_messages(convs: list[dict], system: str | None) -> list[Message]:
    out: list[Message] = []
    if system:
        out.append(Message(role="system", content=system))
    for turn in convs:
        role = _ROLE_MAP.get(turn.get("from") or turn.get("role", ""), "user")
        # a leading system turn inside conversations shouldn't double with `system`
        out.append(Message(role=role, content=turn.get("value") or turn.get("content")))
    return out


def _traj(cfg: dict, origin_id: str, messages: list[Message],
          tools: list[dict] | None = None, **kw) -> Trajectory:
    key = cfg["_key"]
    return Trajectory(
        id=f"{key}__{origin_id}", source=key,
        teacher=cfg.get("teacher", "unknown"), domain=cfg.get("domain", "unknown"),
        messages=messages, tools=tools or [],
        verified=bool(cfg.get("verified", False)),
        license_note=cfg.get("license_note"), **kw,
    )


def _oid(row: dict, *fields: str) -> str:
    for f in fields:
        if row.get(f) is not None:
            return str(row[f])
    return str(abs(hash(json.dumps(row.get("conversations", row), sort_keys=True, default=str))) % (10 ** 12))


# --- ShareGPT-family: system(str) + conversations[{from,value}] + tools(str|list) ---
# Verified shapes: ToolACE, hermes_reasoning_tool_use, APIGen-MT-5k, AgentTraj-L.

@register("toolace", "hermes_reasoning_tool_use", "apigen_mt", "agenttraj_l")
def _sharegpt(row: dict, cfg: dict) -> Iterator[Trajectory]:
    convs = row.get("conversations") or row.get("messages")
    if not convs:
        return
    msgs = _sharegpt_messages(convs, row.get("system"))
    # trim trailing non-assistant turns (env observations after the last action)
    while msgs and msgs[-1].role != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return
    yield _traj(cfg, _oid(row, "id", "item_id", "uuid"), msgs, tools=_parse_tools(row.get("tools")))


# --- When2Call: bespoke decision format -> single-turn trajectory ---
# question + correct_answer{cannot_answer|direct|tool_call} + answers{direct,tool_call} + tools[str]

@register("when2call")
def _when2call(row: dict, cfg: dict) -> Iterator[Trajectory]:
    q = row.get("question")
    if not q:
        return
    ans = row.get("answers") or {}
    ca = row.get("correct_answer")
    # gold assistant turn: the response class the row marks correct
    if ca == "tool_call":
        gold = ans.get("tool_call")
    elif ca == "direct":
        gold = ans.get("direct")
    else:  # cannot_answer / clarify — teach the abstention/clarification text
        gold = ans.get("cannot_answer") or ans.get("clarify") or ans.get("direct")
    if not gold:
        return
    msgs = [Message(role="user", content=q), Message(role="assistant", content=str(gold))]
    yield _traj(cfg, _oid(row, "uuid", "source_id"), msgs,
                tools=_parse_tools(row.get("tools")), meta={"decision": ca})


# --- adapters still needing column verification (raise until checked) --------

def _unverified(name: str) -> Adapter:
    def fn(row: dict, cfg: dict) -> Iterator[Trajectory]:
        raise NotImplementedError(
            f"adapter '{name}': inspect the real card columns, map to Trajectory, "
            f"then register a real adapter. No unverified mapping ships.")
        yield  # pragma: no cover
    return fn


for _k in ("toolbench", "openthoughts3"):
    _REGISTRY[_k] = _unverified(_k)


# --- smoltalk2 subsets: OpenAI `messages` + tools in chat_template_kwargs.xml_tools ---
# Strong teacher (Qwen3-32B/DeepSeek-V3), in-family. smolagents tool-calling + reasoning.

@register("smoltalk2_smolagents", "smoltalk2_xlam")
def _smoltalk2(row: dict, cfg: dict) -> Iterator[Trajectory]:
    msgs_raw = row.get("messages")
    if not msgs_raw:
        return
    msgs = [Message(role=m["role"], content=m.get("content"))
            for m in msgs_raw if m.get("role") in _ROLE_MAP and (m.get("content") or "").strip()]
    while msgs and msgs[-1].role != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return
    tools = _parse_tools((row.get("chat_template_kwargs") or {}).get("xml_tools"))
    yield _traj(cfg, _oid(row, "id"), msgs, tools=tools)


# --- OS-Genesis: ShareGPT GUI traces; strip <image> tokens, keep a11y-tree TEXT + actions ---

@register("os_genesis")
def _os_genesis(row: dict, cfg: dict) -> Iterator[Trajectory]:
    convs = row.get("conversations")
    if not convs:
        return
    msgs = []
    for turn in convs:
        role = _ROLE_MAP.get(turn.get("from", ""), "user")
        c = (turn.get("value") or "").replace("<image>", "").strip()
        if c or role == "assistant":
            msgs.append(Message(role=role, content=c))
    while msgs and msgs[-1].role != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return
    yield _traj(cfg, _oid(row, "id"), msgs)


# --- orca-agentinstruct: `messages` is a JSON STRING of OpenAI turns.
# NOTE: general instruction/reasoning content (not tool-use) — instruction fuel, not agentic core.

@register("orca_agentinstruct")
def _orca(row: dict, cfg: dict) -> Iterator[Trajectory]:
    raw = row.get("messages")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return
    if not raw:
        return
    msgs = [Message(role=m["role"], content=m.get("content"))
            for m in raw if m.get("role") in _ROLE_MAP and (m.get("content") or "").strip()]
    while msgs and msgs[-1].role != "assistant":
        msgs.pop()
    if len(msgs) < 2:
        return
    yield _traj(cfg, _oid(row, "id"), msgs)


# --- Ornith-generated core: local JSONL, already canonical ------------------

@register("ornith_tau")
def _ornith(row: dict, cfg: dict) -> Iterator[Trajectory]:
    yield Trajectory(**row)
