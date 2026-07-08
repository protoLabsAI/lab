"""Canonical trajectory schema for the protoLabs agentic-distill corpus.

Every source — public dataset or Ornith-generated — normalizes into `Trajectory`,
so the corpus is ONE shape, reusable across any student (not just Qwen3.5-2B).

Design choices that make it reusable:
- OpenAI/ShareGPT-style `messages` with `tool_calls` = our serving format → trains directly.
- `teacher` is first-class → ablations filter by teacher-consistency (Ornith-only vs blended)
  without re-parsing. This is the lever the mixing-ratio pilot needs.
- `verified` / `reward` separate the deterministically-verified core from imitation breadth.
- `license_note` is METADATA ONLY, never a gate (see ../DATASETS.md — we train on anything;
  we just don't re-host non-commercial rows verbatim).
"""
from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: Optional[str] = None
    name: str
    arguments: dict[str, Any] | str  # dict preferred; str tolerated pre-normalization


class Message(BaseModel):
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None  # set when role == "tool"


class Trajectory(BaseModel):
    id: str                                  # "<source>__<origin_id>" — globally unique
    source: str                              # toolace | apigen-mt | orca | ornith-tau | ...
    teacher: str                             # ornith-35b | gpt-4 | deepseek-v2 | human | unknown
    domain: str = "unknown"                  # retail | crm | ops | finance | web | science | ...
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)  # OpenAI tool schemas in-context
    verified: bool = False                   # deterministic verifier confirmed success
    reward: Optional[float] = None           # from a verified env, else None
    thinking: Optional[Literal["on", "off"]] = None
    split: Literal["train", "held_out"] = "train"
    license_note: Optional[str] = None       # metadata only — NOT a gate
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m.role == "assistant")

    @property
    def n_tool_calls(self) -> int:
        return sum(len(m.tool_calls or []) for m in self.messages)

    @model_validator(mode="after")
    def _check(self) -> "Trajectory":
        if not self.messages:
            raise ValueError(f"{self.id}: empty messages")
        if self.messages[-1].role != "assistant":
            # must end on the assistant turn we want the student to learn
            raise ValueError(f"{self.id}: must end on an assistant turn")
        # tool_call_id linkage is optional: most SFT trajectory datasets use ShareGPT
        # text format without structured ids. We train on the token sequence, not an
        # OpenAI API replay, so absence is fine.
        return self

    def prompt_hash(self) -> str:
        """Hash of the FIRST user turn — for contamination checks vs held-out eval
        prompts. (Env-trajectory datasets share a boilerplate opener, so this is NOT
        a dedup key — use content_hash for that.)"""
        first_user = next((m.content or "" for m in self.messages if m.role == "user"), "")
        norm = " ".join(first_user.lower().split())
        return hashlib.sha256(norm.encode()).hexdigest()[:16]

    def content_hash(self) -> str:
        """Hash of the full role+content sequence — the dedup key. Distinguishes
        trajectories that share a boilerplate first turn but differ downstream."""
        blob = "\n".join(f"{m.role}:{(m.content or '')}" for m in self.messages)
        return hashlib.sha256(" ".join(blob.lower().split()).encode()).hexdigest()[:16]
