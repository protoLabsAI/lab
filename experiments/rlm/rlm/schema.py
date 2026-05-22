"""Trajectory, Turn, and graph-state schemas.

Trajectory is the canonical SFT-corpus record — keep it stable.
Bumping `schema_version` is a breaking change; old trajectories should be migrated.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class PlannerTurn(BaseModel):
    role: Literal["planner"] = "planner"
    request_messages: list[dict[str, Any]]
    response: str
    reasoning: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: float = 0.0


class ExecTurn(BaseModel):
    role: Literal["exec"] = "exec"
    code: str
    stdout: str
    stderr: str
    truncated: bool = False
    wall_ms: float = 0.0


class LeafCallTurn(BaseModel):
    role: Literal["leaf_call"] = "leaf_call"
    subquery: str
    slice_hash: str | None = None
    slice_preview: str | None = None
    model: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    wall_ms: float = 0.0


Turn = PlannerTurn | ExecTurn | LeafCallTurn


class Trajectory(BaseModel):
    """One full RLM session, persisted as a single JSONL row."""

    schema_version: int = SCHEMA_VERSION
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = Field(default_factory=time.time)
    query: str
    context_meta: dict[str, Any] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)
    final: str | None = None
    final_var: str | None = None
    terminated_reason: Literal["final", "budget", "error", "max_steps"] | None = None
    error: str | None = None
    totals: dict[str, float] = Field(default_factory=dict)


def _env(*names: str, default: str = "") -> str:
    import os

    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


class RLMConfig(BaseModel):
    """Run-time configuration for a single RLM invocation.

    Defaults route through the protoLabs LiteLLM gateway on the ava node:
    - traces land in Langfuse automatically
    - aliases (`protolabs/smart`, `protolabs/fast`) decouple us from the
      currently-loaded model behind each port
    - thinking_normalizer.py callback strips most `<think>` blocks (we still
      run our own _strip_thinking as defense-in-depth)

    Override base_url to `http://localhost:8000/v1` for direct-vLLM (no Langfuse).
    Set GATEWAY_API_KEY (or LITELLM_API_KEY) env var; sourced from ~/.proto/.env.
    """

    planner_base_url: str = Field(
        default_factory=lambda: _env("GATEWAY_URL", default="http://ava:4000/v1")
    )
    planner_model: str = "protolabs/smart"
    planner_api_key: str = Field(
        default_factory=lambda: _env("GATEWAY_API_KEY", "LITELLM_API_KEY", default="EMPTY")
    )

    leaf_base_url: str = Field(
        default_factory=lambda: _env("GATEWAY_URL", default="http://ava:4000/v1")
    )
    leaf_model: str = "protolabs/fast"
    leaf_api_key: str = Field(
        default_factory=lambda: _env("GATEWAY_API_KEY", "LITELLM_API_KEY", default="EMPTY")
    )

    # Budget guards — orchestration-level, not prompt-level
    max_steps: int = 50  # planner turns; verbose 27B-thinking needs headroom
    max_tokens: int = 400_000
    max_wall_seconds: float = 600.0
    max_depth: int = 1  # paper's default

    # Per-call timeouts — bound any single LM call so a stuck thinking turn
    # can't blow through the wall budget by itself. On timeout, OpenAI SDK
    # raises APITimeoutError and closes the httpx connection, which lets
    # vLLM detect the disconnect via request.is_disconnected() polling.
    planner_call_timeout_seconds: float = 180.0
    leaf_call_timeout_seconds: float = 60.0

    # Sandbox
    repl_output_max_chars: int = 4_000

    # Persistence
    trajectory_dir: str = "/mnt/data/training/rlm-trajectories"
    write_trajectory: bool = True


class GraphState(TypedDict, total=False):
    """LangGraph state. `context_obj` deliberately stays *out* of LM messages."""

    config: RLMConfig
    query: str
    context_obj: Any
    context_var: str

    # Conversation with the planner
    messages: list[dict[str, Any]]

    # Sandbox
    repl_globals: dict[str, Any]

    # Loop state
    step: int
    tokens_used: int
    started_at: float

    # Output
    final: str | None
    final_var: str | None
    terminated_reason: str | None
    error: str | None
    pending_final_var: str | None  # set by plan_node, resolved by execute_node

    # Recording
    trajectory: Trajectory
