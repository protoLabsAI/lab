"""Async OpenAI-compat clients for planner + leaf models.

Both clients hit OpenAI-compat endpoints (vLLM direct in M0, gateway in M1+).
We track per-call token counts so the orchestrator can enforce token budgets
without trusting the model's self-report.
"""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI

from rlm.parser import _strip_thinking
from rlm.schema import PlannerTurn, RLMConfig


class PlannerClient:
    def __init__(self, cfg: RLMConfig):
        self._cfg = cfg
        self._client = AsyncOpenAI(base_url=cfg.planner_base_url, api_key=cfg.planner_api_key)

    async def call(self, messages: list[dict[str, Any]]) -> PlannerTurn:
        t0 = time.perf_counter()
        resp = await self._client.with_options(
            timeout=self._cfg.planner_call_timeout_seconds
        ).chat.completions.create(
            model=self._cfg.planner_model,
            messages=messages,
            temperature=0.0,  # deterministic — RLM tasks reward reproducibility
            max_tokens=16384,  # Qwen3.6 thinking model card suggests 32K but our
            # measured per-turn wall at 32K can hit 20+ min; 16K is the floor
            # below which we hit truncated-fence errors from M0
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0].message
        content = choice.content or ""
        # Gateway exposes the trace as `reasoning` (OpenRouter convention) when
        # the normalizer fires; vLLM-direct uses `reasoning_content`.
        reasoning = getattr(choice, "reasoning", None) or getattr(
            choice, "reasoning_content", None
        )
        # Defense-in-depth for the vLLM reasoning-parser bug (see CLAUDE.md):
        # if content is empty but reasoning has the answer, fall through.
        if not content and reasoning:
            content = reasoning.rsplit("</think>", 1)[-1].strip() or reasoning

        usage = resp.usage
        return PlannerTurn(
            request_messages=messages,
            response=content,
            reasoning=reasoning,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            wall_ms=wall_ms,
        )


class LeafClient:
    """Wraps the leaf model. Returns (text, tokens_in, tokens_out)."""

    def __init__(self, cfg: RLMConfig):
        self._cfg = cfg
        self._client = AsyncOpenAI(base_url=cfg.leaf_base_url, api_key=cfg.leaf_api_key)

    async def call(self, subquery: str, slice_obj: Any) -> tuple[str, int, int]:
        if slice_obj is None:
            user_content = subquery
        else:
            try:
                slice_text = repr(slice_obj)
            except Exception:
                slice_text = str(type(slice_obj))
            user_content = (
                f"You are answering a sub-query over a piece of context.\n\n"
                f"=== CONTEXT ===\n{slice_text}\n=== END CONTEXT ===\n\n"
                f"Sub-query: {subquery}\n\n"
                f"Answer concisely. If the answer is not in the context, say so."
            )

        resp = await self._client.with_options(
            timeout=self._cfg.leaf_call_timeout_seconds
        ).chat.completions.create(
            model=self._cfg.leaf_model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.1,
            max_tokens=32768,  # Qwen3.6 spec — leaf may need to emit structured
            # per-chunk output (JSON, hunks, classifications) plus its think block
        )
        choice = resp.choices[0].message
        raw = (
            choice.content
            or getattr(choice, "reasoning", None)
            or getattr(choice, "reasoning_content", "")
            or ""
        )
        # Heretic (and 27B-thinking) often emit <think>...</think> in content
        # even on simple prompts — strip before returning to the planner.
        content = _strip_thinking(raw)
        usage = resp.usage
        return (
            content,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
