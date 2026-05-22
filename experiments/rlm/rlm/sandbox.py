"""Local exec sandbox.

NOT a security boundary — assume the planner LM is trusted (it's our own model).
For user-exposed runs, graduate to subprocess / Docker (M3).

The sandbox keeps a persistent globals dict per session, captures stdout/stderr,
and exposes `RLM(query, slice)` and `RLM_MAP(queries, slices)` as callables that
delegate back to the leaf-model client. `RLM_MAP` runs branches in parallel via
asyncio.gather — that's where the paper's perf win lives.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import time
import traceback
from typing import Any, Callable, Awaitable

from rlm.schema import ExecTurn, LeafCallTurn


def _slice_hash(obj: Any) -> str:
    try:
        s = repr(obj)
    except Exception:
        s = str(type(obj))
    return hashlib.blake2b(s.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _slice_preview(obj: Any, max_chars: int = 200) -> str:
    try:
        s = repr(obj)
    except Exception:
        s = str(type(obj))
    return s[:max_chars]


class Sandbox:
    """One sandbox per RLM session. Globals persist across exec() calls."""

    def __init__(
        self,
        leaf_call: Callable[[str, Any], Awaitable[tuple[str, int, int]]],
        output_max_chars: int = 4_000,
    ):
        """
        leaf_call: async fn(subquery, slice) -> (response_text, tokens_in, tokens_out)
        """
        self._leaf_call = leaf_call
        self._output_max = output_max_chars
        self._globals: dict[str, Any] = {"__name__": "__rlm_session__"}
        self.leaf_turns: list[LeafCallTurn] = []

    def install_context(self, name: str, obj: Any) -> None:
        self._globals[name] = obj

    def get(self, name: str) -> Any:
        return self._globals.get(name)

    @property
    def globals(self) -> dict[str, Any]:
        return self._globals

    def _make_rlm_callables(self) -> dict[str, Callable[..., Any]]:
        leaf_turns = self.leaf_turns
        leaf_call = self._leaf_call

        def _RLM(subquery: str, slice_obj: Any = None) -> str:
            t0 = time.perf_counter()
            response, tin, tout = asyncio.run(leaf_call(subquery, slice_obj))
            leaf_turns.append(
                LeafCallTurn(
                    subquery=subquery,
                    slice_hash=_slice_hash(slice_obj) if slice_obj is not None else None,
                    slice_preview=_slice_preview(slice_obj) if slice_obj is not None else None,
                    model="leaf",
                    response=response,
                    tokens_in=tin,
                    tokens_out=tout,
                    wall_ms=(time.perf_counter() - t0) * 1000,
                )
            )
            return response

        def _RLM_MAP(queries: list[str], slices: list[Any] | None = None) -> list[str]:
            slices = slices if slices is not None else [None] * len(queries)
            if len(queries) != len(slices):
                raise ValueError(
                    f"RLM_MAP: queries ({len(queries)}) and slices ({len(slices)}) length mismatch"
                )

            async def _run_all() -> list[tuple[str, int, int, float]]:
                async def _one(q: str, s: Any) -> tuple[str, int, int, float]:
                    t0 = time.perf_counter()
                    resp, tin, tout = await leaf_call(q, s)
                    return resp, tin, tout, (time.perf_counter() - t0) * 1000

                return await asyncio.gather(*(_one(q, s) for q, s in zip(queries, slices)))

            results = asyncio.run(_run_all())
            for q, s, (resp, tin, tout, wall_ms) in zip(queries, slices, results):
                leaf_turns.append(
                    LeafCallTurn(
                        subquery=q,
                        slice_hash=_slice_hash(s) if s is not None else None,
                        slice_preview=_slice_preview(s) if s is not None else None,
                        model="leaf",
                        response=resp,
                        tokens_in=tin,
                        tokens_out=tout,
                        wall_ms=wall_ms,
                    )
                )
            return [r[0] for r in results]

        return {"RLM": _RLM, "RLM_MAP": _RLM_MAP}

    def execute(self, code: str) -> tuple[ExecTurn, list[LeafCallTurn]]:
        """Run code; return ExecTurn + any leaf calls made during this exec."""
        leaf_turns_before = len(self.leaf_turns)
        stdout, stderr = io.StringIO(), io.StringIO()
        self._globals.update(self._make_rlm_callables())
        t0 = time.perf_counter()

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(code, "<rlm-cell>", "exec"), self._globals)
            except Exception:
                traceback.print_exc(file=stderr)

        wall_ms = (time.perf_counter() - t0) * 1000
        out = stdout.getvalue()
        err = stderr.getvalue()
        truncated = False
        if len(out) > self._output_max:
            out = out[: self._output_max] + f"\n... [truncated {len(out) - self._output_max} chars]"
            truncated = True
        if len(err) > self._output_max:
            err = err[: self._output_max] + f"\n... [truncated {len(err) - self._output_max} chars]"
            truncated = True

        new_leaf_turns = self.leaf_turns[leaf_turns_before:]
        return (
            ExecTurn(code=code, stdout=out, stderr=err, truncated=truncated, wall_ms=wall_ms),
            new_leaf_turns,
        )
