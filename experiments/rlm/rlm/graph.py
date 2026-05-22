"""LangGraph DAG for the RLM loop.

   START
     ↓
   plan ──(FINAL? budget? error?)──→ END
     ↓
   execute ──(FINAL_VAR resolved? budget?)──→ END
     ↓
   plan  (loop)

The graph state holds the planner's message history; the sandbox lives outside
graph state (mutable, holds the REPL globals + the context object). That keeps
the LangGraph state small and serializable; the heavy stuff stays in-process.

Code + FINAL_VAR can co-exist in one planner turn (the model often binds the
final variable inside the same fenced block). plan_node sets `pending_final_var`
on the state; execute_node runs the code, then resolves the var and terminates.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import openai
from langgraph.graph import END, START, StateGraph

from rlm.llm import LeafClient, PlannerClient
from rlm.parser import parse
from rlm.prompts import render_exec_result, render_first_user, render_system
from rlm.sandbox import Sandbox
from rlm.schema import GraphState, RLMConfig, Trajectory
from rlm.trajectory import write_trajectory


def _budget_exceeded(state: GraphState) -> str | None:
    """Return a specific reason string if any budget is hit, else None.

    Reasons are distinct so the trajectory record can show *which* knob fired:
    - "max_steps": planner-turn count
    - "max_tokens": cumulative token usage (planner+leaf)
    - "max_wall": wall-clock since session start
    """
    cfg = state["config"]
    if state.get("step", 0) >= cfg.max_steps:
        return "max_steps"
    if state.get("tokens_used", 0) >= cfg.max_tokens:
        return "max_tokens"
    if time.time() - state.get("started_at", time.time()) >= cfg.max_wall_seconds:
        return "max_wall"
    return None


def build_graph(planner: PlannerClient, sandbox: Sandbox):
    def _resolve_final_var(name: str) -> str | None:
        """Look up `name` in sandbox globals; return final text (str verbatim,
        repr for non-strings, None if undefined)."""
        val = sandbox.get(name)
        if val is None:
            return None
        if isinstance(val, str):
            return val
        return repr(val)

    async def plan_node(state: GraphState) -> dict[str, Any]:
        messages = state["messages"]
        traj = state["trajectory"]

        if not messages:
            messages = [
                {"role": "system", "content": render_system(state["context_var"])},
                {
                    "role": "user",
                    "content": render_first_user(
                        state["query"], state["context_var"], traj.context_meta
                    ),
                },
            ]

        try:
            turn = await planner.call(messages)
        except (openai.APITimeoutError, asyncio.TimeoutError) as e:
            return {
                "step": state.get("step", 0) + 1,
                "terminated_reason": "planner_timeout",
                "error": f"planner call timed out after "
                f"{state['config'].planner_call_timeout_seconds}s: {type(e).__name__}",
            }
        except openai.APIError as e:
            return {
                "step": state.get("step", 0) + 1,
                "terminated_reason": "planner_error",
                "error": f"{type(e).__name__}: {e}",
            }
        traj.turns.append(turn)
        messages = messages + [{"role": "assistant", "content": turn.response}]

        parsed = parse(turn.response)

        update: dict[str, Any] = {
            "messages": messages,
            "step": state.get("step", 0) + 1,
            "tokens_used": state.get("tokens_used", 0) + turn.tokens_in + turn.tokens_out,
            # Always carry pending_final_var (None clears any prior pending).
            "pending_final_var": parsed.final_var,
        }

        # If code is present, defer everything to execute_node — it will run the
        # code, then resolve any pending FINAL_VAR.
        if parsed.code is not None:
            _check_and_label_budget(state, update)
            return update

        # Truncated mid-fence: planner ran out of max_tokens. Don't error;
        # send a short prompt asking it to be more concise and try again.
        if parsed.truncated_fence:
            update["pending_final_var"] = None
            update["messages"] = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response was cut off mid-code-block (token "
                        "limit). Re-send a concise version of just the next code "
                        "block — keep prose to a minimum, no thinking aloud. End "
                        "with FINAL_VAR(name) once the answer variable is bound."
                    ),
                }
            ]
            _check_and_label_budget(state, update)
            return update

        # No code: this is a terminal turn. Resolve final now.
        if parsed.final is not None:
            update["final"] = parsed.final
            update["terminated_reason"] = "final"
        elif parsed.final_var is not None:
            final_text = _resolve_final_var(parsed.final_var)
            if final_text is None:
                update["terminated_reason"] = "error"
                update["error"] = (
                    f"FINAL_VAR({parsed.final_var}) but variable was None or undefined"
                )
            else:
                update["final"] = final_text
                update["final_var"] = parsed.final_var
                update["terminated_reason"] = "final"
        else:
            update["terminated_reason"] = "error"
            update["error"] = "planner emitted neither code nor FINAL"
        _check_and_label_budget(state, update)
        return update

    async def execute_node(state: GraphState) -> dict[str, Any]:
        traj = state["trajectory"]
        last_assistant = state["messages"][-1]["content"]
        parsed = parse(last_assistant)
        assert parsed.code is not None  # routed here only when code present

        # sandbox.execute is sync and the RLM/RLM_MAP callables inside use
        # asyncio.run(), which requires no running loop in the current thread.
        # Push it to a worker thread so the outer event loop is undisturbed.
        exec_turn, leaf_turns = await asyncio.to_thread(sandbox.execute, parsed.code)
        traj.turns.append(exec_turn)
        traj.turns.extend(leaf_turns)

        leaf_tokens = sum(t.tokens_in + t.tokens_out for t in leaf_turns)

        result_msg = render_exec_result(
            exec_turn.stdout, exec_turn.stderr, leaf_call_count=len(leaf_turns)
        )
        new_messages = state["messages"] + [{"role": "user", "content": result_msg}]

        update: dict[str, Any] = {
            "messages": new_messages,
            "tokens_used": state.get("tokens_used", 0) + leaf_tokens,
        }

        # Same-turn FINAL_VAR: code was meant to bind the answer var. Resolve now.
        pending_fv = state.get("pending_final_var")
        if pending_fv:
            final_text = _resolve_final_var(pending_fv)
            if final_text is None:
                update["terminated_reason"] = "error"
                update["error"] = (
                    f"FINAL_VAR({pending_fv}) but variable was None or undefined "
                    f"after exec (stderr: {exec_turn.stderr[:200]!r})"
                )
            else:
                update["final"] = final_text
                update["final_var"] = pending_fv
                update["terminated_reason"] = "final"
            update["pending_final_var"] = None

        _check_and_label_budget(state, update)
        return update

    def _check_and_label_budget(state: GraphState, update: dict[str, Any]) -> None:
        """If a budget fired in this turn, write the *specific* reason to update.

        Called at the end of each node BEFORE we look up routing. The router
        sees the updated state and routes to END if terminated_reason is set.
        """
        if update.get("terminated_reason"):
            return
        # Apply node updates onto a snapshot so the budget check sees post-turn state
        merged = {**state, **update}
        reason = _budget_exceeded(merged)
        if reason:
            update["terminated_reason"] = reason

    def route_after_plan(state: GraphState) -> str:
        if state.get("terminated_reason"):
            return "end"
        return "execute"

    def route_after_execute(state: GraphState) -> str:
        if state.get("terminated_reason"):
            return "end"
        return "plan"

    g: StateGraph = StateGraph(GraphState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, {"execute": "execute", "end": END})
    g.add_conditional_edges("execute", route_after_execute, {"plan": "plan", "end": END})
    return g.compile()


class RLM:
    """Top-level handle. `RLM(cfg).completion(query, context)` → final answer."""

    def __init__(self, cfg: RLMConfig | None = None):
        self.cfg = cfg or RLMConfig()
        self._planner = PlannerClient(self.cfg)
        self._leaf = LeafClient(self.cfg)

    async def completion(
        self,
        query: str,
        context: Any,
        context_var: str = "ctx",
        context_meta: dict[str, Any] | None = None,
    ) -> Trajectory:
        sandbox = Sandbox(
            leaf_call=self._leaf.call,
            output_max_chars=self.cfg.repl_output_max_chars,
        )
        sandbox.install_context(context_var, context)

        meta = context_meta or {}
        meta.setdefault("variable", context_var)
        meta.setdefault("type", type(context).__name__)
        try:
            meta.setdefault("len", len(context))
        except TypeError:
            pass

        traj = Trajectory(query=query, context_meta=meta)
        graph = build_graph(self._planner, sandbox)

        initial: GraphState = {
            "config": self.cfg,
            "query": query,
            "context_obj": context,
            "context_var": context_var,
            "messages": [],
            "step": 0,
            "tokens_used": 0,
            "started_at": time.time(),
            "trajectory": traj,
        }

        # LangGraph's recursion_limit is a guard against infinite loops in routing,
        # not our step budget. Set it generously above max_steps.
        final_state = await graph.ainvoke(
            initial, config={"recursion_limit": self.cfg.max_steps * 4 + 10}
        )

        traj.final = final_state.get("final")
        traj.final_var = final_state.get("final_var")
        # Don't lie about the reason — if no node set one, that's an unknown
        # exit (LangGraph recursion_limit, asyncio cancellation, etc.).
        traj.terminated_reason = final_state.get("terminated_reason") or "unknown"
        traj.error = final_state.get("error")
        traj.totals = {
            "tokens": float(final_state.get("tokens_used", 0)),
            "wall_ms": (time.time() - initial["started_at"]) * 1000,
            "steps": float(final_state.get("step", 0)),
            "leaf_calls": float(sum(1 for t in traj.turns if t.role == "leaf_call")),
        }

        if self.cfg.write_trajectory:
            write_trajectory(traj, self.cfg.trajectory_dir)

        return traj
