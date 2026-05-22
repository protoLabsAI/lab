import asyncio

import pytest

from rlm.sandbox import Sandbox


async def _fake_leaf(subquery: str, slice_obj):
    """Pretend leaf model. Returns deterministic text based on the subquery."""
    return (f"answer-for[{subquery}|{slice_obj!r}]", 10, 5)


def test_persistent_globals():
    sb = Sandbox(leaf_call=_fake_leaf)
    sb.install_context("ctx", [1, 2, 3])
    t1, _ = sb.execute("x = sum(ctx)\nprint(x)")
    assert "6" in t1.stdout
    t2, _ = sb.execute("print(x * 2)")
    assert "12" in t2.stdout


def test_stdout_capture_and_truncation():
    sb = Sandbox(leaf_call=_fake_leaf, output_max_chars=20)
    t, _ = sb.execute("print('A' * 100)")
    assert t.truncated
    assert "[truncated" in t.stdout


def test_stderr_on_exception():
    sb = Sandbox(leaf_call=_fake_leaf)
    t, _ = sb.execute("raise ValueError('boom')")
    assert "ValueError" in t.stderr and "boom" in t.stderr


def test_rlm_callable_records_turn():
    sb = Sandbox(leaf_call=_fake_leaf)
    sb.install_context("ctx", "hello world")
    t, leaf_turns = sb.execute("r = RLM('summarize', ctx)\nprint(r)")
    assert "answer-for[summarize" in t.stdout
    assert len(leaf_turns) == 1
    assert leaf_turns[0].subquery == "summarize"


def test_rlm_map_parallel():
    sb = Sandbox(leaf_call=_fake_leaf)
    sb.install_context("ctx", ["a", "b", "c"])
    t, leaf_turns = sb.execute(
        "results = RLM_MAP(['q1', 'q2', 'q3'], ctx)\nfor r in results: print(r)"
    )
    assert len([t for t in leaf_turns]) == 3
    assert "q1" in t.stdout and "q2" in t.stdout and "q3" in t.stdout


def test_rlm_map_length_mismatch():
    sb = Sandbox(leaf_call=_fake_leaf)
    t, _ = sb.execute("RLM_MAP(['a', 'b'], [1, 2, 3])")
    assert "length mismatch" in t.stderr
