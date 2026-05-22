"""Tiny smoke for the loader. Skipped if dataset isn't cloned."""

from pathlib import Path

import pytest

from eval.locodiff import bucket, load_task, score

DATASET = Path("/tmp/LoCoDiff-bench/locodiff-250425/prompts")
SAMPLE = DATASET / "aider_aider_args.py_prompt.txt"


@pytest.mark.skipif(not SAMPLE.exists(), reason="LoCoDiff dataset not cloned")
def test_load_task_real():
    t = load_task(SAMPLE)
    assert t.name == "aider_aider_args.py"
    assert t.target_path == "aider/args.py"
    assert t.git_log.startswith("commit ")
    assert t.expected.startswith("#!/usr/bin/env python")
    assert t.prompt_bytes > 100_000


def test_score_exact():
    assert score("hello\n", "hello\n")
    assert not score("hello", "hello\n")
    assert not score(None, "x")


def test_score_strips_fence():
    expected = "print('hi')\n"
    fenced = "```python\nprint('hi')\n```"
    assert score(fenced, expected)


def test_score_strips_unmarked_fence():
    expected = "print('hi')\n"
    fenced = "```\nprint('hi')\n```"
    assert score(fenced, expected)


def test_bucket():
    assert bucket(50_000) == "Q1"
    assert bucket(100_000) == "Q2"
    assert bucket(180_000) == "Q3"
    assert bucket(300_000) == "Q4"
