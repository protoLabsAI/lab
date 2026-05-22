from rlm.parser import parse


def test_extracts_python_fence():
    out = "Sure, let me peek.\n```python\nprint(len(ctx))\n```\nThen I'll proceed."
    p = parse(out)
    assert p.code is not None and "print(len(ctx))" in p.code
    assert not p.is_final


def test_extracts_py_fence():
    p = parse("```py\nx = 1\n```")
    assert p.code is not None and "x = 1" in p.code


def test_final_string():
    p = parse('I checked everything.\nFINAL("The answer is 42.")')
    assert p.is_final and p.final == "The answer is 42."


def test_final_unquoted():
    p = parse("FINAL(42)")
    assert p.is_final and p.final == "42"


def test_final_var():
    p = parse("All done.\nFINAL_VAR(answer)")
    assert p.is_final and p.final_var == "answer" and p.final is None


def test_final_var_quoted():
    p = parse('FINAL_VAR("result")')
    assert p.is_final and p.final_var == "result"


def test_final_string_ignored_when_code_present():
    """Ambiguous; we run the code and let the next turn re-emit FINAL."""
    p = parse('```python\nprint("hi")\n```\nFINAL(done)')
    assert p.code is not None and p.final is None and p.final_var is None


def test_final_var_coexists_with_code():
    """Common pattern: bind a var in code, then FINAL_VAR it. Run code first."""
    out = "```python\nresult = 'answer'\n```\nFINAL_VAR(result)"
    p = parse(out)
    assert p.code is not None and "result = 'answer'" in p.code
    assert p.final_var == "result"
    assert p.final is None


def test_neither():
    p = parse("Just thinking out loud, no action yet.")
    assert p.code is None and not p.is_final


def test_strips_thinking_block():
    """Model emits <think>...</think> inline (vLLM reasoning-parser quirk).
    The planner mentions FINAL inside thinking, then emits the real one after."""
    out = (
        "The count is 62.\nI should emit FINAL(62).\n</think>\n\nFINAL(62)"
    )
    p = parse(out)
    assert p.is_final and p.final == "62"


def test_strips_thinking_block_with_open_tag():
    out = "<think>plan: emit FINAL(wrong)</think>\nFINAL(right)"
    p = parse(out)
    assert p.final == "right"


def test_last_final_wins():
    """If model self-quotes (e.g. 'I will emit FINAL(x)'), take the closing one."""
    p = parse("I will emit FINAL(draft) shortly.\nNow: FINAL(real)")
    assert p.final == "real"


def test_code_after_thinking():
    out = "<think>I'll peek</think>\n```python\nprint(len(ctx))\n```"
    p = parse(out)
    assert p.code is not None and "print(len(ctx))" in p.code


def test_truncated_fence_detected():
    """Planner hit max_tokens mid-block: opening ``` with no closer."""
    out = "Let me write some code:\n```python\nimport re\n# ...thinking interrupted"
    p = parse(out)
    assert p.code is None and p.truncated_fence


def test_truncated_fence_not_a_final():
    """Even with FINAL_VAR present, a truncated fence should NOT terminate."""
    out = "Code:\n```python\nx = 1\nFINAL_VAR(x)"
    p = parse(out)
    assert p.truncated_fence
    # final_var IS picked up textually, but caller treats truncated as redo
    assert p.final_var == "x"


def test_full_fence_then_unclosed_second():
    out = "```python\nx = 1\n```\nNow more:\n```python\ny ="
    p = parse(out)
    assert p.code is not None and "x = 1" in p.code
    assert p.truncated_fence
