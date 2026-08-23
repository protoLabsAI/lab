"""protoLabs.nodes — JSON plumbing for LLM-directed graphs.

ProtoJSONGet    pull one value out of a JSON string by path (`shots[0].prompt`), as
                STRING / INT / FLOAT at once, so an LLM plan can drive widgets.
ProtoJSONCount  length of a list (or key count of an object) at a path — for graphs
                that branch on how many shots the model actually returned.

Core's JsonExtractString only reads a flat top-level key, which can't address the
array-of-objects shape a shot list has. These also tolerate the ```json fences and
prose padding a non-guided lane sometimes wraps around its answer.
"""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# `a.b`, `a[0]`, `a.0` all address the same thing — split on dots and brackets alike.
_TOKEN = re.compile(r"[^.\[\]]+")


def _loads(raw: str):
    """Parse JSON that may be fenced or padded with prose."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        raise ValueError("empty JSON string")
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: the outermost {...} or [...] span in the text.
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start == -1 or end <= start:
        raise ValueError(f"not JSON: {text[:120]}")
    return json.loads(text[start : end + 1])


def _walk(data, path: str):
    """Resolve a dotted/bracketed path. Empty path returns the whole document."""
    cur = data
    for tok in _TOKEN.findall(path or ""):
        tok = tok.strip().strip("'\"")
        if isinstance(cur, list):
            try:
                idx = int(tok)
            except ValueError:
                raise KeyError(f"list index expected at '{tok}' in path '{path}'")
            if not -len(cur) <= idx < len(cur):
                raise IndexError(f"index {idx} out of range ({len(cur)} items) in path '{path}'")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if tok not in cur:
                raise KeyError(f"no key '{tok}' in path '{path}' (have: {list(cur)[:8]})")
            cur = cur[tok]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at '{tok}' in path '{path}'")
    return cur


def _as_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _as_number(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return float(m.group(0))
    return 0.0


class ProtoJSONGet:
    """One value out of a JSON string, addressed by path, emitted as text/int/float.

    Paths are dotted and/or bracketed: `shots[1].prompt`, `shots.1.prompt`, `seconds`.
    An empty path returns the whole document. Negative list indices count from the end.
    On a miss the node returns `default` rather than failing the run, so a short shot
    list degrades to a usable graph instead of a red node — `ok` reports which happened.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "path": (
                    "STRING",
                    {"default": "shots[0].prompt", "tooltip": "Dotted/bracketed path. Empty = whole document."},
                ),
                "default": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Returned when the path is missing."},
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("text", "int", "float", "ok")
    FUNCTION = "run"
    CATEGORY = "protoLab/JSON"

    def run(self, json_string, path, default=""):
        try:
            value = _walk(_loads(json_string), path)
        except Exception as exc:  # missing path / bad JSON — fall back, don't kill the graph
            print(f"[ProtoJSONGet] {type(exc).__name__}: {exc} — using default")
            text = default
            return (text, int(_as_number(text)), _as_number(text), False)
        text = _as_text(value)
        num = _as_number(value)
        return (text, int(round(num)), num, True)


class ProtoJSONCount:
    """Number of items in the list (or keys in the object) at `path`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "path": ("STRING", {"default": "shots", "tooltip": "Path to a list or object. Empty = whole document."}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "run"
    CATEGORY = "protoLab/JSON"

    def run(self, json_string, path):
        try:
            value = _walk(_loads(json_string), path)
        except Exception as exc:
            print(f"[ProtoJSONCount] {type(exc).__name__}: {exc} — 0")
            return (0,)
        if isinstance(value, (list, dict, str)):
            return (len(value),)
        return (0,)


NODE_CLASS_MAPPINGS = {
    "ProtoJSONGet": ProtoJSONGet,
    "ProtoJSONCount": ProtoJSONCount,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProtoJSONGet": "JSON Get (protoLab)",
    "ProtoJSONCount": "JSON Count (protoLab)",
}
