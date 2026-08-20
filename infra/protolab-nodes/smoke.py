"""protoLabs.nodes smoke — headless, no ComfyUI server needed.

Run with ComfyUI's venv (it has torch/torchaudio/requests):
  ~/dev/ComfyUI/venv/bin/python ~/dev/lab/infra/protolab-nodes/smoke.py

Offline checks always run; live checks hit the real gateway and report per-node
PASS/FAIL without stopping (TTS is expected to fail while protovoice-stack is parked).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "protolab_nodes", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
pkg = importlib.util.module_from_spec(spec)
sys.modules["protolab_nodes"] = pkg
spec.loader.exec_module(pkg)

from protolab_nodes.proto_client import salvage_text  # noqa: E402
from protolab_nodes.nodes_llm import ProtoLLM, ProtoStructured  # noqa: E402
from protolab_nodes.nodes_prompt import (  # noqa: E402
    ProtoLTXPromptEnhancer,
    ProtoTextTemplate,
)
from protolab_nodes.nodes_audio import ProtoSTT, ProtoTTS  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, str(detail)[:120]))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            if "-v" in sys.argv:
                traceback.print_exc()
    return deco


# ---------------------------------------------------------------- offline

@check("mappings")
def _():
    n = len(pkg.NODE_CLASS_MAPPINGS)
    assert n == 9 and set(pkg.NODE_DISPLAY_NAME_MAPPINGS) == set(pkg.NODE_CLASS_MAPPINGS)
    return f"{n} nodes"


@check("salvage: inline think in content")
def _():
    t, r = salvage_text({"content": "<think>hmm</think>answer"})
    assert (t, r) == ("answer", "hmm"), (t, r)


@check("salvage: unterminated think -> reasoning_content only (vllm#40528)")
def _():
    t, r = salvage_text({"content": "", "reasoning_content": "thoughts</think>the answer"})
    assert t == "the answer" and r == "thoughts", (t, r)
    t2, _ = salvage_text({"content": None, "reasoning_content": "bare answer"})
    assert t2 == "bare answer"


@check("template")
def _():
    out, = ProtoTextTemplate().run("A={a} B={b} keep={x}", a="1", b="2")
    assert out == "A=1 B=2 keep={x}", out


@check("i2v without image raises")
def _():
    try:
        ProtoLTXPromptEnhancer().run("i2v", "x", "protolabs/fast", 0.7, 0)
        raise AssertionError("should have raised")
    except ValueError:
        pass


# ---------------------------------------------------------------- live

@check("live: ProtoLLM pong (protolabs/fast)")
def _():
    text, _ = ProtoLLM().run("protolabs/fast", "Reply with exactly: pong", "", 0.0,
                             512, 0, "auto")
    assert "pong" in text.lower(), repr(text)
    return repr(text)


@check("live: ProtoLLM vision — red square")
def _():
    import torch
    img = torch.zeros(1, 64, 64, 3)
    img[..., 0] = 1.0
    text, _ = ProtoLLM().run("protolabs/fast",
                             "One lowercase word: the dominant color of this image.",
                             "", 0.0, 512, 0, "auto", image=img)
    assert "red" in text.lower(), repr(text)
    return repr(text)


@check("live: ProtoStructured schema")
def _():
    schema = ('{"type":"object","properties":{"name":{"type":"string"},'
              '"year":{"type":"integer"}},"required":["name","year"],'
              '"additionalProperties":false}')
    import json
    out, = ProtoStructured().run("protolabs/fast",
                                 "The first Godzilla film: name and release year.",
                                 schema, 0.0, 512, 0)
    d = json.loads(out)
    assert isinstance(d.get("year"), int), out
    return out.replace("\n", " ")


@check("live: LTX enhancer t2v")
def _():
    p, = ProtoLTXPromptEnhancer().run(
        "t2v", "a rusty robot watering sunflowers at dawn, slow dolly in",
        "protolabs/fast", 0.7, 7)
    words = len(p.split())
    assert 40 <= words <= 260 and "\n" not in p, f"{words} words"
    assert "scene opens" not in p.lower()
    return f"{words} words: {p[:90]}…"


@check("live: ProtoSTT whisper endpoint")
def _():
    import math
    import torch
    sr = 16000
    t = torch.arange(sr).float() / sr
    tone = (0.1 * torch.sin(2 * math.pi * 440 * t)).unsqueeze(0).unsqueeze(0)
    text, = ProtoSTT().run({"waveform": tone, "sample_rate": sr}, "whisper-1")
    return f"transcript={text!r}"  # a tone transcribes to ~nothing; 200 is the test


@check("live: ProtoTTS fish-s2-pro (expected FAIL while protovoice-stack parked)")
def _():
    audio, = ProtoTTS().run("protoLabs nodes online.", "fish-s2-pro", "default", "wav")
    wf = audio["waveform"]
    assert wf.numel() > 1000
    return f"{wf.shape} @ {audio['sample_rate']}Hz"


ok = sum(1 for _, p, _ in RESULTS if p)
print(f"\n=== protoLabs.nodes smoke: {ok}/{len(RESULTS)} ===")
for name, passed, detail in RESULTS:
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
sys.exit(0 if all(p for _, p, _ in RESULTS[:5]) else 1)  # offline checks are gating
