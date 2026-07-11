#!/usr/bin/env python3
"""Boundary probe: is the line 'derivable from visible state', or 'derivable AND model-computable'?

Frames each test IN AgentWorld's own terminal/python domain (its strongest trained turf) but demands
output that requires real computation an LLM cannot do in its head (hashing, big arithmetic, sorting).
If it confabulates here, the trustworthy boundary cuts *across* domains: a simulated sandbox can't be
trusted for any command whose output requires real execution — which is most of what coding tasks
verify. Thinking left ON (the model's designed long-CoT simulation mode) to give it its best shot.
Every answer is checked against locally-computed ground truth.
"""
from __future__ import annotations
import hashlib, json
from openai import OpenAI

AW = OpenAI(base_url="http://localhost:8010/v1", api_key="local")
MODEL = "Qwen/Qwen-AgentWorld-35B-A3B"
TERM_SYS = ("You are a Terminal World Model simulating a Linux bash shell. Given the command, predict "
            "the EXACT stdout the command produces. Output only the stdout, nothing else.")

# (label, command, ground_truth) — ground truth computed locally below.
TESTS = [
    ("sha256('hello')", "python3 -c \"import hashlib;print(hashlib.sha256(b'hello').hexdigest())\"",
     hashlib.sha256(b"hello").hexdigest()),
    ("md5('agentworld')", "python3 -c \"import hashlib;print(hashlib.md5(b'agentworld').hexdigest())\"",
     hashlib.md5(b"agentworld").hexdigest()),
    ("47293*81947", "python3 -c \"print(47293*81947)\"", str(47293 * 81947)),
    ("2**100", "python3 -c \"print(2**100)\"", str(2 ** 100)),
    ("sorted scramble", "python3 -c \"print(sorted([42,7,19,88,3,56,21,64,5,90]))\"",
     str(sorted([42, 7, 19, 88, 3, 56, 21, 64, 5, 90]))),
    ("len+rev", "python3 -c \"s='supercalifragilistic';print(len(s), s[::-1])\"",
     f"{len('supercalifragilistic')} {'supercalifragilistic'[::-1]}"),
]


def ask(cmd: str, think: bool) -> str:
    r = AW.chat.completions.create(
        model=MODEL, temperature=0.2, max_tokens=2000,
        messages=[{"role": "system", "content": TERM_SYS},
                  {"role": "user", "content": f"Command: {cmd}"}],
        extra_body={"chat_template_kwargs": {"enable_thinking": think}})
    m = r.choices[0].message
    return (m.content or getattr(m, "reasoning_content", None) or "").strip()


def main():
    think = "--think" in __import__("sys").argv
    print(f"mode: thinking={'ON' if think else 'OFF (simulator mode)'}")
    print(f"{'test':<20} {'verdict':<7} truth -> raw simulated output")
    print("-" * 90)
    n_ok = 0
    for label, cmd, truth in TESTS:
        raw = ask(cmd, think)
        ok = truth in raw
        n_ok += ok
        shown = raw.replace("\n", "\\n")[:60] or "(empty)"
        print(f"{label:<20} {'OK' if ok else 'WRONG':<7} {truth[:30]} -> {shown}")
    print("-" * 90)
    print(f"correct: {n_ok}/{len(TESTS)}")


if __name__ == "__main__":
    main()
