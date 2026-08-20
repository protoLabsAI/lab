"""protoLabs.nodes — prompt-engineering nodes.

ProtoLTXPromptEnhancer  raw idea -> on-distribution LTX-2.3 prompt, using the ACTUAL
                        Gemma enhancer system prompts shipped in the LTX-2 checkout
                        (the caption distribution the model was trained toward).
                        Runs on a gateway lane instead of loading the 12B Gemma
                        encoder — GPU1 stays free for the render itself.
ProtoTextTemplate       {a}/{b}/{c}/{d} substitution for wiring strings together
ProtoShowText           display a STRING in-graph and pass it through
"""
from __future__ import annotations

import os
import re

from .proto_client import chat, list_chat_models, salvage_text, vision_content

_LTX2_PROMPT_DIR = os.path.expanduser(
    "~/dev/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts"
)
_PROMPT_FILES = {
    "t2v": os.path.join(_LTX2_PROMPT_DIR, "gemma_t2v_system_prompt.txt"),
    "i2v": os.path.join(_LTX2_PROMPT_DIR, "gemma_i2v_system_prompt.txt"),
}

# Distilled from infra/video-bridge/PROMPTING.md — used only when the LTX-2 checkout
# (and its canonical Gemma enhancer prompts) isn't on this machine.
_FALLBACK_SYSTEM = """You are a video-prompt writer for the LTX-2.3 generative video model. \
Rewrite the user's raw idea into ONE flowing paragraph of at most 200 words, chronological, \
present-progressive ("is walking", "speaking"), joined with temporal connectors ("as", \
"then", "while"). Start directly with the action — never "The scene opens with", no \
timestamps, no scene cuts, no markdown. Order: main action in one sentence, then specific \
movements and gestures, precise character/object appearance, environment, camera, lighting \
and color, sudden changes. Use restrained concrete language ("red dress", "soft overhead \
light") — no quality tags or dramatic adjectives. Always specify camera behavior and its \
end-state, but never invent camera motion beyond what the idea implies. Weave the complete \
soundscape chronologically alongside the actions, specific not vague; dialogue as exact \
words in quotes with voice character. Describe only what is seen and heard. Output only the \
rewritten prompt, nothing else.\
{i2v_extra}"""
_FALLBACK_I2V_EXTRA = (
    " An image of the first frame is provided: describe only the CHANGES from it — do not "
    "re-describe what the image already establishes; inaccurate re-description causes scene cuts."
)


def _system_prompt(mode: str) -> str:
    path = _PROMPT_FILES[mode]
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return _FALLBACK_SYSTEM.format(
            i2v_extra=_FALLBACK_I2V_EXTRA if mode == "i2v" else ""
        )


def _clean_paragraph(text: str) -> str:
    """Enhancer output should be one bare paragraph — strip wrappers models add anyway."""
    t = text.strip()
    t = re.sub(r"^```[a-z]*\n?|\n?```$", "", t).strip()
    t = re.sub(r"^(Output|Prompt)\s*:\s*", "", t, flags=re.IGNORECASE)
    if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
        t = t[1:-1]
    return " ".join(t.split())


class ProtoLTXPromptEnhancer:
    """Raw idea -> LTX-2.3 prompt. i2v mode REQUIRES the first frame wired in — the
    canonical enhancer describes changes relative to the image, which it must see."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["t2v", "i2v"],),
                "idea": ("STRING", {"multiline": True, "default": "",
                         "tooltip": "Raw input prompt. Camera moves and speech you want "
                                    "MUST be stated here — the enhancer never invents them."}),
                "model": (list_chat_models(),),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "First frame — required for i2v mode."}),
                "gateway": ("PROTO_GATEWAY",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = "protoLab/Prompt"

    def run(self, mode, idea, model, temperature, seed, image=None, gateway=None):
        if mode == "i2v" and image is None:
            raise ValueError(
                "i2v enhancement needs the first frame (IMAGE input) — it describes "
                "changes FROM the image. Use t2v mode for text-only."
            )
        messages = [
            {"role": "system", "content": _system_prompt(mode)},
            {"role": "user",
             "content": vision_content(f"Raw Input Prompt: {idea}",
                                       image if mode == "i2v" else None)},
        ]
        msg = chat(gateway, model, messages, temperature=temperature,
                   max_tokens=2048, seed=seed, thinking="off")
        text, _ = salvage_text(msg)
        return (_clean_paragraph(text),)


class _Fmt(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class ProtoTextTemplate:
    """format() with {a} {b} {c} {d} placeholders; unknown placeholders pass through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"template": ("STRING", {"multiline": True, "default": "{a}"})},
            "optional": {
                "a": ("STRING", {"forceInput": True}),
                "b": ("STRING", {"forceInput": True}),
                "c": ("STRING", {"forceInput": True}),
                "d": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "protoLab/Prompt"

    def run(self, template, a="", b="", c="", d=""):
        return (template.format_map(_Fmt(a=a, b=b, c=c, d=d)),)


class ProtoShowText:
    """Display a STRING in the graph and pass it through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "protoLab/Prompt"
    OUTPUT_NODE = True

    def run(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "ProtoLTXPromptEnhancer": ProtoLTXPromptEnhancer,
    "ProtoTextTemplate": ProtoTextTemplate,
    "ProtoShowText": ProtoShowText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProtoLTXPromptEnhancer": "LTX-2.3 Prompt Enhancer (protoLab)",
    "ProtoTextTemplate": "Text Template (protoLab)",
    "ProtoShowText": "Show Text (protoLab)",
}
