"""protoLabs.nodes — LLM nodes over the LiteLLM gateway.

ProtoGateway      explicit gateway config (URL/key/timeout) -> PROTO_GATEWAY
ProtoLLM          chat completion vs a gateway lane, vision-capable (fast = Ornith has vision)
ProtoStructured   schema-constrained JSON output (guided decode via response_format)
ProtoAgentChat    same call shape vs ANY OpenAI-compatible endpoint (protoAgent /v1, a raw
                  vLLM lane, anything) — the ecosystem escape hatch
"""
from __future__ import annotations

import json

from .proto_client import (
    DEFAULT_GATEWAY_URL,
    DEFAULT_TIMEOUT,
    chat,
    list_chat_models,
    pretty_json,
    salvage_text,
    vision_content,
)

_THINKING_TOOLTIP = (
    "auto = model/template default. on/off sends chat_template_kwargs.enable_thinking "
    "(vLLM lanes honor it; retried without on a 400). reasoning lane thinks by default."
)


class ProtoGateway:
    """Explicit gateway target. Without this node every Proto* node resolves from env:
    PROTOLAB_GATEWAY_URL / PROTOLAB_GATEWAY_KEY / GATEWAY_API_KEY / evals/.env."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_url": ("STRING", {"default": DEFAULT_GATEWAY_URL}),
            "api_key": ("STRING", {"default": "", "tooltip":
                        "Empty = resolve from env (PROTOLAB_GATEWAY_KEY / GATEWAY_API_KEY "
                        "/ ~/dev/lab/evals/.env). Anything typed here is saved INTO THE "
                        "WORKFLOW JSON — prefer env on shared graphs."}),
            "timeout_s": ("INT", {"default": DEFAULT_TIMEOUT, "min": 5, "max": 3600}),
        }}

    RETURN_TYPES = ("PROTO_GATEWAY",)
    RETURN_NAMES = ("gateway",)
    FUNCTION = "run"
    CATEGORY = "protoLab/LLM"

    def run(self, base_url, api_key, timeout_s):
        from .proto_client import resolve_gateway
        gw = {"base_url": base_url.rstrip("/"),
              "api_key": api_key or resolve_gateway()["api_key"],
              "timeout": timeout_s}
        return (gw,)


class ProtoLLM:
    """Chat completion against a gateway lane. Wire an IMAGE in for vision
    (protolabs/fast keeps bf16 vision on the quant)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list_chat_models(),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768,
                               "tooltip": "Don't token-starve thinking models — the answer "
                                          "arrives after the think closes."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                         "tooltip": "0 = not sent. >0 pins vLLM sampling AND busts ComfyUI's "
                                    "node cache per value; use control_after_generate to re-roll."}),
                "thinking": (["auto", "on", "off"], {"default": "auto", "tooltip": _THINKING_TOOLTIP}),
            },
            "optional": {
                "image": ("IMAGE",),
                "gateway": ("PROTO_GATEWAY",),
                "model_override": ("STRING", {"default": "", "tooltip":
                                   "Non-empty wins over the combo — for lanes newer than "
                                   "the cached model list."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning")
    FUNCTION = "run"
    CATEGORY = "protoLab/LLM"

    def run(self, model, prompt, system, temperature, max_tokens, seed, thinking,
            image=None, gateway=None, model_override=""):
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": vision_content(prompt, image)})
        msg = chat(gateway, model_override.strip() or model, messages,
                   temperature=temperature, max_tokens=max_tokens, seed=seed,
                   thinking=thinking)
        text, reasoning = salvage_text(msg)
        return (text, reasoning)


class ProtoStructured:
    """Schema-constrained JSON via response_format json_schema (vLLM guided decode).
    Empty schema falls back to free json_object mode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list_chat_models(),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "json_schema": ("STRING", {"multiline": True, "default": "", "tooltip":
                                "A JSON Schema object. Empty = json_object mode (any valid "
                                "JSON, no shape guarantee)."}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
                "gateway": ("PROTO_GATEWAY",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "run"
    CATEGORY = "protoLab/LLM"

    def run(self, model, prompt, json_schema, temperature, max_tokens, seed,
            system="", gateway=None):
        if json_schema.strip():
            schema = json.loads(json_schema)  # fail loud on a broken schema
            response_format = {"type": "json_schema",
                               "json_schema": {"name": "output", "schema": schema, "strict": True}}
        else:
            response_format = {"type": "json_object"}
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        msg = chat(gateway, model, messages, temperature=temperature,
                   max_tokens=max_tokens, seed=seed, response_format=response_format,
                   thinking="off")  # guided decode + thinking don't mix
        text, _ = salvage_text(msg)
        return (pretty_json(json.loads(text)),)


class ProtoAgentChat:
    """One-shot chat vs any OpenAI-compatible endpoint that is NOT the gateway —
    protoAgent's /v1, a direct vLLM lane (:8040/:8041/:8032), a remote agent.
    Set PROTOLAB_AGENT_URL to give base_url a default."""

    @classmethod
    def INPUT_TYPES(cls):
        import os
        return {
            "required": {
                "base_url": ("STRING", {"default": os.environ.get("PROTOLAB_AGENT_URL", ""),
                             "tooltip": "OpenAI-compatible /v1 root, e.g. http://ava:8010/v1 "
                                        "or http://localhost:8040/v1"}),
                "model": ("STRING", {"default": "default"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "system": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning")
    FUNCTION = "run"
    CATEGORY = "protoLab/LLM"

    def run(self, base_url, model, prompt, max_tokens, temperature, seed,
            system="", api_key="", image=None):
        if not base_url.strip():
            raise ValueError("ProtoAgentChat needs a base_url (or set PROTOLAB_AGENT_URL)")
        messages = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": vision_content(prompt, image)})
        msg = chat(None, model, messages, temperature=temperature,
                   max_tokens=max_tokens, seed=seed,
                   base_url=base_url, api_key=api_key or None)
        text, reasoning = salvage_text(msg)
        return (text, reasoning)


NODE_CLASS_MAPPINGS = {
    "ProtoGateway": ProtoGateway,
    "ProtoLLM": ProtoLLM,
    "ProtoStructured": ProtoStructured,
    "ProtoAgentChat": ProtoAgentChat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProtoGateway": "Gateway (protoLab)",
    "ProtoLLM": "LLM Chat (protoLab)",
    "ProtoStructured": "LLM Structured JSON (protoLab)",
    "ProtoAgentChat": "Agent Chat — any OpenAI endpoint (protoLab)",
}
