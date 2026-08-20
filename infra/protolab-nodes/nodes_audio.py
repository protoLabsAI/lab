"""protoLabs.nodes — audio nodes over the gateway.

ProtoTTS  text -> AUDIO via /v1/audio/speech (fish-s2-pro = Fish S2-Pro on this box's
          protovoice-stack, :8092). NOTE: the gateway route 500s while protovoice-stack
          is stopped — it is lazy-load and often parked to keep GPU1 free for video work.
ProtoSTT  AUDIO -> text via /v1/audio/transcriptions (whisper-1).
"""
from __future__ import annotations

import requests

from .proto_client import (
    DEFAULT_TIMEOUT,
    audio_to_wav_bytes,
    bytes_to_audio,
    resolve_gateway,
)


def _auth(gw: dict) -> dict:
    return {"Authorization": f"Bearer {gw['api_key']}"} if gw.get("api_key") else {}


class ProtoTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "model": ("STRING", {"default": "fish-s2-pro"}),
                "voice": ("STRING", {"default": "default"}),
                "response_format": (["wav", "mp3"],),
            },
            "optional": {"gateway": ("PROTO_GATEWAY",)},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "protoLab/Audio"

    def run(self, text, model, voice, response_format, gateway=None):
        gw = resolve_gateway(gateway)
        r = requests.post(
            f"{gw['base_url']}/audio/speech",
            headers={**_auth(gw), "Content-Type": "application/json"},
            json={"model": model, "input": text, "voice": voice,
                  "response_format": response_format},
            timeout=gw.get("timeout", DEFAULT_TIMEOUT),
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"TTS {r.status_code}: {r.text[:300]} — if this is a 500 on fish-s2-pro, "
                "protovoice-stack on protolabs is probably stopped "
                "(systemctl status protovoice-stack)"
            )
        return (bytes_to_audio(r.content, response_format),)


class ProtoSTT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": ("STRING", {"default": "whisper-1"}),
            },
            "optional": {
                "language": ("STRING", {"default": "", "tooltip": "ISO code hint, empty = auto"}),
                "gateway": ("PROTO_GATEWAY",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "protoLab/Audio"

    def run(self, audio, model, language="", gateway=None):
        gw = resolve_gateway(gateway)
        data = {"model": model}
        if language.strip():
            data["language"] = language.strip()
        r = requests.post(
            f"{gw['base_url']}/audio/transcriptions",
            headers=_auth(gw),
            data=data,
            files={"file": ("audio.wav", audio_to_wav_bytes(audio), "audio/wav")},
            timeout=gw.get("timeout", DEFAULT_TIMEOUT),
        )
        if r.status_code != 200:
            raise RuntimeError(f"STT {r.status_code}: {r.text[:300]}")
        return (r.json().get("text", "").strip(),)


NODE_CLASS_MAPPINGS = {
    "ProtoTTS": ProtoTTS,
    "ProtoSTT": ProtoSTT,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProtoTTS": "TTS — Fish S2-Pro (protoLab)",
    "ProtoSTT": "STT — Whisper (protoLab)",
}
