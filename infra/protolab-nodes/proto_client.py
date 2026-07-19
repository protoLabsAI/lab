"""protoLabs.nodes shared plumbing — gateway resolution, HTTP, tensor<->wire converters.

Every node in this pack is a thin client over OpenAI-compatible HTTP. All the shared
behavior lives here so the node files stay declarative:

  * Gateway resolution: PROTO_GATEWAY input > PROTOLAB_GATEWAY_URL/_KEY env >
    GATEWAY_API_KEY env > ~/dev/lab/evals/.env (the same key evals use). No secrets
    on disk in this repo.
  * Model discovery: /v1/models fetched lazily with a short timeout + 5 min cache,
    filtered to chat lanes (the gateway also exposes image/TTS/STT/embed lanes that
    make no sense in an LLM model picker). Static fallback so ComfyUI still boots
    with the gateway down.
  * Think-salvage: vLLM's greedy qwen3 reasoning parser can put the ENTIRE answer in
    reasoning_content when the model never closes </think> (vllm#40528) — downstream
    of the gateway that means content="" with the real answer hidden. salvage_text()
    recovers it the same way claw-eval's Message.text accessor does.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time

import requests

logger = logging.getLogger("protolab-nodes")

DEFAULT_GATEWAY_URL = os.environ.get("PROTOLAB_GATEWAY_URL", "http://ava:4000/v1")
DEFAULT_TIMEOUT = 300
_EVALS_ENV = os.path.expanduser("~/dev/lab/evals/.env")

# Gateway lanes that are not chat models — keep them out of the model picker.
_NON_CHAT_MARKERS = ("image", "krea2", "fish", "whisper", "embedding", "tts", "stt")

FALLBACK_CHAT_MODELS = [
    "protolabs/fast",
    "protolabs/reasoning",
    "protolabs/coder",
    "protolabs/smart",
    "protolabs/cloud",
    "protolabs/fusion",
]


def _key_from_evals_env() -> str:
    try:
        with open(_EVALS_ENV) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GATEWAY_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def resolve_gateway(gateway: dict | None = None) -> dict:
    """A PROTO_GATEWAY dict wins; otherwise env, then the evals .env key."""
    if gateway:
        return gateway
    key = (
        os.environ.get("PROTOLAB_GATEWAY_KEY")
        or os.environ.get("GATEWAY_API_KEY")
        or _key_from_evals_env()
    )
    if not key:
        logger.warning(
            "[protolab-nodes] no gateway API key found "
            "(PROTOLAB_GATEWAY_KEY / GATEWAY_API_KEY / %s)", _EVALS_ENV
        )
    return {"base_url": DEFAULT_GATEWAY_URL, "api_key": key, "timeout": DEFAULT_TIMEOUT}


def _headers(gw: dict) -> dict:
    h = {"Content-Type": "application/json"}
    if gw.get("api_key"):
        h["Authorization"] = f"Bearer {gw['api_key']}"
    return h


# ---------------------------------------------------------------- model list

_models_cache: tuple[float, list[str]] = (0.0, [])
_models_lock = threading.Lock()
_MODELS_TTL_S = 300


def list_chat_models() -> list[str]:
    """Chat lanes from the live gateway, cached; static fallback if unreachable.

    Called from INPUT_TYPES, which ComfyUI hits when building /object_info — keep
    the timeout short so a down gateway can't hang the UI.
    """
    global _models_cache
    with _models_lock:
        ts, cached = _models_cache
        if cached and time.time() - ts < _MODELS_TTL_S:
            return cached
        try:
            gw = resolve_gateway()
            r = requests.get(
                f"{gw['base_url']}/models", headers=_headers(gw), timeout=3
            )
            r.raise_for_status()
            ids = sorted(m["id"] for m in r.json()["data"])
            chat = [
                m for m in ids
                if not any(marker in m.lower() for marker in _NON_CHAT_MARKERS)
            ]
            # fast first: it's the default lane for every node in the pack
            chat.sort(key=lambda m: (m != "protolabs/fast", m))
            if chat:
                _models_cache = (time.time(), chat)
                return chat
        except Exception as e:  # noqa: BLE001 — never break UI boot on gateway state
            logger.warning("[protolab-nodes] model list fetch failed: %s", e)
        return FALLBACK_CHAT_MODELS


# ---------------------------------------------------------------- chat

def chat(
    gateway: dict | None,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    seed: int = 0,
    response_format: dict | None = None,
    thinking: str = "auto",
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """POST /chat/completions, return the choices[0].message dict.

    thinking: "auto" sends nothing; "on"/"off" send chat_template_kwargs.enable_thinking.
    Some gateway routes drop unknown params with a 400 — on that specific failure we
    retry once without the kwarg rather than kill the workflow.
    """
    gw = dict(resolve_gateway(gateway))
    if base_url:
        gw["base_url"] = base_url.rstrip("/")
    if api_key:
        gw["api_key"] = api_key

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed > 0:
        body["seed"] = seed
    if response_format:
        body["response_format"] = response_format
    if thinking in ("on", "off"):
        body["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}

    url = f"{gw['base_url']}/chat/completions"
    r = requests.post(url, headers=_headers(gw), json=body, timeout=gw.get("timeout", DEFAULT_TIMEOUT))
    if r.status_code == 400 and "chat_template_kwargs" in body:
        logger.warning("[protolab-nodes] 400 with chat_template_kwargs, retrying without")
        body.pop("chat_template_kwargs")
        r = requests.post(url, headers=_headers(gw), json=body, timeout=gw.get("timeout", DEFAULT_TIMEOUT))
    if r.status_code != 200:
        raise RuntimeError(f"gateway {r.status_code} from {url}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]


def salvage_text(message: dict) -> tuple[str, str]:
    """(content, reasoning) with the vllm#40528 unterminated-think salvage.

    Cases handled:
      * content carries an inline <think>...</think> block -> split it out
      * content empty, reasoning_content holds "thoughts</think>answer" -> rsplit
      * content empty, reasoning_content holds only the answer (parser ate it whole)
    """
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()

    if content.startswith("<think>") and "</think>" in content:
        thought, _, rest = content.partition("</think>")
        reasoning = reasoning or thought.removeprefix("<think>").strip()
        content = rest.strip()

    if not content and reasoning:
        if "</think>" in reasoning:
            thought, _, rest = reasoning.rpartition("</think>")
            reasoning, content = thought.strip(), rest.strip()
        else:
            content = reasoning

    return content, reasoning


# ---------------------------------------------------------------- converters

def image_to_data_url(image, index: int = 0) -> str:
    """ComfyUI IMAGE tensor [B,H,W,C] float 0-1 -> PNG data URL for vision requests."""
    import numpy as np
    from PIL import Image

    arr = image[index].cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def vision_content(prompt: str, image=None) -> list | str:
    if image is None:
        return prompt
    parts = [{"type": "image_url", "image_url": {"url": image_to_data_url(image, i)}}
             for i in range(image.shape[0])]
    parts.append({"type": "text", "text": prompt})
    return parts


def audio_to_wav_bytes(audio: dict) -> bytes:
    """ComfyUI AUDIO {"waveform": [B,C,T], "sample_rate": int} -> WAV bytes.

    soundfile, not torchaudio.save — ComfyUI's torchaudio build routes save through
    torchcodec, which isn't installed (core ComfyUI encodes with av/soundfile too).
    """
    import soundfile as sf

    wav = audio["waveform"]
    if wav.ndim == 3:
        wav = wav[0]
    buf = io.BytesIO()
    sf.write(buf, wav.cpu().numpy().T, int(audio["sample_rate"]), format="WAV")
    return buf.getvalue()


def bytes_to_audio(data: bytes, fmt: str) -> dict:
    """Encoded audio bytes -> ComfyUI AUDIO dict. soundfile handles wav (and mp3 on
    libsndfile >= 1.1); PyAV — ComfyUI's own decode path — is the fallback."""
    import torch

    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wav = torch.from_numpy(arr.T)  # [C, T]
    except Exception:  # noqa: BLE001 — e.g. mp3 on an old libsndfile
        import av
        import numpy as np

        frames = []
        with av.open(io.BytesIO(data)) as af:
            stream = af.streams.audio[0]
            sr = stream.rate
            for frame in af.decode(streams=stream.index):
                frames.append(frame.to_ndarray())
        arr = np.concatenate(frames, axis=-1)
        if arr.dtype.kind == "i":
            arr = arr.astype("float32") / np.iinfo(arr.dtype).max
        wav = torch.from_numpy(arr.astype("float32"))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
    return {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}


def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
