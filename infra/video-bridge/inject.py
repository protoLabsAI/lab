"""Inject OpenAI /v1/videos request params into the LTX-2.3 T2V ComfyUI workflow.

Injection map (ltx2-t2v.json, distilled-decode path):
    prompt          -> node 2483 CLIPTextEncode.text (positive)
    negative_prompt -> node 2612 CLIPTextEncode.text (negative)
    size WxH        -> node 3059 EmptyLTXVLatentVideo.width/height
    seconds         -> node 4979 PrimitiveInt.value (frames, snapped to 8n+1)
    seed            -> node 4832 RandomNoise.noise_seed
LTX-specific knobs (fps, etc.) arrive via extra_body and are applied where present.
"""
from __future__ import annotations
import copy, json, os
from typing import Any

_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "workflows", "ltx2-t2v.json")

# node ids in ltx2-t2v.json (ComfyUI API format)
N_POS, N_NEG, N_LATENT, N_FRAMES, N_NOISE = "2483", "2612", "3059", "4979", "4832"
DEFAULT_FPS = 30
DEFAULT_NEG = "pc game, console game, video game, cartoon, childish, ugly"


def load_template() -> dict[str, Any]:
    return json.load(open(_TEMPLATE_PATH))


def snap_frames(seconds: float, fps: int = DEFAULT_FPS) -> int:
    """LTX latent length must be 8n+1. Map seconds*fps to the nearest valid count."""
    raw = max(1, round(float(seconds) * fps))
    n = round((raw - 1) / 8)
    return max(9, n * 8 + 1)


def parse_size(size: str | None) -> tuple[int, int]:
    if not size:
        return 1280, 704
    w, h = size.lower().split("x")
    return int(w), int(h)


def build_workflow(
    prompt: str,
    *,
    size: str | None = None,
    seconds: float | str | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra_body or {}
    wf = copy.deepcopy(load_template())
    fps = int(extra.get("fps", DEFAULT_FPS))

    wf[N_POS]["inputs"]["text"] = prompt
    wf[N_NEG]["inputs"]["text"] = negative_prompt or extra.get("negative_prompt") or DEFAULT_NEG

    w, h = parse_size(size)
    wf[N_LATENT]["inputs"]["width"] = w
    wf[N_LATENT]["inputs"]["height"] = h

    if seconds is not None:
        wf[N_FRAMES]["inputs"]["value"] = snap_frames(float(seconds), fps)

    # seed: explicit > extra_body.seed > leave template default.
    # NOTE: protoBanana PR #39's per-submission SaveImage nonce defeats ComfyUI's exec
    # cache server-side, so a fixed/absent seed no longer returns a stale cached clip.
    s = seed if seed is not None else extra.get("seed")
    if s is not None:
        wf[N_NOISE]["inputs"]["noise_seed"] = int(s)

    return wf
