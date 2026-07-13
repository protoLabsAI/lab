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
_EXTEND_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "workflows", "ltx2-extend.json")
_FLF_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "workflows", "ltx2-flf.json")

# node ids in ltx2-t2v.json (ComfyUI API format)
N_POS, N_NEG, N_LATENT, N_FRAMES, N_NOISE = "2483", "2612", "3059", "4979", "4832"
# extra ids in ltx2-extend.json: LoadVideo guide + decoupled total-length primitive
N_VIDEO, N_TOTAL = "2004", "4986"
# ltx2-flf.json: first-frame LoadImage (N_VIDEO id reused) + last-frame LoadImage
N_IMG_FIRST, N_IMG_LAST = "2004", "2006"
DEFAULT_FPS = 30
DEFAULT_NEG = "pc game, console game, video game, cartoon, childish, ugly"


def load_template() -> dict[str, Any]:
    return json.load(open(_TEMPLATE_PATH))


def load_extend_template() -> dict[str, Any]:
    return json.load(open(_EXTEND_TEMPLATE_PATH))


def load_flf_template() -> dict[str, Any]:
    return json.load(open(_FLF_TEMPLATE_PATH))


# --- LTX latent framing (8x causal VAE): px<->lat, all frame counts are 8n+1 ---
def _to_lat(px: int) -> int: return (px - 1) // 8 + 1
def _to_px(lat: int) -> int: return (lat - 1) * 8 + 1
def snap_8n1(px: int) -> int: return _to_px(_to_lat(max(1, px)))


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


def build_extend_workflow(
    prompt: str,
    *,
    video_filename: str,
    guide_frames: int,
    guide_w: int,
    guide_h: int,
    size: str | None = None,
    seconds: float | str | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Video-EXTEND: continue an input clip. `LTXVAddGuide` (frame_idx=0) APPENDS the guide's
    latent frames to a generated canvas, so total_lat = canvas_lat + guide_lat, and the guide
    must fit the canvas (guide_lat <= canvas_lat -> min total = 2*guide). We make the requested
    TOTAL the knob: derive canvas = total - guide, and size the audio to the real total (its own
    primitive N_TOTAL) so A/V stay in sync. Returns (workflow, meta) where meta reports what was
    actually emitted (guide/canvas/total frames) for the job record.
    """
    extra = extra_body or {}
    wf = copy.deepcopy(load_extend_template())
    fps = int(extra.get("fps", DEFAULT_FPS))

    wf[N_POS]["inputs"]["text"] = prompt
    wf[N_NEG]["inputs"]["text"] = negative_prompt or extra.get("negative_prompt") or DEFAULT_NEG
    wf[N_VIDEO]["inputs"]["file"] = video_filename

    # size: explicit > guide's native resolution (avoids a resize/distortion)
    w, h = parse_size(size) if size else (guide_w, guide_h)
    wf[N_LATENT]["inputs"]["width"] = w
    wf[N_LATENT]["inputs"]["height"] = h

    g_lat = _to_lat(snap_8n1(guide_frames))
    # target total: from `seconds` if given, else a continuation ~= the guide's length (2x guide)
    total_lat = _to_lat(snap_frames(seconds, fps)) if seconds is not None else 2 * g_lat
    # guard the code's constraint (guide must fit canvas): min total_lat = 2*g_lat
    clamped = total_lat < 2 * g_lat
    total_lat = max(total_lat, 2 * g_lat)
    canvas_lat = total_lat - g_lat

    wf[N_FRAMES]["inputs"]["value"] = _to_px(canvas_lat)   # canvas (generated region)
    wf[N_TOTAL]["inputs"]["value"] = _to_px(total_lat)     # real total -> audio length

    s = seed if seed is not None else extra.get("seed")
    if s is not None:
        wf[N_NOISE]["inputs"]["noise_seed"] = int(s)

    meta = {"mode": "extend", "guide_frames": _to_px(g_lat), "canvas_frames": _to_px(canvas_lat),
            "total_frames": _to_px(total_lat), "size": f"{w}x{h}", "min_clamped": clamped}
    return wf, meta


def build_flf_workflow(
    prompt: str,
    *,
    first_image: str,
    last_image: str,
    first_w: int,
    first_h: int,
    size: str | None = None,
    seconds: float | str | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """FIRST-LAST-FRAME: generate a clip that starts at `first_image` and ends at `last_image`.
    Two CHAINED `LTXVAddGuide` (frame_idx 0 and -1); each single-frame image guide appends 1 latent
    frame, so total_lat = canvas_lat + 2. TOTAL is the knob; audio is sized to the real total.
    Returns (workflow, meta).
    """
    extra = extra_body or {}
    wf = copy.deepcopy(load_flf_template())
    fps = int(extra.get("fps", DEFAULT_FPS))

    wf[N_POS]["inputs"]["text"] = prompt
    wf[N_NEG]["inputs"]["text"] = negative_prompt or extra.get("negative_prompt") or DEFAULT_NEG
    wf[N_IMG_FIRST]["inputs"]["image"] = first_image
    wf[N_IMG_LAST]["inputs"]["image"] = last_image

    # size: explicit > first image's native resolution
    w, h = parse_size(size) if size else (first_w, first_h)
    wf[N_LATENT]["inputs"]["width"] = w
    wf[N_LATENT]["inputs"]["height"] = h

    # target total length; default ~4s if unspecified. 2 image keyframes append 2 latent frames.
    total_lat = _to_lat(snap_frames(seconds, fps)) if seconds is not None else _to_lat(snap_frames(4, fps))
    canvas_lat = max(1, total_lat - 2)
    total_lat = canvas_lat + 2

    wf[N_FRAMES]["inputs"]["value"] = _to_px(canvas_lat)
    wf[N_TOTAL]["inputs"]["value"] = _to_px(total_lat)

    s = seed if seed is not None else extra.get("seed")
    if s is not None:
        wf[N_NOISE]["inputs"]["noise_seed"] = int(s)

    meta = {"mode": "flf", "canvas_frames": _to_px(canvas_lat), "total_frames": _to_px(total_lat),
            "size": f"{w}x{h}"}
    return wf, meta
