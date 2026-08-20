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
_A2V_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "workflows", "ltx2-a2v.json")

# node ids in ltx2-t2v.json (ComfyUI API format)
N_POS, N_NEG, N_LATENT, N_FRAMES, N_NOISE = "2483", "2612", "3059", "4979", "4832"
# extra ids in ltx2-extend.json: LoadVideo guide + decoupled total-length primitive
N_VIDEO, N_TOTAL = "2004", "4986"
# ltx2-flf.json: first-frame LoadImage (N_VIDEO id reused) + last-frame LoadImage
N_IMG_FIRST, N_IMG_LAST = "2004", "2006"
# ltx2-a2v.json: LoadAudio (drive) node
N_LOADAUD = "2020"
DEFAULT_FPS = 30
A2V_FPS = 24  # ltx2-a2v.json renders at 24fps (CreateVideo primitive 4978)
DEFAULT_NEG = "pc game, console game, video game, cartoon, childish, ugly"


def load_template() -> dict[str, Any]:
    return json.load(open(_TEMPLATE_PATH))


def load_extend_template() -> dict[str, Any]:
    return json.load(open(_EXTEND_TEMPLATE_PATH))


def load_flf_template() -> dict[str, Any]:
    return json.load(open(_FLF_TEMPLATE_PATH))


def load_a2v_template() -> dict[str, Any]:
    return json.load(open(_A2V_TEMPLATE_PATH))


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


def build_a2v_workflow(
    prompt: str,
    *,
    audio_filename: str,
    audio_seconds: float,
    size: str | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """AUDIO->VIDEO: generate a clip whose scene is directed by an input audio track.

    The audio analog of extend: `LoadAudio -> LTXVAudioVAEEncode -> LTXVSetAudioRefTokens`
    injects the composed audio as a reference into the guiders (frozen_audio -> AV concat), and
    the video length is sized to the audio duration (24fps to match the graph's CreateVideo). The
    model still RE-RENDERS audio, so the bridge muxes the ORIGINAL track back onto the output in
    /content — this workflow only cares about the video. Returns (workflow, meta).
    """
    extra = extra_body or {}
    wf = copy.deepcopy(load_a2v_template())

    wf[N_POS]["inputs"]["text"] = prompt
    wf[N_NEG]["inputs"]["text"] = negative_prompt or extra.get("negative_prompt") or DEFAULT_NEG
    wf[N_LOADAUD]["inputs"]["audio"] = audio_filename

    w, h = parse_size(size) if size else (768, 512)
    wf[N_LATENT]["inputs"]["width"] = w
    wf[N_LATENT]["inputs"]["height"] = h

    # video length tracks the audio duration (the audio is the knob)
    frames = snap_frames(audio_seconds, A2V_FPS)
    wf[N_FRAMES]["inputs"]["value"] = frames

    s = seed if seed is not None else extra.get("seed")
    if s is not None:
        wf[N_NOISE]["inputs"]["noise_seed"] = int(s)

    meta = {"mode": "a2v", "audio_seconds": round(float(audio_seconds), 2),
            "video_frames": frames, "fps": A2V_FPS, "size": f"{w}x{h}"}
    return wf, meta


# ============================ MiniMax H3 (joint audio+video) ============================
# Same bridge contract, different model family: `protolabs/minimax-h3`. One template
# (h3-t2v.json) serves t2v / i2v / flf — MiniMaxH3ImageToVideo takes optional
# first_frame / last_frame inputs which the builder wires in when references are given.
# H3 renders its own audio natively (32kHz stereo in the same forward pass), so there is
# no a2v mux path — the mp4 that lands in the output dir is already the final AV asset.

_H3_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "workflows", "h3-t2v.json")
# node ids in h3-t2v.json
H3_MAIN, H3_NOISE, H3_SAVE = "5", "8", "14"
H3_FPS = 24
H3_MIN_FRAMES, H3_MAX_FRAMES = 124, 481  # trained 124-362; 481 (20s) validated locally 2026-08-07


def load_h3_template() -> dict[str, Any]:
    return json.load(open(_H3_TEMPLATE_PATH))


def snap_h3_frames(seconds: float) -> int:
    """H3 frame counts live on a 17k+5 grid at 24fps; clamp to the supported window."""
    raw = max(1, round(float(seconds) * H3_FPS))
    n = raw + (5 - (raw % 17)) % 17  # snap UP to 17k+5 (matches the official template math)
    return min(H3_MAX_FRAMES, max(H3_MIN_FRAMES, n))


def snap_h3_size(size: str | None) -> tuple[int, int]:
    """H3 canvas: multiples of 32, native short edge <= 768 (cap 1344x768)."""
    if not size:
        return 864, 480
    w, h = parse_size(size)
    w, h = (max(32, (d // 32) * 32) for d in (w, h))
    return min(w, 1344), min(h, 1344)


def build_h3_workflow(
    prompt: str,
    *,
    size: str | None = None,
    seconds: float | str | None = None,
    seed: int | None = None,
    first_image: str | None = None,
    last_image: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """T2V by default; i2v with first_image; flf with first_image+last_image.

    H3 has no negative-prompt input (CFG-free BasicGuider path) — negative_prompt is
    accepted upstream and ignored here rather than 400ing existing clients.
    """
    extra = extra_body or {}
    wf = copy.deepcopy(load_h3_template())

    wf[H3_MAIN]["inputs"]["prompt"] = prompt
    w, h = snap_h3_size(size)
    wf[H3_MAIN]["inputs"]["width"] = w
    wf[H3_MAIN]["inputs"]["height"] = h
    frames = snap_h3_frames(float(seconds)) if seconds is not None else H3_MIN_FRAMES
    wf[H3_MAIN]["inputs"]["length"] = frames

    mode = "t2v"
    if first_image:
        wf["30"] = {"class_type": "LoadImage", "inputs": {"image": first_image}}
        wf[H3_MAIN]["inputs"]["first_frame"] = ["30", 0]
        mode = "i2v"
    if last_image:
        if not first_image:
            raise ValueError("h3 flf needs first_image when last_image is given")
        wf["31"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
        wf[H3_MAIN]["inputs"]["last_frame"] = ["31", 0]
        mode = "flf"

    if steps := extra.get("steps"):
        wf["7"]["inputs"]["steps"] = int(steps)
    s = seed if seed is not None else extra.get("seed")
    if s is not None:
        wf[H3_NOISE]["inputs"]["noise_seed"] = int(s)

    meta = {"mode": f"h3-{mode}", "video_frames": frames, "fps": H3_FPS,
            "size": f"{w}x{h}", "native_audio": True}
    return wf, meta
