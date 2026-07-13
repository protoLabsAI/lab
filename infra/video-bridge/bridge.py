"""protoLabs video bridge — OpenAI /v1/videos over ComfyUI (LTX-2.3 NVFP4).

Co-located with ComfyUI on protolabs:8188 (issue protoBanana#38, piece 2). Implements
exactly the three AGREED contract URLs so protoDirector's video runner + any `sora-2`/
`veo-*` client hit `protolabs/ltx2-*` identically:

    POST /v1/videos              {model, prompt, seconds, size, ...} -> {id, status:"queued"}
    GET  /v1/videos/{id}         -> {id, status, progress, error?}
    GET  /v1/videos/{id}/content -> mp4 bytes

Job store is JSON-file-backed (restart-survivable job ids, #38 hard req): the job->prompt_id
map persists, and status is re-derived from ComfyUI's /history (which also survives a bridge
restart), so polling resumes cleanly after a crash/redeploy. mp4 bytes are read off ComfyUI's
output dir on local disk — no multi-hundred-MB clips hauled through an extra hop.

Reuses protoBanana's ComfyUIClient (the #39 cache-nonce fix comes along for free). protoBanana
itself stays image-only; this is a separate service that depends on it.

Run:  uvicorn bridge:app --host 0.0.0.0 --port 8100
Env:  COMFY_URL (default http://127.0.0.1:8188), COMFY_OUTPUT_DIR (/mnt/data/ltx-out),
      JOB_STORE (/mnt/data/ltx-out/video-bridge/jobs.json), MODEL_PREFIX (protolabs/ltx2)
"""
from __future__ import annotations
import os, sys, json, time, uuid, asyncio, pathlib, subprocess
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response

# co-located protoBanana checkout provides the reusable ComfyUI client
sys.path.insert(0, os.path.expanduser("~/dev/protoBanana"))
from protobanana.client import ComfyUIClient  # noqa: E402

import inject  # noqa: E402

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = pathlib.Path(os.environ.get("COMFY_OUTPUT_DIR", "/mnt/data/ltx-out"))
# co-located: uploaded guide clips are written straight to ComfyUI's input dir (LoadVideo reads from there)
INPUT_DIR = pathlib.Path(os.environ.get("COMFY_INPUT_DIR", os.path.expanduser("~/dev/ComfyUI/input")))
JOB_STORE = pathlib.Path(os.environ.get("JOB_STORE", "/mnt/data/ltx-out/video-bridge/jobs.json"))
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "protolabs/ltx2")
_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".avi", ".m4v")


def _probe_video(path: pathlib.Path) -> tuple[int, int, int]:
    """(width, height, nb_frames) via ffprobe; nb_frames counted if the container omits it."""
    def q(entries: str, extra: list[str] = []):
        return subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", entries, *extra, "-of", "default=nw=1:nk=1", str(path)],
            text=True).split()
    w, h = (int(x) for x in q("stream=width,height")[:2])
    nb = q("stream=nb_frames")
    n = int(nb[0]) if nb and nb[0].isdigit() else 0
    if n == 0:  # some encoders don't tag nb_frames — count packets
        n = int(q("stream=nb_read_packets", ["-count_packets"])[0])
    return w, h, n


def _probe_image_dims(path: pathlib.Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "default=nw=1:nk=1", str(path)], text=True).split()
    return int(out[0]), int(out[1])

app = FastAPI(title="protoLabs video bridge", version="0.1-stub")
_client: Optional[ComfyUIClient] = None
_lock = asyncio.Lock()


# ---- job store (JSON-file-backed; restart-survivable) --------------------------
def _load_jobs() -> dict[str, Any]:
    if JOB_STORE.exists():
        try:
            return json.loads(JOB_STORE.read_text())
        except Exception:
            return {}
    return {}


def _save_jobs(jobs: dict[str, Any]) -> None:
    JOB_STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = JOB_STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=1))
    tmp.replace(JOB_STORE)  # atomic


async def _put(job: dict[str, Any]) -> None:
    async with _lock:
        jobs = _load_jobs()
        jobs[job["id"]] = job
        _save_jobs(jobs)


def _get(job_id: str) -> Optional[dict[str, Any]]:
    return _load_jobs().get(job_id)


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = ComfyUIClient(COMFY_URL, poll_interval_s=1.0, default_timeout_s=600.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client:
        await _client.aclose()


# ---- status derivation (re-derived from ComfyUI, so it survives bridge restart) --
async def _derive(job: dict[str, Any]) -> dict[str, Any]:
    """Refresh a job's status/progress/out_file from ComfyUI /history + /queue."""
    if job["status"] in ("completed", "failed"):
        return job
    pid = job["prompt_id"]
    http = _client.http
    hist = (await http.get(f"{COMFY_URL}/history/{pid}")).json()
    entry = hist.get(pid)
    if entry:
        st = entry.get("status", {})
        if st.get("completed"):
            if st.get("status_str") == "error":
                job.update(status="failed", progress=None,
                           error={"message": str(st.get("messages"))[:500]})
            else:
                job.update(status="completed", progress=100,
                           out_file=_find_output(entry))
            await _put(job)
            return job
    # not done yet — queued vs in_progress from /queue
    q = (await http.get(f"{COMFY_URL}/queue")).json()
    running = any(pid == item[1] for item in q.get("queue_running", []))
    job["status"] = "in_progress" if running else "queued"
    job["progress"] = None if running else 0
    return job


def _find_output(entry: dict[str, Any]) -> Optional[dict[str, str]]:
    for _nid, out in entry.get("outputs", {}).items():
        for key in ("videos", "gifs", "images"):
            for f in out.get(key) or []:
                return {"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                        "type": f.get("type", "output")}
    return None


# ---- the three contract endpoints -----------------------------------------------
@app.post("/v1/videos")
async def create_video(request: Request,
                       input_reference: Optional[UploadFile] = File(None),
                       last_frame: Optional[UploadFile] = File(None)):
    # accept both JSON and multipart (input_reference forces multipart).
    # multipart: pull ALL contract fields from the form (not just model/prompt) so an
    # I2V request doesn't silently drop seconds/size/seed/negative_prompt/extra_body.
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
    else:
        form = await request.form()
        body = {k: v for k, v in form.multi_items() if k not in ("input_reference", "last_frame")}
        # scalars that must be numeric downstream
        for k in ("seconds", "seed"):
            if k in body and body[k] not in (None, ""):
                body[k] = float(body[k]) if k == "seconds" else int(body[k])
        # extra_body arrives as a JSON string over multipart
        if isinstance(body.get("extra_body"), str):
            try:
                body["extra_body"] = json.loads(body["extra_body"])
            except json.JSONDecodeError:
                raise HTTPException(400, "extra_body must be valid JSON")
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(400, "prompt is required")
    model = body.get("model") or f"{MODEL_PREFIX}-distilled"
    extra = body.get("extra_body") or {}

    # Mode routing on the reference upload(s):
    #   input_reference image + last_frame image -> FLF (first->last frame interpolation)
    #   video reference (or extra_body.mode=="extend") -> EXTEND (continue the clip)
    #   image reference (stub)                   -> accepted but template is T2V-only (I2V TODO)
    #   no reference                             -> T2V
    ref_name = None
    mode = "t2v"
    sub_meta: Optional[dict[str, Any]] = None
    if input_reference is not None:
        fn = input_reference.filename or "ref"
        is_video = fn.lower().endswith(_VIDEO_EXTS) or str(extra.get("mode", "")).lower() == "extend"
        data = await input_reference.read()
        if last_frame is not None:
            # FIRST-LAST-FRAME: two images -> interpolate. Reject a video first frame here.
            if is_video:
                raise HTTPException(400, "first-last-frame needs image inputs, not a video")
            mode = "flf"
            first_name = await _client.upload_image(data, filename=fn)
            last_name = await _client.upload_image(
                await last_frame.read(), filename=last_frame.filename or "last.png")
            try:
                fw, fh = _probe_image_dims(INPUT_DIR / first_name)
            except Exception as e:
                raise HTTPException(400, f"could not probe first-frame image: {e}")
            ref_name = first_name
        elif is_video:
            mode = "extend"
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            ref_name = f"extend_{uuid.uuid4().hex[:16]}{pathlib.Path(fn).suffix or '.mp4'}"
            (INPUT_DIR / ref_name).write_bytes(data)  # co-located: LoadVideo reads from input dir
            try:
                gw, gh, gframes = _probe_video(INPUT_DIR / ref_name)
            except Exception as e:
                raise HTTPException(400, f"could not probe input video: {e}")
            if gframes < 9:
                raise HTTPException(400, f"guide clip too short ({gframes} frames); need >= 9")
        else:
            ref_name = await _client.upload_image(data, filename=fn)
    elif last_frame is not None:
        raise HTTPException(400, "last_frame given without input_reference (need both for first-last-frame)")

    if mode == "flf":
        wf, sub_meta = inject.build_flf_workflow(
            prompt, first_image=first_name, last_image=last_name, first_w=fw, first_h=fh,
            size=body.get("size"), seconds=body.get("seconds"), seed=body.get("seed"),
            negative_prompt=body.get("negative_prompt"), extra_body=extra)
    elif mode == "extend":
        wf, sub_meta = inject.build_extend_workflow(
            prompt, video_filename=ref_name, guide_frames=gframes, guide_w=gw, guide_h=gh,
            size=body.get("size"), seconds=body.get("seconds"), seed=body.get("seed"),
            negative_prompt=body.get("negative_prompt"), extra_body=extra)
    else:
        wf = inject.build_workflow(
            prompt, size=body.get("size"), seconds=body.get("seconds"), seed=body.get("seed"),
            negative_prompt=body.get("negative_prompt"), extra_body=extra)
    try:
        pid = await _client.submit_prompt(wf)
    except Exception as e:
        raise HTTPException(502, f"ComfyUI submit failed: {e}")

    job = {"id": f"vid_{uuid.uuid4().hex[:24]}", "prompt_id": pid, "model": model,
           "status": "queued", "progress": 0, "created_at": int(time.time()),
           "params": {"size": body.get("size"), "seconds": body.get("seconds"),
                      "input_reference": ref_name, "mode": mode,
                      "sub_meta": sub_meta}, "out_file": None, "error": None}
    await _put(job)
    return JSONResponse(_public(job), status_code=201)


@app.get("/v1/videos/{job_id}")
async def get_video(job_id: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    job = await _derive(job)
    return _public(job)


@app.get("/v1/videos/{job_id}/content")
async def get_content(job_id: str):
    job = _get(job_id)
    if not job:
        raise HTTPException(404, "unknown job id")
    job = await _derive(job)
    if job["status"] != "completed":
        raise HTTPException(409, f"job not completed (status={job['status']})")
    out = job.get("out_file")
    if not out:
        raise HTTPException(500, "completed but no output file recorded")
    # read off local disk (co-located); fall back to ComfyUI /view
    path = OUTPUT_DIR / out["subfolder"] / out["filename"]
    if path.exists():
        data = path.read_bytes()
    else:
        r = await _client.http.get(f"{COMFY_URL}/view",
                                   params={"filename": out["filename"],
                                           "subfolder": out["subfolder"], "type": out["type"]})
        r.raise_for_status()
        data = r.content
    return Response(content=data, media_type="video/mp4",
                    headers={"Content-Disposition": f'inline; filename="{out["filename"]}"'})


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """The client-facing shape (hide prompt_id / internal fields)."""
    out = {"id": job["id"], "object": "video", "model": job["model"],
           "status": job["status"], "progress": job.get("progress"),
           "created_at": job["created_at"]}
    if job.get("error"):
        out["error"] = job["error"]
    return out


@app.get("/healthz")
async def healthz():
    try:
        await _client.http.get(f"{COMFY_URL}/system_stats")
        return {"ok": True, "comfy": COMFY_URL}
    except Exception as e:
        raise HTTPException(503, f"comfy unreachable: {e}")
