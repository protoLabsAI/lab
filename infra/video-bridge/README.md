# protoLabs video bridge (stub)

OpenAI `/v1/videos` async-jobs API over ComfyUI + **LTX-2.3-22B NVFP4**, co-located with
ComfyUI on `protolabs:8188`. This is **piece 2** of [protoBanana#38](https://github.com/protoLabsAI/protoBanana/issues/38) —
the standalone video bridge the team settled on (litellm's native `/v1/videos` router shadows
passthrough routes, so the job layer can't live inside the gateway). protoBanana stays
image-only; this depends on it (reuses `ComfyUIClient`, inherits the #39 cache-nonce fix).

## Contract (AGREED — `protoDirector/GATEWAY_CONTRACT.md`)

```
POST /v1/videos              {model, prompt, seconds, size, negative_prompt?, seed?, extra_body?}
                             (+ optional multipart input_reference for first-frame/I2V)
                             -> 201 {id, object:"video", model, status:"queued", progress, created_at}
GET  /v1/videos/{id}         -> {id, status: queued|in_progress|completed|failed, progress, error?}
GET  /v1/videos/{id}/content -> mp4 bytes (video/mp4)
```

- **Job id survives restarts** — `job_id -> prompt_id` persisted to `JOB_STORE`; status is
  re-derived from ComfyUI `/history` (also persistent), so a client resumes polling after a
  bridge crash/redeploy.
- **Bytes off local disk** — `/content` reads ComfyUI's output dir directly; `/view` fallback.
- **LTX knobs in `extra_body`** — `fps`, `seed`, `negative_prompt` (model-agnostic surface stays clean).

## Run

```bash
pip install -r requirements.txt
uvicorn bridge:app --host 0.0.0.0 --port 8100          # needs ComfyUI up on :8188
# smoke:
python smoke.py
```

Env: `COMFY_URL` (`http://127.0.0.1:8188`), `COMFY_OUTPUT_DIR` (`/mnt/data/ltx-out`),
`JOB_STORE` (`/mnt/data/ltx-out/video-bridge/jobs.json`), `MODEL_PREFIX` (`protolabs/ltx2`).

## Edge routing (homelab-iac task)

Route `/v1/videos*` → this bridge, everything else → the gateway unchanged. Client contract
holds exactly as written (same base URL, key, shapes). When `sora-2`/`veo-*` become real, the
bridge dispatches by `model`: `protolabs/ltx2-*` local, anything else proxied to the gateway's
native `/v1/videos`.

## What's stubbed / next

- **`seconds` → frames** snaps to LTX's `8n+1` at `fps` (default 30). Verify the mapping matches
  intended clip length on real requests.
- **`input_reference` (I2V)** is accepted + uploaded, but the template is **T2V-only**; wiring
  first-frame conditioning needs the I2V workflow variant (un-bypass the "Load Image" group).
- **Fine progress** — currently coarse (queued 0 / in_progress null / completed 100). ComfyUI's
  `/ws` gives per-step progress; wire it for a real 0-100.
- **Concurrency** — single ComfyUI instance serializes; `n=1` only (per contract v1).

Workflow: `workflows/ltx2-t2v.json` (distilled-decode path, Full fork removed). Injection map
in `inject.py`.
