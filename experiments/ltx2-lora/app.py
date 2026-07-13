#!/usr/bin/env python3
"""Gradio UI for LTX-2.3 video LoRAs: TRAIN one from clips, then GENERATE + COMPARE videos
with it against the NVFP4 base — all in the browser.

  Train tab:            clips (upload/folder) -> caption→preprocess→train→wire -> LoRA in ComfyUI
  Generate & Compare:   pick a LoRA, prompt -> video; or A/B base-vs-LoRA (same seed) side by side

Talks to ComfyUI on :8188 (must be up). Run:
  ~/dev/LTX-2/.venv/bin/python experiments/ltx2-lora/app.py     (UI on :7862)
"""
import gradio as gr, subprocess, os, shutil, signal, json, time, urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ORCH = HERE / "make_video_lora.py"
PY = str(Path.home() / "dev/LTX-2/.venv/bin/python")
COMFY = "http://127.0.0.1:8188"
OUT = "/mnt/data/ltx-out"
LORA_DIR = Path.home() / "dev/ComfyUI/models/loras"
BASE_API = "/mnt/data/ltx-out/base_api.json"   # ComfyUI's own serialized fp4 T2V graph
BUCKETS = ["768x512x49", "960x544x49", "768x448x25", "512x320x25", "1280x704x49"]
_proc = {"p": None}

# ------------------------------- TRAIN -------------------------------
def train(mode, uploads, folder, style, rank, steps, bucket, with_audio, auto_caption, gpu):
    if not style or not style.strip():
        yield "❌ provide a style / LoRA name"; return
    style = style.strip().replace(" ", "_")
    work = Path(f"/mnt/data/video-lora/{style}")
    if mode == "Upload clips":
        if not uploads:
            yield "❌ upload at least one clip"; return
        clips = work / "raw"; clips.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            shutil.copy2(f, clips / Path(f).name)
        clips_dir = str(clips)
    else:
        if not folder or not os.path.isdir(folder):
            yield f"❌ not a folder on the box: {folder}"; return
        clips_dir = folder
    cmd = [PY, str(ORCH), clips_dir, style, "--rank", str(int(rank)), "--steps", str(int(steps)),
           "--bucket", bucket, "--gpu", str(int(gpu))]
    if with_audio: cmd.append("--with-audio")
    if not auto_caption: cmd.append("--no-caption")
    log = "$ " + " ".join(cmd) + "\n\n"; yield log
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, preexec_fn=os.setsid)
    _proc["p"] = p
    for line in p.stdout:
        log += line; yield log
    p.wait(); _proc["p"] = None
    log += (f"\n\n✅ DONE — '{style}' LoRA trained + wired. Switch to **Generate & Compare**, hit "
            f"↻ Refresh, pick `{style}.safetensors`." if p.returncode == 0
            else f"\n\n❌ exited {p.returncode} — see log above.")
    yield log

def stop():
    p = _proc.get("p")
    if p and p.poll() is None:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM); return "🛑 stop sent."
    return "nothing running."

# --------------------------- GENERATE ---------------------------
def list_loras():
    return ["(none / base)"] + sorted(f.name for f in LORA_DIR.glob("*.safetensors")) if LORA_DIR.exists() else ["(none / base)"]

def _upload_image(path):
    """POST an image to ComfyUI /upload/image; return the stored filename for LoadImage."""
    import mimetypes, uuid
    data = open(path, "rb").read()
    name = f"ref_{uuid.uuid4().hex[:12]}{Path(path).suffix or '.png'}"
    boundary = "----lorastudio" + uuid.uuid4().hex
    ct = mimetypes.guess_type(path)[0] or "image/png"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ct}\r\n\r\n").encode() + data + \
           (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n"
            f"--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"{COMFY}/upload/image", body,
                                 {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req)).get("name", name)

def _build(prompt, seed, lora, strength, w, h, ref_image=None, ref_strength=0.7):
    api = json.load(open(BASE_API)); api.pop("4823", None)          # distilled-only
    if prompt.strip():
        api["2483"]["inputs"]["text"] = prompt
    api["4832"]["inputs"]["noise_seed"] = int(seed)
    api["3059"]["inputs"]["width"] = int(w); api["3059"]["inputs"]["height"] = int(h)
    # reference image (I2V first-frame conditioning): flip bypass off + feed the image
    if ref_image and "2004" in api and "4977" in api:
        fname = _upload_image(ref_image)
        api["2004"]["inputs"]["image"] = fname
        api["4977"]["inputs"]["value"] = False                      # bypass off -> I2V
        if "3159" in api:
            api["3159"]["inputs"]["strength"] = float(ref_strength)
    if lora and lora != "(none / base)":
        ckpt = next(n for n, v in api.items() if v["class_type"] == "CheckpointLoaderSimple")
        guiders = [(n, inp) for n, v in api.items() for inp, val in v["inputs"].items()
                   if isinstance(val, list) and val[0] == ckpt and val[1] == 0 and "Guider" in v["class_type"]]
        api["9001"] = {"class_type": "LoraLoaderModelOnly",
                       "inputs": {"model": [ckpt, 0], "lora_name": lora, "strength_model": float(strength)}}
        for gid, inp in guiders:
            api[gid]["inputs"][inp] = ["9001", 0]
    return api

def _run(api):
    pid = json.load(urllib.request.urlopen(urllib.request.Request(
        f"{COMFY}/prompt", json.dumps({"prompt": api}).encode(), {"Content-Type": "application/json"})))["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 420:
        time.sleep(2)
        h = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed"):
            for _nid, o in h[pid].get("outputs", {}).items():
                for k in ("videos", "gifs", "images"):
                    for f in o.get(k, []):
                        return f"{OUT}/{f.get('subfolder','')}/{f['filename']}".replace("//", "/")
            return None
    return None

def _check_base():
    if not os.path.exists(BASE_API):
        return "⚠️ No base T2V graph yet — run one LTX-2.3 T2V in ComfyUI once (seeds base_api.json), then retry."
    return None

def generate(prompt, lora, strength, seed, w, h, ref_image, ref_strength):
    err = _check_base()
    if err: return None, err
    try:
        v = _run(_build(prompt, seed, lora, strength, w, h, ref_image, ref_strength))
        mode = "I2V" if ref_image else "T2V"
        return v, (f"✅ {mode} · {lora} @ str {strength}, seed {seed}" if v else "❌ no output (check ComfyUI log)")
    except Exception as e:
        return None, f"❌ {e}"

def compare(prompt, lora, strength, seed, w, h, ref_image, ref_strength):
    err = _check_base()
    if err: return None, None, err
    if not lora or lora == "(none / base)":
        return None, None, "pick a LoRA to compare against base"
    try:
        base = _run(_build(prompt, seed, "(none / base)", 0, w, h, ref_image, ref_strength))
        withl = _run(_build(prompt, seed, lora, strength, w, h, ref_image, ref_strength))
        mode = "I2V" if ref_image else "T2V"
        return base, withl, f"✅ {mode}, same seed {seed}: base vs {lora}@{strength}"
    except Exception as e:
        return None, None, f"❌ {e}"

# ------------------------------- UI -------------------------------
with gr.Blocks(title="LTX-2.3 Video LoRA Studio") as demo:
    gr.Markdown("# 🎬 LTX-2.3 Video LoRA Studio\nTrain a LoRA from clips, then generate + A/B it on the **NVFP4 22B** base.")
    with gr.Tab("Train"):
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(["Upload clips", "Folder on box"], value="Upload clips", label="Clip source")
                uploads = gr.File(file_count="multiple", label="Clips", file_types=[".mp4", ".mov", ".webm", ".mkv"])
                folder = gr.Textbox(label="…or a folder path on the box", visible=False, placeholder="/mnt/data/my-clips")
                style = gr.Textbox(label="Style / LoRA name", placeholder="e.g. neon_noir")
                with gr.Row():
                    rank = gr.Slider(8, 128, value=32, step=8, label="LoRA rank")
                    steps = gr.Number(value=1000, label="Steps", precision=0)
                bucket = gr.Dropdown(BUCKETS, value="768x512x49", label="Resolution bucket (W×H×Frames)")
                with gr.Row():
                    with_audio = gr.Checkbox(False, label="Train audio branch")
                    auto_caption = gr.Checkbox(True, label="Auto-caption")
                    gpu = gr.Number(value=1, label="GPU", precision=0)
                with gr.Row():
                    btn = gr.Button("Train LoRA", variant="primary"); stopb = gr.Button("Stop")
            with gr.Column(scale=2):
                tout = gr.Textbox(label="Pipeline log", lines=30, max_lines=30, autoscroll=True)
        mode.change(lambda m: (gr.update(visible=m == "Upload clips"), gr.update(visible=m == "Folder on box")),
                    mode, [uploads, folder])
        btn.click(train, [mode, uploads, folder, style, rank, steps, bucket, with_audio, auto_caption, gpu], tout)
        stopb.click(stop, None, tout)

    with gr.Tab("Generate & Compare"):
        with gr.Row():
            with gr.Column(scale=1):
                g_prompt = gr.Textbox(label="Prompt", lines=4,
                                      value="cinematic, a lone figure walks down a rain-slicked neon street at night")
                with gr.Row():
                    g_lora = gr.Dropdown(list_loras(), value="(none / base)", label="LoRA")
                    g_refresh = gr.Button("↻", scale=0)
                with gr.Row():
                    g_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA strength")
                    g_seed = gr.Number(value=42, label="Seed", precision=0)
                with gr.Row():
                    g_w = gr.Number(value=768, label="Width", precision=0)
                    g_h = gr.Number(value=448, label="Height", precision=0)
                g_ref = gr.Image(type="filepath", label="Reference image (first frame — optional → I2V)", height=180)
                g_refstr = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="First-frame strength (I2V anchor)")
                with gr.Row():
                    g_gen = gr.Button("Generate", variant="primary")
                    g_cmp = gr.Button("Compare vs base")
                g_status = gr.Markdown()
            with gr.Column(scale=2):
                g_video = gr.Video(label="Output", autoplay=True)
                with gr.Row():
                    g_base = gr.Video(label="Base (no LoRA)", autoplay=True)
                    g_with = gr.Video(label="With LoRA", autoplay=True)
        g_refresh.click(lambda: gr.update(choices=list_loras()), None, g_lora)
        g_gen.click(generate, [g_prompt, g_lora, g_strength, g_seed, g_w, g_h, g_ref, g_refstr], [g_video, g_status])
        g_cmp.click(compare, [g_prompt, g_lora, g_strength, g_seed, g_w, g_h, g_ref, g_refstr], [g_base, g_with, g_status])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7862, show_error=True)
