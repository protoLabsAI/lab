#!/usr/bin/env python3
"""Emit the H3 Director API graph — an LLM shot-list driving N MiniMax-H3 shots into one cut.

    idea -> ProtoStructured (gateway lane, guided JSON) -> shot list
         -> per shot: MiniMaxH3ImageToVideo -> SamplerCustomAdvanced -> video+audio decode
         -> ImageBatch / AudioConcat assembly -> one CreateVideo -> SaveVideo

Why the shape it has:

  * **The LLM picks `frames` from an enum, not a duration.** H3 snaps length to a 17k+5
    grid; a model asked for "6 seconds" produces an off-grid number that the node silently
    rounds. An enum of pre-validated grid values removes the arithmetic from the model.
  * **Every shot has its own noise seed but shares the sampler/scheduler.** Sharing the
    scheduler keeps steps consistent across the cut; separate seeds keep the shots from
    collapsing onto the same motion.
  * **`--ref-chain` runs ref2va for EVERY shot, including the first.** ref2va accepts zero
    reference images and is then a plain t2v path (verified 2026-08-19), so the whole graph
    needs one checkpoint instead of pairing fl2va with ref2va. One model is simply simpler:
    a single scheduler, a single guider source, no swap mid-graph.
  * **Shots are saved individually AND as one cut.** The per-shot files are what you
    re-cut from when one shot out of three is wrong — regenerating all three to fix one
    is the failure mode this avoids.
  * **ProtoJSONGet degrades instead of failing.** A short shot list falls back to the
    node's default prompt, so the graph still renders and you can see what happened in
    the Show Text nodes rather than reading a red node.

The H3 node map (loaders, sampler chain, decode) is the one recovered from the 2026-08-06
characterization and used by experiments/video-worldmodel/run_battery_h3.py — not re-derived.

Usage:
  python build_h3_director.py > workflows/h3-director.api.json
  python build_h3_director.py --shots 4 --steps 8 --lora turbo8
"""
from __future__ import annotations

import argparse
import json

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
UNET_REF = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VVAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"
LORAS = {
    "turbo8": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    "turbo4": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
}

# Frame counts on H3's 17k+5 grid, inside the trained 124-362 range. At 24 fps:
# 124 = 5.2s, 175 = 7.3s, 226 = 9.4s, 277 = 11.5s.
FRAME_ENUM = [124, 175, 226, 277]

SYSTEM = """\
You are the director of a short generated film. You turn one raw idea into a shot list for \
MiniMax H3, a model that generates picture and sound together in a single pass.

Write exactly {n} shots that read as one continuous piece: the same characters, wardrobe, \
location, time of day, weather and visual style throughout, advancing in time from shot to \
shot. Each shot is generated independently and cannot see the others, so every shot's prompt \
must restate the style and re-describe the subject in the same concrete words. Consistency \
comes from repeating the description verbatim, not from referring back.

Each shot's `prompt` is ONE flowing paragraph, at most 150 words:

- Open with a short style clause, identical in every shot, then a colon. \
Example: "Realistic live-action cinematic look, warm tungsten interior:".
- State the main action first, in the present progressive ("is walking", "is speaking"), \
then specific movements and gestures, then the precise appearance of the subject, then the \
environment.
- Always state the camera: static frame, slow pan, tracking shot, dolly in, push in, pull \
back, orbit, handheld, overhead, and the framing (wide, medium, close). Name where the shot \
ends up ("settling into a close-up of her hands"). Unspecified camera means random drift.
- Weave the sound chronologically alongside the actions, never appended at the end, and be \
specific: "boot heels on wet concrete", not "ambient sound". Dialogue is the exact words in \
quotes with the voice described: "says quietly, in a low even voice: 'we are out of time'". \
Pure ambience without any speech or music is this model's weakest audio — give most shots \
something definite to hear.
- Restrained, concrete language. "Red coat", not "vibrant crimson coat". Never use quality \
words like epic, stunning, masterpiece, 8K, best quality — they measurably hurt.
- Only what a camera and a microphone could capture. No smells, no thoughts, no internal \
states: render feeling as an observable cue ("her jaw tightens and she looks away").
- No timestamps, no scene headings, no markdown, no shot numbers inside the prompt text.

Pick each shot's `frames` from the allowed values by how much action it holds: 124 for a \
single beat, 175 for a normal shot, 226 or 277 only when the action genuinely needs the time.

`title` is a short slug for the piece, lowercase words separated by underscores.\
{ref}"""

REF_CHAIN_RULE = """

Shot 1 establishes the subject and is generated from your words alone, so describe the subject \
in full there. Every later shot is generated with a still frame from shot 1 supplied as a \
reference image, addressed in the prompt as <Picture 1>. In shots 2 and later, refer to the \
subject through that reference — "the man from <Picture 1> is now kneeling by the door" — \
instead of re-describing their face and build, but keep restating the style clause, the \
wardrobe and the location. The reference carries identity; your words still carry everything \
else."""

FALLBACK_PROMPT = (
    "Realistic live-action cinematic look, overcast daylight: a person in a grey coat is "
    "standing still at the end of an empty pier, looking out at flat water as wind moves "
    "their hair, the camera holding a slow push in from a wide shot to a medium, wind "
    "hissing over the microphone and water slapping the pilings below."
)


def schema(n_shots: int) -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "shots": {
                    "type": "array",
                    "minItems": n_shots,
                    "maxItems": n_shots,
                    "items": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "frames": {"type": "integer", "enum": FRAME_ENUM},
                        },
                        "required": ["prompt", "frames"],
                    },
                },
            },
            "required": ["title", "shots"],
        },
        indent=2,
    )


def build(
    idea: str,
    n_shots: int = 3,
    steps: int = 20,
    lora: str | None = None,
    lora_strength: float = 1.0,
    aspect: str = "16:9 (Widescreen)",
    megapixels: float = 0.4,
    fps: float = 24.0,
    director_model: str = "protolabs/smart",
    seed0: int = 101,
    prefix: str = "video/H3_Director",
    ref_chain: bool = False,
) -> dict:
    g: dict = {
        # ---- shared model stack -------------------------------------------------
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_REF if ref_chain else UNET, "weight_dtype": "default"},
        },
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VVAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
        # ---- the idea, and the director that expands it -------------------------
        "10": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": idea}},
        "11": {
            "class_type": "ProtoStructured",
            "inputs": {
                "model": director_model,
                "prompt": ["10", 0],
                "json_schema": schema(n_shots),
                "system": SYSTEM.format(n=n_shots, ref=REF_CHAIN_RULE if ref_chain else ''),
                "temperature": 0.6,
                "max_tokens": 8192,
                "seed": 0,
            },
        },
        "12": {"class_type": "ProtoShowText", "inputs": {"text": ["11", 0]}},
        # ---- frame geometry, shared by every shot -------------------------------
        "13": {
            "class_type": "ResolutionSelector",
            "inputs": {"aspect_ratio": aspect, "megapixels": megapixels, "multiple": 32},
        },
        # ---- sampler pieces shared across shots ---------------------------------
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "15": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0},
        },
    }

    model_ref = ["1", 0]
    if lora and ref_chain:
        raise SystemExit("--lora is fl2v-trained and does not apply to the ref2va shots; use --lora none with --ref-chain")
    if lora:
        g["16"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": LORAS[lora], "strength_model": lora_strength},
        }
        model_ref = ["16", 0]
        g["15"]["inputs"]["model"] = model_ref

    # ---- one branch per shot ----------------------------------------------------
    for i in range(n_shots):
        b = 100 + i * 20  # node-id block per shot, so ids stay readable in the UI
        p, f, sh = str(b), str(b + 1), str(b + 2)
        g[p] = {
            "class_type": "ProtoJSONGet",
            "inputs": {"json_string": ["11", 0], "path": f"shots[{i}].prompt", "default": FALLBACK_PROMPT},
        }
        g[f] = {
            "class_type": "ProtoJSONGet",
            "inputs": {"json_string": ["11", 0], "path": f"shots[{i}].frames", "default": str(FRAME_ENUM[0])},
        }
        g[sh] = {"class_type": "ProtoShowText", "inputs": {"text": [p, 0]}}
        if ref_chain:
            g[str(b + 3)] = {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "audio_vae": ["4", 0],
                    "prompt": [p, 0],
                    "width": ["13", 0],
                    "height": ["13", 1],
                    "length": [f, 1],  # ProtoJSONGet slot 1 = INT
                    "ref_image_size": "match",
                },
            }
            if i > 0:
                # autogrow inputs are addressed by their dotted API name
                g[str(b + 3)]["inputs"]["ref_images.ref_image_0"] = ["800", 0]
        else:
            g[str(b + 3)] = {
                "class_type": "MiniMaxH3ImageToVideo",
                "inputs": {
                    "clip": ["2", 0],
                    "vae": ["3", 0],
                    "prompt": [p, 0],
                    "width": ["13", 0],
                    "height": ["13", 1],
                    "length": [f, 1],  # ProtoJSONGet slot 1 = INT
                },
            }
        g[str(b + 4)] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed0 + i}}
        g[str(b + 5)] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": [str(b + 3), 0]}}
        g[str(b + 6)] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [str(b + 4), 0],
                "guider": [str(b + 5), 0],
                "sampler": ["14", 0],
                "sigmas": ["15", 0],
                "latent_image": [str(b + 3), 1],
            },
        }
        g[str(b + 7)] = {"class_type": "VAEDecode", "inputs": {"samples": [str(b + 6), 0], "vae": ["3", 0]}}
        g[str(b + 8)] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [str(b + 6), 0], "vae": ["4", 0]}}
        # per-shot artifact — what you re-cut from when one shot of N is wrong
        g[str(b + 9)] = {
            "class_type": "CreateVideo",
            "inputs": {"images": [str(b + 7), 0], "fps": fps, "audio": [str(b + 8), 0], "bit_depth": 8},
        }
        g[str(b + 10)] = {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [str(b + 9), 0],
                "filename_prefix": f"{prefix}_shot{i + 1}",
                "format": "auto",
                "codec": "auto",
            },
        }

    # ---- the reference frame: shot 1's last frame, identity for every later shot ----
    if ref_chain:
        g["800"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["107", 0], "batch_index": -1, "length": 1}}
        g["801"] = {"class_type": "SaveImage", "inputs": {"images": ["800", 0], "filename_prefix": prefix + "_ref"}}

    # ---- assemble the cut -------------------------------------------------------
    img, aud = [str(100 + 7), 0], [str(100 + 8), 0]
    for i in range(1, n_shots):
        b = 100 + i * 20
        ib, ab = str(900 + i), str(950 + i)
        g[ib] = {"class_type": "ImageBatch", "inputs": {"image1": img, "image2": [str(b + 7), 0]}}
        g[ab] = {
            "class_type": "AudioConcat",
            "inputs": {"audio1": aud, "audio2": [str(b + 8), 0], "direction": "after"},
        }
        img, aud = [ib, 0], [ab, 0]

    g["990"] = {"class_type": "CreateVideo", "inputs": {"images": img, "fps": fps, "audio": aud, "bit_depth": 8}}
    g["991"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["990", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"},
    }
    return g


DEFAULT_IDEA = (
    "A lighthouse keeper on a rocky island realises a storm is coming in, and secures the "
    "lamp room before the weather hits."
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", default=DEFAULT_IDEA)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lora", choices=list(LORAS) + ["none"], default="none")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--aspect", default="16:9 (Widescreen)")
    ap.add_argument("--megapixels", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--model", default="protolabs/smart")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--prefix", default="video/H3_Director")
    ap.add_argument("--ref-chain", action="store_true",
                    help="shot 1 via t2v, its last frame becomes the reference image for every later shot")
    a = ap.parse_args()
    print(
        json.dumps(
            build(
                a.idea,
                n_shots=a.shots,
                steps=a.steps,
                lora=None if a.lora == "none" else a.lora,
                lora_strength=a.lora_strength,
                aspect=a.aspect,
                megapixels=a.megapixels,
                fps=a.fps,
                director_model=a.model,
                seed0=a.seed,
                prefix=a.prefix,
                ref_chain=a.ref_chain,
            ),
            indent=1,
        )
    )
