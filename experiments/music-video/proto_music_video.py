#!/usr/bin/env python3
"""Compose a music video: audio -> bar-aligned a2v segments -> cut -> mux the ORIGINAL audio.

WHY IT'S BUILT THIS WAY (measured 2026-07-15, see RESULTS.md):

  * LTX makes ~8s clips. 193 frames @ 24fps = 8.04s. A 169s track is 21 segments. There is no
    whole-track path to choose; chopping is the only door.
  * a2v RE-RENDERS the audio — it does not pass yours through. Measured waveform correlation
    between input and output audio: +0.040 (chroma 0.977, so it tracks harmonically while being
    a different signal). infra/video-bridge/inject.py's build_a2v_workflow docstring says the
    same. So LTX's audio is DISCARDED and the original is muxed over the finished cut.
  * **a2v is NOT beat-locked.** Video motion vs the original audio's onsets: r=+0.036; vs LTX's
    own audio: r=+0.045. Both ~0 — it conditions on vibe/texture, not transients. Two
    consequences: (1) muxing the original costs nothing, there was never a beat-lock to lose;
    (2) **all rhythm must come from the EDIT.** The generator supplies 8s looks; the cut points
    supply the music. Which is exactly how music videos are actually cut — the editor carries the
    rhythm, not the camera.

  Hence: segment on BARS, not on arbitrary 8s boundaries, and change scene on song STRUCTURE.
  That is the whole trick. Everything else is plumbing.

Reuses infra/video-bridge/inject.py (proven node map: 2483 pos / 2612 neg / 3059 latent /
4979 frames / 4832 noise / 2020 LoadAudio) rather than re-deriving it.

Usage:
  python proto_music_video.py --audio /path/track.mp3 --prompt "..." [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time, warnings

warnings.filterwarnings("ignore")
# GPU1. GPU0 is the 3-lane vLLM fleet and is packed solid (~93G of 94.9G) — a bare device="cuda"
# lands there and OOMs on a 20MiB alloc. Every other proto_* script pins this; this one didn't.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import numpy as np

sys.path.insert(0, "/home/ava/dev/lab/infra/video-bridge")
import inject  # noqa: E402

COMFY = "http://localhost:8188"
COMFY_INPUT = "/mnt/data/comfyui/input"
COMFY_OUT = "/mnt/data/ltx-out"
FPS = inject.A2V_FPS           # 24
MAX_SEG_S = 8.04               # 193 frames @ 24fps — LTX's practical clip
MIN_SEG_S = 4.0

# Shot language per scene. a2v gives us a look, not a camera move, so scene variety is what
# makes it read as an edit rather than 21 clips of the same thing.
SHOTS = [
    "wide establishing shot, slow push in",
    "medium shot, shallow depth of field, subtle handheld drift",
    "close detail shot, macro texture, slow rack focus",
    "low angle wide, sweeping lateral camera move",
    "static locked-off frame, subject centered, atmospheric haze",
    "high angle looking down, slow rotation",
]


def sh(*a, **k):
    return subprocess.run(a, capture_output=True, text=True, **k)


def analyze(path: str) -> dict:
    """Beat/bar grid + structural boundaries. The grid is where the rhythm comes from."""
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    dur = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    # bars = every 4 beats (4/4 assumed; our corpus is 4/4 and the sidecars hardcode "4")
    bars = list(beats[::4])
    # structural boundaries from a recurrence matrix — real section changes, not guesses
    try:
        S = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        n = max(2, min(8, int(dur // 20)))
        bounds = librosa.segment.agglomerative(S, n)
        bound_t = list(librosa.frames_to_time(bounds, sr=sr))
    except Exception as e:
        print(f"  [warn] structure segmentation failed ({e}); using one scene")
        bound_t = [0.0]
    return {"duration": dur, "tempo": tempo, "beats": list(map(float, beats)),
            "bars": list(map(float, bars)), "boundaries": list(map(float, bound_t))}


def plan(a: dict) -> list[dict]:
    """Walk the bar grid; cut when a segment would exceed ~8s, and FORCE a cut at every
    structural boundary. Every cut lands on a bar -> every cut lands musically."""
    bars, dur, bounds = a["bars"], a["duration"], a["boundaries"]
    if len(bars) < 2:
        bars = [i * MAX_SEG_S for i in range(int(dur // MAX_SEG_S) + 1)]
    # COVER THE WHOLE TRACK. librosa's first beat is wherever the groove starts — on this test
    # track that was 20.6s in, because the intro is a beatless vocal sample. Seeding the grid
    # from the first BAR silently dropped the entire intro (147s covered of 169s). A music video
    # that starts 20s into the song is not a music video. So: anchor at 0, close at dur, and
    # backfill the pre-groove intro with bar-length steps so those cuts stay on the same grid.
    grid = [b for b in bars if MIN_SEG_S < b < dur - MIN_SEG_S]
    if grid and grid[0] > MIN_SEG_S:
        bar_s = 60.0 / max(a["tempo"], 1e-6) * 4
        pre, t = [], grid[0]
        while t - bar_s > MIN_SEG_S:
            t -= bar_s
            pre.append(t)
        grid = sorted(pre) + grid
    grid = [0.0] + grid + [dur]

    # Cut at the LAST BAR THAT FITS under MAX_SEG_S — never at start+MAX_SEG_S.
    # The earlier version did `end = min(next_bar, start + MAX_SEG_S)`, which silently overrode
    # the bar grid the moment the next bar sat past the ceiling: cuts landed on an arbitrary
    # 8.04s clock offset (measured 0.68s / 1.08s off the beat) and the whole point was lost.
    # a2v is NOT beat-locked, so these cut points are the ONLY rhythm in the finished video.
    # If a cut is off the bar, nothing else puts it back.
    segs, start, scene = [], 0.0, 0
    i = 0
    while start < dur - MIN_SEG_S:
        # candidate bars strictly inside (start, start+MAX_SEG_S]
        fits = [t for t in grid if start + 1e-6 < t <= start + MAX_SEG_S]
        # force a cut at a structural boundary if one falls in range and is long enough
        bnd = [b for b in bounds if b > 0 and start + MIN_SEG_S <= b <= start + MAX_SEG_S]
        if bnd:
            end = min(bnd)
            end = min(fits, key=lambda t: abs(t - end)) if fits else end   # snap to the bar
        elif fits:
            end = max(fits)                      # the last bar that fits — the normal path
        else:
            end = min(start + MAX_SEG_S, dur)    # no bar in range (beatless passage)
        end = min(end, dur)
        # Quantize the cut to the 24fps frame grid so a segment is a WHOLE number of frames.
        # Costs <=0.02s of bar accuracy (inaudible) and buys exact audio/video alignment.
        end = round(round(end * FPS) / FPS, 6)
        if end - start < MIN_SEG_S:
            if segs and end > segs[-1]["end"]:
                segs[-1]["end"] = round(end, 3)  # absorb a runt tail
            break
        segs.append({"start": round(start, 6), "end": round(end, 6), "scene": scene})
        if any(start < b <= end + 1e-6 for b in bounds if b > 0):
            scene += 1
        start = end
        i += 1
        if i > 500:
            break
    return segs


def scene_prompt(base: str, scene: int) -> str:
    return f"{base}, {SHOTS[scene % len(SHOTS)]}"


# --- ABSTRACT MODE -------------------------------------------------------------------------
# Literal music-video imagery reads as cheese (verified by eye on v3: "cinematic music video,
# 90s hip-hop soul aesthetic" + shot language = stock-footage energy). a2v conditions on vibe
# and texture, NOT narrative — it has no idea what the lyrics mean and cannot stage a scene. So
# stop asking it to. Instead: take the lyrics actually sung in each window and abstract them
# into texture/colour/motion. Plays to what the model does well and dodges the uncanny middle.
#
# Needs lyrics mapped to TIME. acestep-transcriber emits [Verse]/[Chorus] tags with NO
# timestamps, so Whisper (which returns them) does the timing while the 11B stays the
# authority on content.

ABSTRACT_SYS = (
    "You write ABSTRACT visual prompts for an experimental music video. Given a fragment of "
    "lyrics, respond with ONE prompt of purely abstract imagery evoking its FEELING.\n"
    "RULES: no people, no faces, no hands, no text or letters, no literal depiction of objects "
    "named in the lyrics, no narrative or story. Only: texture, colour, material, light, motion, "
    "scale. Think camera-less film, macro fluid, smoke, refraction, grain, decay, crystal, "
    "liquid metal, fibre, ink.\n"
    "Under 22 words. No preamble, no quotes, no explanation — output the prompt only."
)


def transcribe_timed(path: str) -> list[dict]:
    """Whisper with timestamps -> [{start,end,text}]. Timing only; the 11B owns the words."""
    import torch
    from transformers import pipeline
    print("  loading whisper-large-v3-turbo for TIMED lyrics ...")
    # NO chunk_length_s, and WORD-level timestamps.
    #   * chunk_length_s=30 (copied from proto_label_taste.py, where only the TEXT matters)
    #     produced a single chunk spanning 0.0->45.0s — so seven consecutive 8s segments all
    #     inherited the same lyric. Whisper itself warns: "Using chunk_length_s is very
    #     experimental with seq2seq models... use the model's generate method directly."
    #     Dropping it gives 38 chunks, median 3.4s, none >20s.
    #   * word level goes further: 332 chunks at 0.2s median, so any window maps exactly.
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                   dtype=torch.float16, device="cuda")
    out = asr(path, return_timestamps="word",
              generate_kwargs={"language": "en", "task": "transcribe"})
    rows = []
    for c in out.get("chunks", []) or []:
        ts = c.get("timestamp") or (None, None)
        if ts[0] is None:
            continue
        rows.append({"start": float(ts[0]), "end": float(ts[1] if ts[1] is not None else ts[0] + 2),
                     "text": (c.get("text") or "").strip()})
    del asr
    torch.cuda.empty_cache()
    print(f"  timed lyric chunks: {len(rows)}")
    return rows


def lyrics_in(rows: list[dict], a: float, b: float) -> str:
    """Lyrics sung during [a,b) — any chunk that overlaps the window."""
    return " ".join(r["text"] for r in rows if r["end"] > a and r["start"] < b).strip()


def llm_abstract(lyric: str, url: str, model: str) -> str | None:
    """Lyrics -> abstract imagery prompt.

    enable_thinking=False is LOAD-BEARING. The `fast` lane is a thinking model: with thinking on
    it NEVER terminates for this task — measured finish_reason='length' with content=None at
    max_tokens=60 AND at 600, burning the entire budget mid-thought. Off: finish='stop', 22
    tokens, clean prose. (Our "no thinking-off evals" rule is about EVALS, where it robs
    reasoning models. This is a prompt-writer; thinking is pure waste.)

    Failures are LOUD. The first version returned None on empty content and the caller silently
    fell back to the template — so every segment got generic shot-language and the run LOOKED
    fine. That is the same dead-path-looking-alive failure as everything else today.
    """
    import urllib.request
    body = {"model": model, "temperature": 0.9, "max_tokens": 80,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": ABSTRACT_SYS},
                         {"role": "user", "content": f"Lyrics: {lyric}"}]}
    try:
        r = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=90))
        ch = d["choices"][0]
        msg = ch["message"]
        t = (msg.get("content") or "").strip()
        if not t:
            # thinking-model trap: everything landed in reasoning_content, or the budget ran out
            rc = (msg.get("reasoning_content") or "").strip()
            print(f"    [LLM EMPTY] finish={ch.get('finish_reason')} "
                  f"tokens={d.get('usage',{}).get('completion_tokens')} "
                  f"reasoning_content={'yes' if rc else 'no'} -> falling back to template")
            return None
        return t.strip('"').split("\n")[0].strip() or None
    except Exception as e:
        print(f"    [LLM ERROR] {type(e).__name__}: {e} -> falling back to template")
        return None


def submit(wf: dict) -> str:
    import urllib.request
    r = urllib.request.Request(f"{COMFY}/prompt",
                              data=json.dumps({"prompt": wf, "client_id": "protolab-mv"}).encode(),
                              headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))["prompt_id"]


def wait(pid: str, timeout: int = 400) -> dict | None:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=15))
        if h:
            return list(h.values())[0]
        time.sleep(3)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--prompt", default="cinematic music video, moody film grain, "
                                        "anamorphic lens, dramatic volumetric light")
    ap.add_argument("--negative", default=None)
    ap.add_argument("--size", default="768x512")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="only first N segments (probe)")
    ap.add_argument("--out", default="/mnt/data/acestep-lora/music-video")
    ap.add_argument("--dry-run", action="store_true", help="plan only, generate nothing")
    ap.add_argument("--abstract", action="store_true",
                    help="derive each segment's prompt from the lyrics ACTUALLY SUNG in that "
                         "window (Whisper timing + LLM abstraction). Literal music-video "
                         "imagery reads as cheese; a2v does vibe, not narrative.")
    ap.add_argument("--llm-url", default="http://localhost:8040/v1")
    ap.add_argument("--llm-model", default="fast")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"analyzing {os.path.basename(args.audio)} ...")
    a = analyze(args.audio)
    segs = plan(a)
    if args.limit:
        segs = segs[: args.limit]

    bar_s = 60.0 / a["tempo"] * 4
    print(f"  duration   : {a['duration']:.1f}s")
    print(f"  tempo      : {a['tempo']:.1f} BPM  (bar = {bar_s:.2f}s)")
    print(f"  bars       : {len(a['bars'])}")
    print(f"  boundaries : {len(a['boundaries'])} -> {[round(b,1) for b in a['boundaries']]}")
    print(f"  segments   : {len(segs)}  (~{sum(s['end']-s['start'] for s in segs):.0f}s covered)")
    print(f"  scenes     : {len(set(s['scene'] for s in segs))}")
    print()
    for i, s in enumerate(segs):
        d = s["end"] - s["start"]
        print(f"    seg{i:02d} scene{s['scene']}  {s['start']:7.2f} -> {s['end']:7.2f}  "
              f"({d:.2f}s, {inject.snap_frames(d, FPS)} frames)")
    # ABSTRACT: prompt per segment from the lyrics sung in that exact window
    if args.abstract:
        print("\n  --abstract: timing lyrics with whisper, abstracting with the LLM ...")
        rows = transcribe_timed(args.audio)
        for i, s in enumerate(segs):
            lyr = lyrics_in(rows, s["start"], s["end"])
            s["lyric"] = lyr
            # <3 words = an instrumental passage, not a lyric. Whisper stretches a lone word
            # across a vocal-less intro (the first "So" spans 0->12.3s here); abstracting a
            # single word gives the LLM nothing to work with.
            if len(lyr.split()) < 3:
                s["prompt"] = (f"{args.prompt}, abstract instrumental passage, "
                               f"{SHOTS[s['scene'] % len(SHOTS)].split(',')[0]}")
                print(f"    seg{i:02d} [instrumental: {lyr[:20]!r}]")
                continue
            p = llm_abstract(lyr, args.llm_url, args.llm_model)
            s["prompt"] = f"{p}, abstract, no people, no text" if p else \
                          f"{args.prompt}, {SHOTS[s['scene'] % len(SHOTS)]}"
            print(f"    seg{i:02d} “{lyr[:44]}…”\n           -> {s['prompt'][:88]}")
    else:
        for s in segs:
            s["prompt"] = scene_prompt(args.prompt, s["scene"])

    est = len(segs) * 25
    print(f"\n  est. generate time: ~{est//60}m{est%60:02d}s at ~25s/segment")
    if args.dry_run:
        json.dump({"analysis": a, "segments": segs}, open(f"{args.out}/plan.json", "w"), indent=2)
        print(f"\n  dry run — plan written to {args.out}/plan.json")
        return 0

    mp4s = []
    for i, s in enumerate(segs):
        d = s["end"] - s["start"]
        wav = f"mv_{i:02d}.wav"
        sh("ffmpeg", "-nostdin", "-v", "error", "-ss", f"{s['start']:.6f}", "-t", f"{d:.6f}",
           "-i", args.audio, "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2",
           "-map_metadata", "-1", f"{COMFY_INPUT}/{wav}", "-y")
        wf, meta = inject.build_a2v_workflow(
            s.get("prompt") or scene_prompt(args.prompt, s["scene"]),
            audio_filename=wav, audio_seconds=d,
            size=args.size, seed=args.seed + i, negative_prompt=args.negative)
        wf[inject.N_LOADAUD]["inputs"]["audio"] = wav

        # Generate AT LEAST the frames the audio window needs, so the trim below can always
        # hit `want` exactly. inject.snap_frames() rounds to the NEAREST 8n+1, which sometimes
        # rounds DOWN (6.13s wants 147 frames, snap gives 145) — you cannot trim frames back
        # into existence, so short clips leave a residual -0.25s drift across the track.
        # Ceil to the next valid 8n+1 instead, then trim down to exact.
        want_frames = int(round((s["end"] - s["start"]) * FPS))
        gen_frames = max(9, -(-(want_frames - 1) // 8) * 8 + 1)   # ceil to 8n+1
        wf[inject.N_FRAMES]["inputs"]["value"] = gen_frames
        meta["video_frames"] = gen_frames
        for nid, v in wf.items():
            if v["class_type"] == "SaveVideo":
                v["inputs"]["filename_prefix"] = f"mv_seg_{i:02d}"
        t0 = time.time()
        pid = submit(wf)
        h = wait(pid)
        if not h or h.get("status", {}).get("status_str") != "success":
            print(f"    seg{i:02d} FAILED"); return 1
        got = None
        for _, o in (h.get("outputs") or {}).items():
            for k in ("images", "videos", "gifs"):
                for x in o.get(k, []) or []:
                    if str(x.get("filename", "")).endswith(".mp4"):
                        got = os.path.join(COMFY_OUT, x.get("subfolder") or "", x["filename"])
        if not got or not os.path.exists(got):
            print(f"    seg{i:02d} no mp4 in outputs"); return 1

        # TRIM to the exact frame count the audio window needs.
        # snap_frames() rounds the length to 8n+1 (LTX's latent constraint), which makes each
        # clip ~0.13s LONGER than its audio window. Concatenated untrimmed, that error
        # ACCUMULATES: measured +1.27s over 25 segments — more than half a bar. The picture
        # slides off the beat as the song plays and every bar-aligned cut is wasted.
        want = int(round((s["end"] - s["start"]) * FPS))
        trimmed = f"{args.out}/seg_{i:02d}_trim.mp4"
        sh("ffmpeg", "-nostdin", "-v", "error", "-i", got, "-frames:v", str(want),
           "-an", "-c:v", "copy", trimmed, "-y")
        have = sh("ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
                  "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", trimmed).stdout.strip()
        mp4s.append(trimmed)
        print(f"    seg{i:02d} scene{s['scene']} gen {meta['video_frames']}f -> trim {have}f "
              f"(want {want}) {time.time()-t0:.0f}s")

    # concat (video only — LTX's re-rendered audio is discarded) then mux the ORIGINAL
    lst = f"{args.out}/concat.txt"
    open(lst, "w").write("".join(f"file '{p}'\n" for p in mp4s))
    sh("ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
       "-an", "-c:v", "copy", f"{args.out}/video_only.mp4", "-y")
    span = sum(s["end"] - s["start"] for s in segs)
    sh("ffmpeg", "-nostdin", "-v", "error", "-ss", f"{segs[0]['start']:.6f}", "-t", f"{span:.6f}",
       "-i", args.audio, "-c:a", "pcm_s16le", f"{args.out}/original_audio.wav", "-y")
    out = f"{args.out}/MUSIC_VIDEO.mp4"
    sh("ffmpeg", "-nostdin", "-v", "error", "-i", f"{args.out}/video_only.mp4",
       "-i", f"{args.out}/original_audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
       "-shortest", out, "-y")
    json.dump({"analysis": a, "segments": segs, "mp4s": mp4s}, open(f"{args.out}/plan.json", "w"), indent=2)
    dur = sh("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out).stdout.strip()
    print(f"\n  ==> {out}  ({dur}s, {len(segs)} cuts, {len(set(s['scene'] for s in segs))} scenes)")
    print(f"      audio = YOUR ORIGINAL (LTX's re-render discarded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
