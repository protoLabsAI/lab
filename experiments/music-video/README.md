# Music video from generated audio — bar-aligned a2v segments

`proto_music_video.py --audio track.mp3 --prompt "..."` → `MUSIC_VIDEO.mp4`

    audio → beat/bar grid + structure boundaries
          → bar-aligned segments (≤8.04s, frame-quantized)
          → LTX a2v per segment (scene prompt varies on structure)
          → trim to exact frames → concat → mux the ORIGINAL audio

~25s/segment warm. A 169s track = 25 segments ≈ 10 min. 768×512 @ 24fps.

## The three measurements that dictate the design

1. **LTX makes ~8s clips.** 193 frames @ 24fps = 8.04s. Chopping isn't a choice.
2. **a2v RE-RENDERS the audio** — it does not pass yours through (waveform correlation
   input↔output: **+0.040**; chroma 0.977, so it tracks harmonically while being a different
   signal). LTX's audio is discarded; the original is muxed over the finished cut.
   `infra/video-bridge/inject.py::build_a2v_workflow` documents the same.
3. **a2v is NOT beat-locked.** Video motion vs the original audio's onsets: **r=+0.036**; vs
   LTX's own audio: **r=+0.045**. Both ≈0 — it conditions on vibe/texture, not transients.
   Consequences: muxing the original costs nothing (no beat-lock to lose), and **all rhythm must
   come from the EDIT.** The generator supplies looks; the cut points supply the music. Which is
   how music videos are actually cut — the editor carries the rhythm, not the camera.

## Three bugs worth not repeating

* **Dropped the first 20s.** librosa's first beat lands where the *groove* starts (20.6s on the
  test track — the intro is a beatless vocal sample). Seeding the grid from the first BAR silently
  cut 147s of a 169s song. Anchor at 0, backfill the pre-groove intro on the same bar grid.
* **Cuts weren't on bars — the entire trick was missing.** `end = min(next_bar, start+MAX_SEG_S)`
  overrides the grid whenever the next bar sits past the ceiling; cuts landed on an arbitrary
  clock offset (measured 0.68s / 1.08s off the beat). Fix: cut at the LAST BAR THAT FITS, never
  at start+MAX. Verified 24/24 cuts on bar, worst offset 0.000s.
* **+1.27s cumulative drift.** Frame counts snap to 8n+1 (LTX latent constraint), which rounds
  each clip ~0.13s longer than its audio window — and it ACCUMULATES: by the end the picture is
  half a bar ahead and every bar-aligned cut is wasted. Fix: quantize cut times to the frame grid,
  **ceil** to 8n+1 so gen ≥ want (snap_frames rounds to NEAREST and can round DOWN — you can't
  trim frames back into existence), then trim to exact. Verified 0 frames short.

## Still open

* **flf chaining** — `ltx2-flf.json` exists; chain first/last frame *within* a scene for
  continuity, hard-cut at scene changes. Currently every segment is independent.
* **Reference images per scene.** a2v is image+audio→video; all segments currently share
  `example.png`, so scene variety comes only from the shot-language prompt.
* Section tags from `acestep-transcriber` (`[Intro]/[Verse]/[Chorus]`) carry no timestamps, so
  scenes come from librosa structure segmentation instead. Aligning the two would let a chorus
  reuse its look.
