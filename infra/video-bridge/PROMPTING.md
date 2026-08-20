# LTX-2.3 generation best practices — prompting + settings audit (2026-07-18)

Synthesis of official Lightricks guidance, community A/B findings, and our own measured
results (music-video, ltx2-lora, ltx2-nvfp4). The ground truth for prompt style is the
**Gemma enhancer system prompts shipped in the LTX-2 checkout** — they define the caption
distribution the model was trained toward:

```
~/dev/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt
~/dev/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/gemma_i2v_system_prompt.txt
```

Each ends with a full worked example — the canonical "ideal LTX-2 prompt."

## Prompt spec

**Form.** One single flowing paragraph, ≤200 words (enhancer caps ~150). Chronological,
present-progressive ("is walking", "speaking"), temporal connectors ("as", "then", "while").
Start directly with the action — no "The scene opens with…", no timestamps, no scene cuts,
no markdown. Optional trained-for prefix: `Style: <style>, ` (default cinematic-realistic).

**Order.** Main action in one sentence → specific movements/gestures → precise
character/object appearance → environment → camera → lighting/color → sudden changes.

**Restrained language is on-distribution.** "Red dress" not "vibrant red"; "soft overhead
light" not "blinding light". Quality tags and decorative adjectives ("epic", "stunning",
"masterpiece") measurably hurt. Long-and-concrete wins, long-and-flowery loses. Scale
density to duration: ~one main action per 2–3 s of clip; 10 s ≈ 6–8 sentences.

**Camera.** Always specify — unspecified camera = default drift + random angle changes.
Vocabulary: static frame, slow pan, tracking shot, dolly in/out, push in / pull back,
jib up/down, circles around (orbit), handheld, overhead, wide/medium/close framing.
Specify the end-state ("…settling on a close-up of her face"). The official enhancer never
invents camera motion — with `enhance_prompt` on, camera moves must be in the raw prompt.

**Audio.** Weave the soundscape chronologically alongside the actions — never appended at
the end. Specific, not vague: "soft footsteps on tile", not "ambient sound is present".
Dialogue = exact words in quotes + voice character + volume ("says in an excited voice:
'…'"); specify language/accent. Known weakness (official HF caveat): audio without speech
(pure ambience/music) is lower quality. Audio always runs higher CFG than video (7.0 vs 3.0).

**I2V: describe only the changes.** Don't re-describe the image — "inaccurate descriptions
may cause scene cuts" (official). Empty motion prompt → idle breathing-level motion. The
old ComfyUI-LTXVideo enhancer's image-first advice is LTX-Video-era, superseded for 2.x.

**Only visual + auditory.** No smells/feelings; internal states become observable cues
("her jaw tightens and she looks away", not "she feels sad"). POV shots exclude the POV
subject.

## Settings reference

```
                   ours (bridge, distilled fp4)   official
steps              8 (ManualSigmas)               8 distilled / 30 dev-2.3 / 15 HQ
video cfg          3.0 (MultimodalGuider)         3.0 dev; CFG=1 distilled
audio cfg          7.0                            7.0
stg                1.0, skip_blocks 28            1.0 block 28 dev; OFF for HQ + distilled
rescale            0.9 video / 0.7 audio          0.7 dev; 0.45/1.0 HQ
fps (t2v)          30                             24 (2.3 also 48/50; ~2x gen time)
resolution (t2v)   1280x704 single-stage          768x512 stage-1 -> 2x upsample (dev)
frames             8n+1                           8n+1
max clip           ~8 s (193f @ 24fps)            121f @ 24fps default
```

Hard VAE constraints (verified locally; violations pad with −1 → corrupted latents):
frame count `% 8 == 1`; width/height multiples of **32** (64 for two-stage). Latent math
in `inject.py`: `lat=(px−1)/8+1`. For audio-sync work, **ceil** to the next 8n+1 then trim —
nearest-rounding drifted +1.27 s over 25 segments (music-video).

Distilled sigmas (ours = official): `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375,
0.725, 0.421875, 0.0`; stage-2 refiner = last 4.

## Failure modes → fixes

```
symptom              cause / fix
slow motion          model's consistency-safe default. Motion cue in EVERY sentence
                     (subject + environment + camera); LOWER cfg toward 2–3 before
                     rewriting. Raising fps does not fix it.
static video         prompt reads like a still image — add motion verbs + camera move
morphing             overloaded simultaneous actions — one continuous scene, sequential
                     micro-actions; avoid multi-body physics + readable text/logos
i2v scene cuts       prompt contradicts the image — describe changes only
flicker              cfg ~4 + steps ~40 + fixed seed; on fp4, A/B vs bf16 before
                     blaming the prompt (may be the quant)
unfinished cam move  missing end-state — say where the motion concludes
audio desync         usually input preprocessing (normalization, leading silence), not
                     the model
```

Locally proven landmines (see `experiments/ltx2-lora/RESEARCH.md`, `experiments/music-video/`):

- **Subject LoRAs collapse motion** (near-static at weight 1.0, overrides action prompts).
  Style LoRAs transfer well — do style first; vendor answer for motion+identity = IC-LoRA.
- **Dev-trained LoRA on distilled = degraded motion** (train-on-distilled, infer-on-distilled).
- **a2v conditions on vibe, not beats** — onset correlation ≈ +0.04. All rhythm must come
  from the edit (cut on bars). The model re-renders audio (SetAudioRefTokens conditions,
  doesn't freeze; corr +0.04–0.05) — bridge muxes the original track back in `/content`.
- **Literal music-video imagery reads as stock-footage cheese** — abstract texture /
  colour / material / light / motion prompts (<22 words, no people/faces/text) land better
  for music-driven work. See `ABSTRACT_SYS` in `experiments/music-video/proto_music_video.py`.
- **cu128 trap**: NVFP4 is silently ~2× slower than fp8 without cu130 torch.
- **Full-decode branch artifacts on fp4** — distilled decode only (our graphs comply).
- Bare `device="cuda"` lands on packed GPU0 — always `CUDA_VISIBLE_DEVICES=1`.

## Bridge audit findings (divergences, open items)

1. **Our negative prompt is likely a no-op.** `inject.py` DEFAULT_NEG ("pc game, console
   game, …") only bites where CFG > 1; the distilled branch runs the fixed sigma schedule
   with no negative-conditioning pass. Verify which guider path is live in the graph; if
   guidance IS active, adopt Lightricks' full default (`ltx-pipelines/utils/constants.py`
   `DEFAULT_NEGATIVE_PROMPT` — covers anatomy, flicker, lip-sync, audio artifacts).
2. **T2V fps default 30 vs official 24.** a2v was already fixed to 24 after the desync
   bug; t2v still off-distribution.
3. **Single-stage 1280×704 vs official two-stage** (low-res stage 1 → 2× spatial upsample
   with `ltx-2.3-22b-distilled-lora-384` refiner). Our choice is a speed tradeoff; the
   two-stage path is the documented quality lever if clips look soft.
4. **Highest-leverage: wire prompt enhancement into the bridge.** Reuse the Gemma t2v/i2v
   system prompts verbatim as the rewrite prompt on a local LLM lane. Gotcha from
   music-video: `enable_thinking=False` is load-bearing (thinking lane never terminates
   on this task) + loud fallback on empty content.

## Sources

Official: github.com/Lightricks/LTX-2 README §Prompting · huggingface.co/Lightricks/LTX-2{,.3}
cards · ltx.io prompting/adherence/slow-motion guides · docs.comfy.org LTX-2 tutorial ·
`~/dev/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` (verbatim defaults).
Community (hands-on A/Bs): apatero.com, cosmo-edge.com, film.fun, ltxworkflow.com,
genaintel.com. Local measurements: `experiments/music-video/README.md`,
`experiments/ltx2-lora/{RESEARCH,EXEMPLARS}.md`, `experiments/ltx2-nvfp4/README.md`.
