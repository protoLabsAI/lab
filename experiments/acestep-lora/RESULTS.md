# ACE-Step 1.5 — caption format: PROSE beats TAGS (2026-07-15)

**Finding: for ACE-Step 1.5, prose captions beat structured tag captions. Our own "tags, not
prose" guidance is REFUTED.** It was v1-era/community lore that we inherited, encoded in a
`proto_label.py` docstring, and then propagated downstream into protoDirector's prompt craft.

## The claim we killed

`proto_label.py` docstring (now annotated as superseded):

> Per the ACE-Step prompting guide: captions must be structured tags (3-7), lead with genre,
> explicit BPM, specific instruments — poetic prose confuses the model. So we force tag output.

That sentence is why `proto_label.py`'s `CAP_PROMPT` forces *"ONLY a comma-separated list of 6-9
tags ... No sentences"*, and why protoDirector PR #23 taught its agent to emit comma-separated
tags (protoLab#22).

## Method

ComfyUI workflow `ACE-roundtrip-AB-prose-vs-tags-n12.json` (in `user/default/workflows/`).

- **3 tracks x 2 formats x 2 seeds = 12 generations**, one Run.
- Sources: taste-corpus tracks that already carry known-good sidecars (428 labeled, 160 with
  audio) written by `proto_label_taste.py` = **acestep-captioner (11B) + acestep-transcriber**.
- **PROSE arm** = the acestep-captioner caption verbatim.
- **TAGS arm** = a generous hand reduction of that same prose, per the documented tag rule
  (genre-first, specific instruments, explicit BPM).
- Controlled: verified programmatically that per track+seed the two encoder nodes differ in
  **exactly one field (`tags`)** — lyrics, seed, bpm, duration, keyscale, timesignature,
  language, sampler settings all byte-identical, and **both branches share the same empty latent**.
- Turbo config copied from the known-working `ACE-Step-1.5_music.json`, not guessed:
  `ModelSamplingAuraFlow` shift=3, KSampler 8 steps, cfg=1, euler/simple.

| track | genre | bpm | dur | key (est) | prose chars | tags chars |
|---|---|---:|---:|---|---:|---:|
| Big K.R.I.T. — Higher Calling | neo-soul | 103 | 234s | A# minor (0.53) | 471 | 208 |
| Bas — Passport Bros | afrobeats | 152 | 165s | A major (0.38) | 436 | 218 |
| J. Cole — c l o s e | boom-bap | 172 | 169s | A minor (0.51) | 382 | 199 |

Seeds: 42, 1337.

## Result

**Prose won across all three genres at both seeds.** First observed at n=1 (neo-soul, seed 42),
then confirmed at n=12. Judgment was subjective listening by Josh, not a metric — see caveats.

## Why (the mechanism, which is why we believe it)

`acestep-captioner` is, per its HF card, *"the annotation model used by ACE-Step v1.5 for training
data labeling"* — and it emits **prose**. So prose is not merely tolerated by 1.5, it is what the
model saw in training. **Prose is in-distribution.** The tags rule most likely predates 1.5.

Corroborating: the core ComfyUI node `TextEncodeAceStepAudio1.5` names the field `tags`, which is
probably where the lore keeps getting re-derived. The field name is not evidence about format.

## Caveats — read these before citing this

- **Subjective judgment, n=12, one listener, one corpus.** No metric, no blind protocol. Josh knew
  which arm was which when listening (filenames encode it). Directionally strong and consistent,
  but this is not a scored eval.
- **The TAGS arm was a hand reduction, not `proto_label.py`'s live output.** So "prose > tags"
  is proven; "prose > our actual Qwen2-Audio tag pipeline" is *inferred*. The real head-to-head
  needs Qwen2-Audio-7B (~16 GB) on a GPU1 with ~12 GB free — blocked on the residency decision.
- **Corpus is one family** (hip-hop / soul / afrobeats — Big K.R.I.T, Smino, Bas, J. Cole,
  EarthGang, OutKast, Dreamville). Not tested on rock, electronic, classical, metal.
- **Keys are chroma estimates, not ground truth** — the captioner never writes a keyscale.
  Afrobeats key is the weakest (corr 0.38). Same value in both arms, so no A/B bias.

## Consequences

1. **`proto_label.py` annotated as SUPERSEDED** with the refutation inline, so the lore can't be
   re-derived from it. Tag path kept — it's still the implementation if tags are ever wanted.
2. **`proto_label_taste.py` (acestep-captioner -> prose) is the labeling path.** Already is.
3. **protoDirector's prompt craft (protoLab#22, PR #23) is optimizing for the losing format.**
   Its agent emits comma-separated tags. Should emit prose. This is *less* work than what shipped.
4. **The round trip needs no prose->tags reduction step**, which was the open design seam.

---

# understand_music() FABRICATES — the cheap parse path is dead (2026-07-15)

**Second finding, same day, opposite direction.** The *other* claim in that same `proto_label.py`
docstring — "ACE-Step's native understand_music() hallucinated captions" — is **CONFIRMED**.

**Lesson: the docstring was right about one claim and wrong about the other. Both needed testing.
A stale doc is not uniformly wrong.**

## Method

`proto_understand_ab.py` (in the protoLab fork). Same 3 tracks as the prose-vs-tags A/B, so the
two experiments are comparable. Ground truth = the known-good sidecars (acestep-captioner 11B +
acestep-transcriber). Under test = `dit.convert_src_audio_to_codes(audio)` ->
`understand_music(llm, audio_codes)` — exactly what the stock Gradio "Analyze Source Audio"
button does. Cheap arm: resident 1.7B 5Hz LM, no extra model, no extra VRAM.

## Result — it does not drift, it fabricates

| track | lyric similarity | bpm (und / librosa) | lang (und / truth) |
|---|---:|---|---|
| neo-soul — Higher Calling | **0.037** | 79 / 103 | en / en |
| boom-bap — c l o s e | **0.046** | 91 / 172 | en / en |
| afrobeats — Passport Bros | **0.030** | 115 / 152 | **fi** / en |

- **Higher Calling** -> invented a Michael Jackson pastiche: *"Billie Jean, you know she robbin'
  dance / 1942, I want another dance"*. Truth: *"We got a higher calling / This life / I treasure
  this high"*.
- **Passport Bros** -> invented an entire **FINNISH verse** (*"Kuuntelen ja sun takia riitatut
  niin..."*), `language=fi`, genre *"Funk, electronic, Finnish rap"*. It's an English-language
  Bas afrobeats track.
- Captions read **fluent and plausible**. That is what makes them dangerous — same failure class
  as the llm_judge 0.5-fallback and the AceMusic `Understand` stub: a dead path that looks alive.
- Only the boom-bap *genre label* landed. Its narrative was still wrong.

## Why — and why no tuning fixes it

`understand_music` reads **5 Hz discrete FSQ codes**: **823 codes for a 165 s track**. Five tokens
per second cannot represent ~390 words of sung lyrics — the phonetic content is not in the
representation, so the LM confabulates something plausible. The 11B captioner reads the **raw
waveform**. This is an information-theoretic ceiling, not a temperature/decoding setting.

## Consequence — this answers the residency question

**The cheap co-resident parse path is not viable.** The round trip requires the 11B
captioner + transcriber (~22 GB each, or sequential). **The round trip is not free** — it needs a
real GPU1 slot. That is a hard input to the residency decision, not a preference.

## BPM caveat — our own "truth" is suspect

understand/librosa ratios: 1.30 / **1.89** / 1.32. The 1.89 is a classic librosa **octave error** —
understand's **91 BPM is probably MORE correct than our librosa 172** for a boom-bap track (the
genre sits ~85-95). Trust neither column. **Our sidecar BPMs need an audit** — 428 tracks were
labeled with librosa beat tracking and some are likely double/half.

## ComfyUI cannot do this leg at all (checked, not assumed)

Deliberately NOT run through ComfyUI, because both parse paths there are broken:
- **No AUDIO -> audio_codes node exists.** `AceStepAudioCodesUnderstand` needs `audio_codes: LIST`;
  the only LIST producers are a file-loader, a text-conditioning splitter, and code mixers.
  `ScromfyAceStepVAEEncode` emits a **LATENT** (25 Hz continuous), not 5 Hz discrete codes.
  `convert_src_audio_to_codes` is never exposed as a node.
- **`AceStepLLMLoader` is unusable** — feeds `AutoModelForCausalLM.from_pretrained()` a ComfyUI
  `models/checkpoints/` dir with no `config.json`. So the Kaola captioner/transcriber nodes
  (which DO take AUDIO) can't be loaded either.

scromfyUI-AceStep is 13 stars, last commit 2026-03-27. This is the adopt-maintenance cost showing.

---

# BPM audit: ~31% of the corpus is labeled at DOUBLE tempo (2026-07-15)

Follow-on from the BPM caveat above. `proto_bpm_audit.py` (protoLab fork), 160 tracks with audio.

**49 / 160 flagged (31%).** Not scattered — they cluster on exactly **172 / 161 / 152**, which
halve to **86 / 80.5 / 76**: the canonical hip-hop range. `sidecar >= 140` counts **51**, of which
**49 are flagged** — essentially every fast-labeled track in a hip-hop/soul corpus is a doubling
candidate.

| track | sidecar | librosa free | prior90 | ratio |
|---|---:|---:|---:|---:|
| OutKast — West Savannah / Y'all Scared | 172 | 172.3 | 86.1 | **2.00** |
| Big K.R.I.T. — Precious Metal | 172 | 172.3 | 86.1 | **2.00** |
| J. Cole — c l o s e | 172 | 172.3 | 92.3 | 1.86 |
| J. Cole — t h e . c l i m b . b a c k | 161 | 161.5 | 80.7 | 1.99 |
| Bas — Black Jedi / 179 Deli | 152 | 152.0 | 76.0 | **2.00** |
| EarthGang — Top Down | 152 | 152.0 | 76.0 | **2.00** |

Corroboration on `c l o s e`: sidecar 172, but a 90-BPM prior says 92.3 **and** understand_music
independently said 91. Two unrelated estimators agree on ~91.

## Blast radius — it's in TWO places per track

The error is in the `bpm` field **and inside the prose caption itself** (captions end "...172
BPM."), both from the same librosa call. So:
- **LoRA training** has been learning "172 BPM" against 86 BPM audio for ~a third of the corpus.
- **Every generation** feeds `bpm` to `TextEncodeAceStepAudio1.5` — our own prose-vs-tags A/B told
  ACE-Step 172 for a track that is probably 86. (Both arms got the same wrong value, so that
  comparison still stands.)

## Method + the metric I had to throw away

Flag = the unpriored librosa estimator and a 90-BPM-prior estimator land on **different metrical
levels** (ratio ~2x either way). Deliberately conservative: **flags, does not auto-fix.**

**A metric I built was biased and the sanity check caught it.** I scored candidates by
onset-envelope autocorrelation. On the known-bad `c l o s e`: ac(172)=0.622 > ac(86)=0.549 — it
"supported" the wrong answer. Autocorrelation **decays with lag**, so the faster candidate (shorter
lag) wins almost unconditionally; it would have voted "faster" on every track and laundered the
error as evidence. Demoted to informational-only with a warning in the docstring. The principled
arbiter is a tempo **prior**, not raw AC. Same failure class as the llm_judge 0.5-fallback: a
number that looks like evidence and isn't.

## Caveats

- **FLAG != wrong.** Afrobeats/trap genuinely run fast, and hip-hop tempo notation is
  conventionally ambiguous (a "140 trap" track is a 70 half-time feel). Not proven per-track.
- Sidecar BPMs don't always reproduce under re-analysis (e.g. Blue Moon: sidecar 185, free 117.5)
  because `proto_label_taste.py` used a different window than this audit. So some sidecars are
  unreproducible, not merely doubled.
- 160 of 428 sidecars have audio present; the other 268 are unaudited.

## Recommendation

Do NOT bulk-rewrite. Spot-check ~5 flagged tracks by ear against a metronome, and if the halving
holds, re-label the flagged set with a genre-informed prior (`start_bpm=90`) and regenerate the
BPM suffix in the affected captions. Then re-run any LoRA trained on this corpus.

---

# The caption MUST end "N BPM." — omitting it audibly wrecks the render (2026-07-15)

**Third caption finding, and the sharpest.** The round trip's first full render sounded, per
Josh, "busted af". Same graph, same models, same seed. The only difference from the A/B that
sounded good: **the caption had no `N BPM.` suffix.**

    A/B (good):  "...reminiscent of late '90s R&B. 103 BPM."
    busted:      "...before returning to the main rap flow."
    fixed:       "...before returning to the main rap flow. 99 BPM."   -> "banger. ship it"

## Why

Every one of the 428 training sidecars ends that way — `proto_label_taste.py` does literally
`caption_full = cap_text + (f" {bpm} BPM." if bpm else "")`. So **every caption ACE-Step 1.5 was
conditioned on carries the suffix.** A caption without it is out-of-distribution. Same root cause
as prose-vs-tags: match the training distribution; don't reason from first principles about what
"should" work.

## How it happened — the caution worth keeping

`append_bpm` was added as a knob defaulted to **0/off**, reasoning that since ~31% of our librosa
BPMs are doubled, stamping a wrong BPM into captions would re-seed that bug. Locally sound,
globally wrong: switching it off didn't avoid a bad value, it made **every** caption
out-of-distribution. The right move was always to feed it the *corrected* bpm.
**Wired now:** `ProtoAceAudioMeta.bpm -> ProtoAceCaptioner.append_bpm`.

## The diagnosis was wrong first

I blamed the garbled lyrics — the transcriber renders Smino's slurred delivery as *"I'll pick a
couple wild Irish roll, jish"* — and argued that was the text bottleneck being honest rather than
a bug. **Josh: "its not the lyrics, its the bpm issue. it was not a lyric problem before."**
He was right. The A/B used transcribed lyrics too and sounded fine. I over-weighted the garble
because it was *visible*, instead of suspecting the one thing I had actually changed.

DSP couldn't see it either: the busted render measured dur=169.4s (exact), tempo=95.7 (vs 99
requested), flatness=0.026 — spectrally adjacent to the render that sounded good.
**Coherent music that sounds wrong is invisible to these metrics. Ears are the instrument.**

## Status: SHIPPED

Round trip live and validated end-to-end in ComfyUI. Every encoder field derives from the source
(tags, lyrics, bpm, keyscale, duration, language + latent seconds) — nothing hardcoded. Workflows
version-controlled in the fork at `comfy_nodes/workflows/`.

---

# CONTROL ARM: the text bottleneck BEATS native audio-conditioned cover (2026-07-15)

**The experiment's answer, and it inverts the premise.** The round trip was designed as a
deliberate handicap — force the whole track through text, see how much survives. The control arm
was supposed to price that loss. Instead: **there is no loss. Text wins.**

Josh, after listening: *"yeah A1 is the best still."* A1 is the pure text bottleneck.

## Method

Five renders, `/mnt/data/acestep-lora/cover-ab/`. All share the SAME caption (542ch, ends
"99 BPM."), lyrics (367w), bpm=99, keyscale="C major", duration=169.4s, seed=42, 8 steps.
**The only variable is how — or whether — the source audio reaches the model.**

| | path | verdict |
|---|---|---|
| A0 | SOURCE (real Smino track) | reference |
| **A1** | **TEXT BOTTLENECK — no audio crosses** | **BEST (by ear + by metric)** |
| A2 | cover, src_audio, `thinking=True` | src silently ignored (= text2music) |
| A3 | cover, src_audio, `thinking=False` | worse than A1 |
| A4 | cover via `convert_src_audio_to_codes` (the Gradio Analyze→Remix path) | worse than A1 |

Measured (120s window vs source; self-ceiling 1.000):

                         chroma/frame   onset corr
  A1 text-mediated              0.749        0.033     <- LEADS
  A2 cover(thinking=True)       0.700        0.020
  A3 cover thinkFalse+src       0.727        0.037
  A4 cover codes-from-audio     0.715        0.070

Ear and metric agree here — which is worth noting precisely because they disagreed on the BPM
render. A4 genuinely consumed the audio (logs "Using precomputed LM hints"); it still lost.

## Why — the same lesson as everything else today

`cover` conditions on **5Hz semantic codes**: 823 codes for a 165s track. We already PROVED that
representation is too lossy to carry lyrics (understand_music fabricates from it). It is
evidently too lossy to carry melody as well.

Meanwhile **ACE-Step 1.5 is a TEXT-conditioned model.** Its DiT was trained on prose captions
written by acestep-captioner. Text is the native, in-distribution, high-bandwidth channel; the
5Hz codes are a narrow auxiliary hint. So a rich prose caption + full lyrics carries MORE usable
information into this model than the audio's own semantic codes do.

**"cover" is therefore not an audio-conditioned reference at all — it is a DIFFERENT bottleneck,
and a narrower one.**

## The through-line of this whole experiment

Every finding today reduces to one rule: **match the training distribution.**
- prose > tags — because the captioner that labeled 1.5's training data writes prose
- caption must end "N BPM." — because all 428 training captions do
- text > 5Hz codes — because the model is text-conditioned; codes are the auxiliary channel

Reasoning from first principles about what *should* condition better lost three times out of three.

## Gotchas (both silent, see proto_cover_ab.py / proto_cover_probe.py)

* **`thinking=True` VOIDS cover.** The LM generates codes FROM TEXT; codes take precedence and
  `src_audio` is dropped. Documented in `acestep/text2music_src_audio_test.py`. The first control
  arm ran this way and was just text2music wearing a cover label. Tell: `encoder_time_cost=0.012s`
  — you cannot encode 169s of audio in 12ms.
* **`reference_audio` does NOT work for cover**, despite its docstring ("a reference audio file
  for style transfer or cover tasks"). Raises "Task 'cover' requires source audio". Use
  `src_audio` or `audio_codes`.
* The `:8110` adapter does not expose cover. ComfyUI has no native cover either
  (`VAEEncodeAudio` + denoise<1 is a latent remix — a different thing).

## Still open
- Patch `AceStepLLMLoader` to `Qwen2_5OmniForConditionalGeneration` (present in ComfyUI's
  transformers 5.3.0) to unlock the Kaola 11B nodes inside ComfyUI — an alternative to our pack.
- **BLOG.md** — per the operating cycle, a shipped experiment owes a blog draft. This one has a
  genuine, counterintuitive result (describe-and-regenerate beats the model's own audio-to-audio
  path, because the model is text-conditioned) plus three refuted pieces of our own lore.
- **Whether the 20 corrupt Big K.R.I.T. tracks were silently dropped from LoRA preprocessing.**
- **Audit the 428 sidecar BPMs** (49/160 flagged) — now doubly load-bearing, since the BPM rides
  in the caption text as well as the bpm field.
- `keyscale`: chroma estimate, no ground truth. Wild Irish Roses came out C major (0.59) with
  G minor (0.58) a near-tie — a coin flip that the render apparently tolerated.
