# We handicapped a music model on purpose. The handicap won.

> Code: [`experiments/acestep-lora/`](https://github.com/protoLabsAI/lab/tree/main/experiments/acestep-lora)
> (RESULTS.md has the full numbers). Nodes, workflows and probes live in our ACE-Step fork.
> Model: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) turbo, 2× RTX PRO 6000 Blackwell.

The idea was a text bottleneck. Take a real track, run it through our captioner and
transcriber, throw the audio away, and hand *only the words* to the music model. No embeddings,
no audio conditioning, no latent init. Whatever survives is whatever the system could **say**
about the song.

It was supposed to be a lossy experiment. The point was to measure the loss.

Then we ran the control arm — ACE-Step's own `cover` task, which gets the actual audio — and the
text bottleneck **won**. Not "held its own." Won, by ear and by metric.

Here's how that happened, plus the three pieces of our own documentation we had to kill along the
way, and the six times today something reported success while doing nothing at all.

---

## The setup

We have 428 tracks labeled by `acestep-captioner` (11B, Qwen2.5-Omni) and `acestep-transcriber`
(11B) — prose captions, sectioned lyrics, measured BPM. That corpus exists for LoRA training, but
it doubles as ground truth: for any track we can compare a live parse against a known-good one.

The round trip, once it worked:

```
LoadAudio ─┬→ Captioner(11B)   → caption  ──→ tags
           ├→ Transcriber(11B) → lyrics   ──→ lyrics
           │                   → language ──→ language
           └→ AudioMeta        → bpm      ──→ bpm
                               → keyscale ──→ keyscale
                               → duration ──→ duration + latent length
```

Audio enters the two parsers and stops. Everything downstream is a string. The bottleneck is
enforced by the graph's type system, not by discipline.

---

## Lore #1: "prose confuses the model, so we force tag output"

That sentence lived in our own `proto_label.py` docstring. It's why our labeler forced
comma-separated tags, and why our downstream agent was taught to emit
`"indie folk, acoustic guitar, piano, warm, 95 BPM, female vocal"` instead of sentences.

We A/B'd it: 3 tracks × 2 formats × 2 seeds, every input identical except the caption field,
same shared latent. Verified programmatically that the two encoder nodes differed in **exactly
one field**.

**Prose won across all three genres at both seeds.**

The mechanism is embarrassing in hindsight. `acestep-captioner` is, per its own model card, *"the
annotation model used by ACE-Step v1.5 for training data labeling."* It emits prose. **Every
caption the model saw during training was prose.** The tags rule was v1-era community lore we
inherited, hardened into a docstring, and then propagated into a downstream product.

A likely reason it keeps getting re-derived: ComfyUI's node is `TextEncodeAceStepAudio1.5` and
the field is called `tags`. A field name is not evidence.

## Lore #2: "understand_music hallucinated captions"

Same docstring, next line. ACE-Step ships a cheap native path — `understand_music()` — that reads
audio and returns caption, lyrics, BPM, key. It runs on the resident 1.7B LM. Free, co-resident,
no 22 GB tenant.

Having just refuted the docstring's first claim, we assumed this one was stale too.

It wasn't. It's true, and worse than "hallucinated" implies:

| track | lyric similarity vs truth |
|---|---:|
| Big K.R.I.T. — *Higher Calling* | 0.037 |
| J. Cole — *c l o s e* | 0.046 |
| Bas — *Passport Bros* | 0.030 |

For *Higher Calling* it invented a Michael Jackson pastiche (*"Billie Jean, you know she robbin'
dance"*). For *Passport Bros* — an English-language afrobeats track — it produced **an entire
Finnish verse**, tagged `language=fi`, genre *"Funk, electronic, Finnish rap"*.

The captions read fluent and plausible. That's what makes them dangerous.

**Why:** `understand_music` reads **5 Hz discrete codes** — 823 codes for a 165-second track.
Five tokens per second cannot represent ~390 words of sung lyrics. The information isn't in the
representation, so the LM confabulates something that scans. No temperature setting recovers data
that was never there. The 11B captioner reads the raw waveform.

**One docstring, two claims, opposite verdicts.** If we'd trusted it we'd have shipped tags. If
we'd dismissed it as stale we'd have built the round trip on a fabricator. Both needed testing.

## Lore #3: the caption must end "N BPM."

This one we discovered by breaking it.

The first full round-trip render sounded, verbatim, *"busted af."* Same graph, same models, same
seed as the runs that sounded fine. The only difference: the caption had no BPM suffix.

```
good:    "...reminiscent of late '90s R&B. 103 BPM."
busted:  "...before returning to the main rap flow."
fixed:   "...before returning to the main rap flow. 99 BPM."   → "banger. ship it"
```

Every one of the 428 training captions ends that way — the labeler literally does
`cap_text + f" {bpm} BPM."`. A caption without it is out of distribution.

The way we broke it is the instructive part. We'd found that ~31% of our corpus BPMs are
**doubled** (librosa octave errors — 172/161/152 clustering, which halve to 86/80.5/76). So when
building the caption node we defaulted the BPM suffix **off**, reasoning that stamping a possibly
wrong tempo into captions would re-seed the bug.

Locally sound. Globally wrong. Switching it off didn't avoid a bad value — it made *every* caption
out of distribution to dodge a value that *might* be wrong. The answer was always to feed it the
**corrected** BPM.

---

## The control arm: text vs the model's own audio path

`cover` is ACE-Step's native audio-conditioned task. Same caption, same lyrics, same bpm/key/
duration/seed — but it also gets the source audio. The delta is what the bottleneck costs.

Five renders. Only the audio path varies.

```
                         chroma/frame   onset corr      (vs source; self-ceiling 1.000)
A1 text-mediated              0.749        0.033        ← no audio crosses at all
A2 cover(thinking=True)       0.700        0.020
A3 cover thinkFalse+src       0.727        0.037
A4 cover codes-from-audio     0.715        0.070
```

**A1 wins.** By metric, and by ear — which is the verdict that counts, because DSP couldn't
distinguish the busted no-BPM render from the good one either (it measured exact duration, tempo
within 3, and sat spectrally adjacent to the good take).

A4 is the real deal — the Gradio *Analyze → Remix* path, confirmed consuming the source's codes
via a `"Using precomputed LM hints"` log line. It still lost.

**Why, and it's the same answer as Lore #2:** `cover` conditions on those same 5 Hz semantic
codes. 823 codes for a 165-second track. We already proved that channel can't carry lyrics; it
evidently can't carry melody either. Meanwhile ACE-Step 1.5 is a **text-conditioned model** — its
DiT was trained on prose captions. Text is the native, high-bandwidth channel. The codes are a
narrow auxiliary hint.

**`cover` isn't an audio-conditioned reference. It's a narrower bottleneck.**

So describe-and-regenerate gives up nothing versus the model's own audio-to-audio path. The
handicap wasn't a handicap.

---

## The through-line

Every finding reduces to one rule:

```
prose > tags               — the captioner that labeled 1.5's data writes prose
caption must end "N BPM."  — all 428 training captions do
text > 5Hz codes           — the model is text-conditioned; codes are auxiliary
```

**Match the training distribution.** Reasoning from first principles about what *should*
condition better went 0 for 3. Every losing intuition was well-argued, and every one was beaten
by "what did the model actually see."

---

## Six things that reported success while doing nothing

This is the other half of the day, and honestly it might be the more useful half.

1. **`thinking=True` silently voids `cover`.** The LM generates codes *from text*; codes take
   precedence; `src_audio` is dropped. Our first control arm ran this way — it was text2music
   wearing a cover label. Tell: `encoder_time_cost: 0.012s`. You cannot encode 169 seconds of
   audio in 12 milliseconds.
2. **A `STRING → COMBO` link is rejected silently.** ComfyUI validates each output
   independently, drops the failing branch with one log line, and `/prompt` **still returns 200
   with a prompt_id**. The run reports `status: success` having never executed the generate leg.
   A queued prompt is not proof the graph ran.
3. **`reference_audio` doesn't work for `cover`** — despite its docstring saying *"a reference
   audio file for style transfer or cover tasks."* It raises `"Task 'cover' requires source
   audio."`
4. **A third-party `Understand` node** returns hardcoded `"Audio track"` / `120 BPM` / `"C Major"`
   behind a bare `except: pass`.
5. **`ffmpeg` inside a `while read` loop** eats stdin and converts every *other* file. Ours did
   10 of 20 — tracks 01, 03, 05, 07. The alternating pattern is the tell. `-nostdin`.
6. **Our own metrics.** Twice. An autocorrelation-based tempo scorer that structurally favors the
   faster candidate (it "proved" 172 over 86 on a track two independent estimators put at 91).
   And a tempo prior applied unconditionally, which returned ~90 for everything — not detecting
   tempo, just echoing its own assumption.

The unifying shape: **a success signal that isn't**. Exit code 0, HTTP 200, `status: success`, a
plausible number. Each one survives exactly as long as nobody checks the thing it claims to have
done.

---

## What shipped

A round trip that runs in ComfyUI on nodes we own and version — swap the source track and caption,
lyrics, BPM, key, duration and latent length all follow. It sounds good. It beats the audio path.

And a corpus with ~31% doubled BPMs and one album (20/20 tracks) carrying a corrupt trailing frame
that every tolerant decoder silently swallowed. Both found only because something else broke first.
