# mood-to-palette — drive orb expression from internal state

**Pipe**: visual.
**Status**: planned (Phase 4).

## Problem

ORBIS exposes `apply_palette(name)` and `adjust_param(key, value)`
as tools the *user* can invoke ("be warmer", "set palette to
ocean"). The orb itself doesn't drive its own appearance from its
own internal state.

Result: the orb's visible form is static during a conversation —
same palette during a tense exchange as during a warm one. Misses
the whole point of having a visible companion.

## Target behavior

A continuous mapping from internal state → visual parameters:

- Audio-tags mood (valence, arousal, mood_class)
- LLM-inferred mood (from response sentiment)
- Personality drift (current `personality_axes` snapshot)
- Speaking state (listening / preparing / speaking-quiet /
  speaking-loud / thinking)

→

- Palette interpolation (current → target over ~2 seconds)
- Shader parameter adjustments (energy, warmth, focus, motion)
- Optional special states (`[laughing]`-paired animation,
  `[whispered]` reduced amplitude, etc.)

## Why ORBIS needs it specifically

The orb is the *visible expressive form*. Half the companion
experience comes from it being alive. Static-during-conversation is
the failure mode.

## Candidate architectures

1. **Pure rule engine** — hand-tuned mappings from mood → palette
   coordinates + smooth interpolation. Zero ML. Probably the right v0.
2. **Small classifier** — input mood vector, output palette name
   (categorical) + intensity (regression). Only useful if hand-rules
   fail to cover the state space.
3. **Embedding-similarity** — embed (mood description, palette
   metadata), retrieve nearest palette. Generalizes to user-added
   palettes.

V0 should be pure-rule. Replace with ML if/when the rule engine
gets unwieldy.

## What I'd actually build first

A 100-line `MoodPaletteDriver` Pipecat frame processor that:

1. Subscribes to the audio-tags `mood_class` + `valence` + `arousal`
   stream.
2. Maintains a target palette (from a hand-curated mapping table).
3. Smoothly interpolates current → target on every animation frame.
4. Emits `adjust_param` calls to the orb shader.

That's not even research; it's a wiring exercise. The research
question only opens up once we have data on whether the hand-rules
feel right.

## Eval plan

1. **Demo + subjective UX** — does the orb feel alive in a way it
   doesn't today? Recorded video of a 2-minute conversation, side-by-
   side with current ORBIS. Vibes test.
2. **Mood-palette correlation** — log mood predictions + palette
   choices over a long session. Does the model produce the
   expected co-variation, or is it stuck in a corner?
3. **Stability** — palette doesn't oscillate jitterily as mood
   estimates flicker between turns.

## Deliverables

- ORBIS integration: `voice/visual/mood_driver.py` — Pipecat frame
  processor that wires audio-tags → orb shader params.
- Default mood-palette mapping table at `config/visual/mood_palette.yaml`
  (user can override).
- Blog post: "Making the orb alive — closing the perception-to-
  expression loop."

## Open questions

- Per-persona palette mapping (each starter orb has its own
  emotional vocabulary)?
- Should the user be able to *teach* the orb a custom mood-to-
  palette mapping ("when I'm tired, fade to indigo")?
- Visual feedback to the user — should the orb's palette also
  signal its *understanding* of the user's mood (e.g., subtle warm
  pulse when audio-tags says "user is sad")?

## Dependencies

- Audio-tags shipping as a real-time signal source (Phase 1).
- ORBIS visual layer is the consumer — no API change needed,
  just call `adjust_param` from the new driver.
