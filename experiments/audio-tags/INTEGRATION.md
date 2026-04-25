# ORBIS Integration Sketch — audio-tags

How the trained tag service plugs into ORBIS without modifying its core
architecture. **Sketch only** — no code lives in the ORBIS repo yet.
The audio-tags service is a sibling: ORBIS calls it the same way it
calls Whisper.

## Where it slots in

ORBIS pipeline (per its README):

```
Browser ──WebRTC──▶ Pipecat ─┬─▶ STT (Whisper) ──▶ LLM ──▶ TTS ──▶ Browser
                              │
                              └─▶ audio-tags  (NEW)
                                       │
                                       ▼
                                   tags JSON
                                       │
                       ┌───────────────┼─────────────────┐
                       ▼               ▼                 ▼
                  inject into     write to mood       (optional)
                  LLM context     SQLite table        log to facts
```

The same audio frame that hits Whisper is forwarded to the audio-tags
service. The two run **in parallel** — neither blocks the other. Tags
arrive ~50 ms after Whisper text on a Blackwell GPU; on CPU expect
200-500 ms. Either way, we don't gate the LLM on tags — if tags are
late, the LLM call goes ahead without them and we fold tags into the
*next* turn.

## Pipecat hook point

ORBIS uses Pipecat. There's a clean hook between `transport.input` and
the STT service: a `FrameProcessor` that observes `AudioRawFrame`s,
buffers per-utterance audio, and on `UserStoppedSpeakingFrame` fires
the audio-tags request asynchronously.

Pseudocode (a sibling to `voice/agent/prosody.py`'s
`ProsodyTagStripper` pattern):

```python
class AudioTagsTap(FrameProcessor):
    def __init__(self, service_url: str, mood_writer):
        super().__init__()
        self._buf: list[np.ndarray] = []
        self._url = service_url
        self._mood = mood_writer

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            self._buf.append(np.frombuffer(frame.audio, dtype=np.int16))
        elif isinstance(frame, UserStoppedSpeakingFrame):
            wav = np.concatenate(self._buf).astype(np.float32) / 32768.0
            self._buf.clear()
            asyncio.create_task(self._tag_and_inject(wav))

        await self.push_frame(frame, direction)

    async def _tag_and_inject(self, wav):
        tags = await call_audio_tags(self._url, wav, sr=16000)
        # 1. Write to mood table for short-term mood drift
        if tags.get("mood"):
            self._mood.update(tags["mood"]["valence"], tags["mood"]["arousal"])
        # 2. Stash for the next LLM turn (see ContextInjector below)
        self._latest_tags = tags
```

## Injecting tags into the LLM call

Two viable shapes:

### A. System-message append (clean, recommended)

Append a single line to ORBIS's system prompt for each turn that has
fresh tags:

```
[user_audio] mood=warm valence=0.62 arousal=-0.15 gender=female
volume=normal pitch=medium speaking_speed=normal style=conversational
environment=indoor_quiet snr=38dB
```

ORBIS's persona system prompt already permits adding small annotations
(it works with prosody hints, fillers, etc.). One annotation per turn,
auto-cleared on the next turn so we don't accumulate.

### B. Pre-user-message annotation

Prepend the tag dict to the user's transcribed text:

```
{"audio_context": {"mood": "warm", "valence": 0.62, ...}}\n\n
{transcribed user message}
```

Cleaner separation but slightly worse LLM compliance — the system
prompt route gets respected more reliably.

**Recommendation: A**, low-confidence tags filtered out client-side.
Confidence threshold suggestion: `>= 0.65` for inclusion.

## ORBIS-side wiring (no code yet, just plan)

1. **New env var.** `AUDIO_TAGS_URL` in `.env` / `config/orbis.yaml`,
   default unset (feature off → tap is a no-op).
2. **New file.** `voice/agent/audio_tags.py` — the `AudioTagsTap`
   class, the `ContextInjector` that splices tags into the LLM call.
3. **Pipeline placement.** In `voice/pipeline.py` (or wherever Pipecat
   pipeline is built), insert `AudioTagsTap` before the STT service
   and `ContextInjector` before the LLM service.
4. **Mood writer.** Reuse the existing SQLite `mood` table writer —
   just swap the prosody-derived heuristic with the model's
   valence/arousal output.
5. **Persona impact.** The companion-layer "slow-drift personality"
   axes can react to mood trends (a week of `tense` user audio →
   companion becomes more solicitous). That's a follow-up; v0 just
   feeds the *current* mood into the *current* turn.

## Wire format

Service: `POST /tag` (multipart, or POST raw PCM bytes), returns the
schema in `labels/taxonomy.py::example_output()`. Bumping
`SCHEMA_VERSION` is a contract break — clients pin a version.

Health: `GET /healthz` returns `{status, schema_version, device}`.

Discovery: `GET /schema` returns the full taxonomy so ORBIS can map
class names → user-facing strings without shipping the taxonomy file.

## Latency budget

| Step | Budget | Notes |
|------|------:|-------|
| Audio buffering | — | bounded by VAD turn length |
| HTTP round-trip | ~5 ms | tailnet-local |
| Mel feature extract | ~10 ms | CPU, single-threaded |
| Forward pass (Blackwell) | ~5 ms | 8.3 M params, bf16 |
| Forward pass (CPU)        | ~150 ms | 8.3 M params, fp32 |
| Total (Blackwell)         | ~25 ms |  |
| Total (CPU)               | ~200 ms | still under VAD silence threshold |

Both are well under ORBIS's first-token-to-LLM threshold, so even on
CPU-only ORBIS hosts the tap doesn't add user-perceived latency.

## What we're explicitly NOT doing in v0

- Streaming tags (per-frame). Only emit on `UserStoppedSpeakingFrame`.
- Bidirectional — we don't tag the orb's *own* speech.
- Speaker diarization — single-speaker assumed.
- On-device deployment to the browser (the orb client). The model is
  small enough that we *could* WASM-port it later; for now ORBIS calls
  the service over the tailnet.
