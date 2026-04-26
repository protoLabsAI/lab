# SenseVoice Audio-Pre Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Whisper + emotion2vec with SenseVoice-Small (one forward pass: STT + 7-class emotion + lang + audio events) and wire emotion output into ORBIS's mood system and LLM context.

**Architecture:** `SenseVoiceSTT` is a new `SegmentedSTTService` backend in `voice/stt.py` that emits `EmotionFrame` + `AudioEventFrame` before `TranscriptionFrame`. `AudioTagsTap` is a new `FrameProcessor` that reads `EmotionFrame`, nudges mood (delta-write, R15-safe), and injects a `[audio]` `SystemFrame` into each turn. audio-tags v5 non-emotion heads (SNR/env/speaking-rate) run inside the tap via a parallel call.

**Tech Stack:** FunASR (`funasr`), SenseVoiceSmall, pipecat FrameProcessor, Python 3.11+, pytest, uv

**Tracking:** ORBIS [issue #66](https://github.com/protoLabsAI/ORBIS/issues/66) — lands after #35 (speaker-gate + audio-tags v5-soft) closes.
**Prerequisite:** #35 PR 1 (speaker-gate) and PR 2 (audio-tags v5-soft) must be merged first.

---

## Files to create

| File | Purpose |
|---|---|
| `/home/ava/dev/ORBIS/agent/frames.py` | New — custom ORBIS frame types (`EmotionFrame`, `AudioEventFrame`) |
| `/home/ava/dev/ORBIS/agent/audio_tags.py` | New — `AudioTagsTap` FrameProcessor |
| `/home/ava/dev/ORBIS/tests/test_frames.py` | New — unit tests for frame dataclasses |
| `/home/ava/dev/ORBIS/tests/test_stt.py` | New — unit tests for `make_stt()` routing + `SenseVoiceSTT` |
| `/home/ava/dev/ORBIS/tests/test_audio_tags_tap.py` | New — unit tests for `AudioTagsTap` |
| `/home/ava/dev/ORBIS/tests/test_integration_smoke.py` | New — end-to-end frame ordering test |

## Files to modify

| File | What changes |
|---|---|
| `/home/ava/dev/ORBIS/pyproject.toml` | Add `funasr` dependency + `pytest-asyncio` to test extra + `asyncio_mode = "auto"` |
| `/home/ava/dev/ORBIS/voice/stt.py` | Add `SenseVoiceSTT` class, extend `make_stt()`, extend `prewarm()` |
| `/home/ava/dev/ORBIS/app.py` | Add `audio_context_block()`, add to `_effective_prompt()`, wire `AudioTagsTap` into pipeline |
| `/home/ava/dev/ORBIS/tests/test_neglect.py` | Add R15 interaction test |

---

## Task 1: Install FunASR dependency

**Files:**
- Modify: `/home/ava/dev/ORBIS/pyproject.toml`

- [ ] **Step 1: Add funasr to dependencies**

In `pyproject.toml`, add to the `[project]` `dependencies` list:
```toml
"funasr>=1.1",
```
Also add to `[project.optional-dependencies]` `test` extra:
```toml
"pytest-asyncio",
```
Also add to `[tool.pytest.ini_options]`:
```toml
asyncio_mode = "auto"
```

- [ ] **Step 2: Sync environment**

```bash
cd /home/ava/dev/ORBIS && uv sync --extra test
```
Expected: exits 0, no errors.

- [ ] **Step 3: Verify imports**

```bash
cd /home/ava/dev/ORBIS && uv run python -c "import funasr; import pytest_asyncio; print('OK funasr', funasr.__version__)"
```
Expected: `OK funasr <version>`

- [ ] **Step 4: Commit**

```bash
cd /home/ava/dev/ORBIS && git add pyproject.toml && git commit -m "deps: add funasr>=1.1 for SenseVoice STT backend"
```

---

## Task 2: Define EmotionFrame and AudioEventFrame

**Files:**
- Create: `/home/ava/dev/ORBIS/agent/frames.py`
- Create: `/home/ava/dev/ORBIS/tests/test_frames.py`

- [ ] **Step 1: Write the failing test**

```bash
cat > /home/ava/dev/ORBIS/tests/test_frames.py << 'EOF'
"""Tests for custom ORBIS frame types in agent/frames.py."""

from __future__ import annotations
import pytest
from pipecat.frames.frames import Frame
from agent.frames import AudioEventFrame, EmotionFrame


def test_emotion_frame_is_frame_subclass():
    assert issubclass(EmotionFrame, Frame)


def test_emotion_frame_fields():
    f = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    assert f.emotion == "happy"
    assert f.confidence == "medium"
    assert f.lang == "en"
    assert f.speaker_verified is True
    assert f.audio_bytes == b""


def test_emotion_frame_bool_field():
    f = EmotionFrame(emotion="neutral", confidence="medium", lang="en", speaker_verified=False, audio_bytes=b"")
    assert f.speaker_verified is False


def test_audio_event_frame_is_frame_subclass():
    assert issubclass(AudioEventFrame, Frame)


def test_audio_event_frame_events_is_list():
    f = AudioEventFrame(events=["Laughter", "BGM"])
    assert isinstance(f.events, list)
    assert f.events == ["Laughter", "BGM"]


def test_audio_event_frame_default_events_is_empty_list():
    f = AudioEventFrame()
    assert f.events == []


def test_emotion_frame_all_seven_labels_are_valid_strings():
    for label in ("neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"):
        f = EmotionFrame(emotion=label, confidence="medium", lang="en", speaker_verified=False, audio_bytes=b"")
        assert f.emotion == label
EOF
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_frames.py -v
```
Expected: `ImportError` — `agent.frames` does not exist yet.

- [ ] **Step 3: Create `agent/frames.py`**

```bash
cat > /home/ava/dev/ORBIS/agent/frames.py << 'EOF'
"""Custom ORBIS frame types for the audio-perception pipeline.

Frames emitted by SenseVoiceSTT and consumed by AudioTagsTap.
Kept separate from pipecat's own frame module so the import graph
is clean — agent/ depends on pipecat frames, not the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipecat.frames.frames import Frame


@dataclass
class EmotionFrame(Frame):
    """Emitted by SenseVoiceSTT once per utterance.

    emotion: one of neutral | happy | sad | angry | fearful | disgusted |
             surprised | emo_unk
    confidence: "high" | "medium" | "low"  — always "medium" until FunASR
                exposes token-level logits (open question from design doc)
    lang: BCP-47 code detected by SenseVoice (en, zh, ja, ko, yue)
    speaker_verified: True when the last SpeakerGate frame seen by
                      SenseVoiceSTT was OwnerVerifiedFrame
    audio_bytes: raw WAV bytes for the utterance, carried for v5 head
                 inference in AudioTagsTap (empty if unavailable)
    """

    emotion: str
    confidence: str
    lang: str
    speaker_verified: bool
    audio_bytes: bytes = field(default_factory=bytes)


@dataclass
class AudioEventFrame(Frame):
    """Emitted by SenseVoiceSTT when non-speech audio events are detected.

    events: list of detected events, e.g. ["Laughter", "BGM"]
    Empty list is never emitted — this frame is only created when
    len(events) > 0.
    """

    events: list[str] = field(default_factory=list)
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_frames.py -v
```
Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/ava/dev/ORBIS && git add agent/frames.py tests/test_frames.py && git commit -m "feat(frames): add EmotionFrame and AudioEventFrame"
```

---

## Task 3: SenseVoiceSTT — stub + routing

**Files:**
- Modify: `/home/ava/dev/ORBIS/voice/stt.py`
- Create: `/home/ava/dev/ORBIS/tests/test_stt.py`

- [ ] **Step 1: Write the failing tests**

```bash
cat > /home/ava/dev/ORBIS/tests/test_stt.py << 'EOF'
"""Tests for voice/stt.py routing and SenseVoiceSTT skeleton."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from pipecat.services.stt_service import SegmentedSTTService


def _reload_stt(backend: str):
    """Reload voice.stt with patched STT_BACKEND."""
    os.environ["STT_BACKEND"] = backend
    if "voice.stt" in sys.modules:
        del sys.modules["voice.stt"]
    import voice.stt as fresh
    return fresh


def test_make_stt_local_returns_local_whisper():
    mod = _reload_stt("local")
    svc = mod.make_stt()
    assert isinstance(svc, mod.LocalWhisperSTT)


def test_make_stt_sensevoice_returns_sensevoice():
    mod = _reload_stt("sensevoice")
    svc = mod.make_stt()
    assert isinstance(svc, mod.SenseVoiceSTT)


def test_make_stt_unknown_falls_back_to_local():
    mod = _reload_stt("unknown_xyz")
    svc = mod.make_stt()
    assert isinstance(svc, mod.LocalWhisperSTT)


def test_sensevoicestt_is_segmented_stt_service():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT()
    assert isinstance(svc, SegmentedSTTService)


def test_sensevoicestt_run_stt_raises_not_implemented():
    """Stub must raise NotImplementedError until Task 4 is implemented."""
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT()

    async def _run():
        gen = svc.run_stt(b"fake")
        await gen.__anext__()

    with pytest.raises((NotImplementedError, StopAsyncIteration)):
        asyncio.run(_run())
EOF
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_stt.py -v
```
Expected: `AttributeError` or `ImportError` — `SenseVoiceSTT` does not exist yet.

- [ ] **Step 3: Add SenseVoiceSTT skeleton + routing to `voice/stt.py`**

Find the block where `_local_pipe = None` is defined. Add after it:

```python
# SenseVoice backend env vars
SENSEVOICE_MODEL = os.environ.get("SENSEVOICE_MODEL", "FunAudioLLM/SenseVoiceSmall")
SENSEVOICE_DEVICE = os.environ.get("SENSEVOICE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_sensevoice_model = None


def _get_sensevoice_model(loader_factory=None):
    """Return the loaded SenseVoice AutoModel, loading on first call.

    loader_factory: optional callable() -> model, injected in tests.
    """
    global _sensevoice_model
    if _sensevoice_model is not None:
        return _sensevoice_model
    if loader_factory is not None:
        _sensevoice_model = loader_factory()
        return _sensevoice_model
    from funasr import AutoModel
    logger.info(f"Loading SenseVoice {SENSEVOICE_MODEL} on {SENSEVOICE_DEVICE}")
    t0 = time.time()
    _sensevoice_model = AutoModel(
        model=SENSEVOICE_MODEL,
        device=SENSEVOICE_DEVICE,
        disable_update=True,
    )
    logger.info(f"SenseVoice ready in {time.time() - t0:.1f}s")
    return _sensevoice_model
```

Add `SenseVoiceSTT` class after `LocalWhisperSTT`:

```python
class SenseVoiceSTT(SegmentedSTTService):
    """Pipecat STT wrapper around FunASR SenseVoiceSmall.

    Emits EmotionFrame and AudioEventFrame before TranscriptionFrame on
    each utterance. See agent/frames.py for frame definitions.
    """

    def __init__(self, *, user_id: str = "user", loader_factory=None, **kwargs):
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        super().__init__(**kwargs)
        self._user_id = user_id
        self._loader_factory = loader_factory
        self._speaker_verified: bool = False

    async def run_stt(self, audio: bytes):
        raise NotImplementedError("SenseVoiceSTT.run_stt not yet implemented — Task 4")
        yield  # make this an async generator
```

Replace `make_stt()` body:

```python
def make_stt() -> SegmentedSTTService:
    """Return the configured STT service for the pipeline."""
    if STT_BACKEND == "openai":
        logger.info(f"STT backend: openai @ {STT_URL} model={STT_MODEL}")
        return OpenAISTTService(
            api_key=STT_API_KEY,
            base_url=STT_URL,
            settings=STTSettings(model=STT_MODEL, language=None),
        )
    if STT_BACKEND == "sensevoice":
        logger.info(f"STT backend: sensevoice model={SENSEVOICE_MODEL}")
        return SenseVoiceSTT()
    if STT_BACKEND != "local":
        logger.warning(f"Unknown STT_BACKEND={STT_BACKEND!r}; falling back to local")
    return LocalWhisperSTT()
```

Extend `prewarm()`:

```python
def prewarm() -> None:
    if STT_BACKEND == "openai":
        logger.info("STT backend: openai (no local prewarm)")
        return
    if STT_BACKEND == "sensevoice":
        _get_sensevoice_model()
        return
    _get_local_pipe()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_stt.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/ava/dev/ORBIS && git add voice/stt.py tests/test_stt.py && git commit -m "feat(stt): add SenseVoiceSTT skeleton and make_stt routing"
```

---

## Task 4: SenseVoiceSTT — inference + frame emission

**Files:**
- Modify: `/home/ava/dev/ORBIS/voice/stt.py`
- Modify: `/home/ava/dev/ORBIS/tests/test_stt.py`

- [ ] **Step 1: Add regex constants and postprocess helper to `voice/stt.py`**

After the `_get_sensevoice_model` function, add:

```python
import re

_EMOTION_RE = re.compile(
    r"<\|(HAPPY|SAD|ANGRY|FEARFUL|DISGUSTED|SURPRISED|NEUTRAL|EMO_UNK)\|>",
    re.IGNORECASE,
)
_LANG_RE = re.compile(r"<\|(en|zh|ja|ko|yue|UNKNOWN)\|>", re.IGNORECASE)


def _postprocess(raw_text: str) -> str:
    """Strip SenseVoice special tokens and return clean transcription text."""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        return rich_transcription_postprocess(raw_text)
    except Exception:
        return re.sub(r"<\|[^|]+\|>", "", raw_text).strip()
```

- [ ] **Step 2: Add imports to `voice/stt.py`**

Add to the existing pipecat imports block (check for existing imports first):
```python
from agent.frames import AudioEventFrame, EmotionFrame
from pipecat.processors.frame_processor import FrameDirection
```

- [ ] **Step 3: Add `process_frame` override to `SenseVoiceSTT`**

Add before `run_stt` in the `SenseVoiceSTT` class:

```python
async def process_frame(self, frame, direction) -> None:
    """Track SpeakerGate output to set speaker_verified on emitted frames."""
    await super().process_frame(frame, direction)
    try:
        from agent.speaker_gate import OwnerVerifiedFrame, StrangerDetectedFrame
        if isinstance(frame, OwnerVerifiedFrame):
            self._speaker_verified = True
        elif isinstance(frame, StrangerDetectedFrame):
            self._speaker_verified = False
    except ImportError:
        pass  # speaker_gate not yet merged — no-op
    await self.push_frame(frame, direction)
```

- [ ] **Step 4: Implement `run_stt` in `SenseVoiceSTT`**

Replace the `NotImplementedError` stub:

```python
async def run_stt(self, audio: bytes):
    logger.info(f"[stt.sensevoice] run_stt audio bytes={len(audio)}")
    try:
        data, sr = sf.read(io.BytesIO(audio), dtype="float32")
    except Exception as e:
        logger.warning(f"[stt.sensevoice] decode failed: {e}")
        from pipecat.frames.frames import ErrorFrame
        yield ErrorFrame(error=f"STT decode failed: {e}")
        return

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        data = soxr.resample(data, sr, 16000)

    if data.size == 0:
        logger.info("[stt.sensevoice] empty audio — no frames emitted")
        return

    duration_s = len(data) / 16000.0

    try:
        t0 = time.time()
        model = _get_sensevoice_model(self._loader_factory)
        result = model.generate(
            input=data,
            cache={},
            language="auto",
            use_itn=True,
            ban_emo_unk=True,
        )
        elapsed = time.time() - t0
        raw_text = result[0]["text"] if result else ""
    except Exception as e:
        logger.error(f"[stt.sensevoice] inference failed: {e}")
        from pipecat.frames.frames import ErrorFrame
        yield ErrorFrame(error=f"STT inference failed: {e}")
        return

    # Parse tokens from raw output before postprocessing strips them
    emotion_match = _EMOTION_RE.search(raw_text)
    emotion = emotion_match.group(1).lower() if emotion_match else "neutral"
    if emotion == "emo_unk":
        emotion = "neutral"

    lang_match = _LANG_RE.search(raw_text)
    lang = lang_match.group(1).lower() if lang_match else "en"
    if lang == "unknown":
        lang = "en"

    all_tokens = re.findall(r"<\|([^|]+)\|>", raw_text)
    skip = {emotion_match.group(1).lower() if emotion_match else "",
            lang_match.group(1).lower() if lang_match else "",
            "withitn", "woitn", "event"}
    events = [
        t for t in all_tokens
        if t.lower() not in skip
        and not _EMOTION_RE.match(f"<|{t}|>")
        and not _LANG_RE.match(f"<|{t}|>")
    ]

    text = _postprocess(raw_text).strip()

    logger.info(
        f"[stt.sensevoice] {duration_s:.2f}s → {elapsed:.2f}s "
        f"emotion={emotion} lang={lang} events={events} chars={len(text)}"
    )

    yield EmotionFrame(
        emotion=emotion,
        confidence="medium",
        lang=lang,
        speaker_verified=self._speaker_verified,
        audio_bytes=audio,
    )

    if events:
        yield AudioEventFrame(events=events)

    if text:
        from pipecat.utils.time import time_now_iso8601
        yield TranscriptionFrame(text, self._user_id, time_now_iso8601())
    else:
        logger.info("[stt.sensevoice] empty transcription — no TranscriptionFrame emitted")
```

- [ ] **Step 5: Write new tests in `tests/test_stt.py`**

Append to `tests/test_stt.py`:

```python
import io
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf
from pipecat.frames.frames import TranscriptionFrame

from agent.frames import AudioEventFrame, EmotionFrame


def _stub_model(raw_text: str):
    stub = MagicMock()
    stub.generate.return_value = [{"text": raw_text}]
    return lambda: stub


def _minimal_wav(seconds: float = 1.0) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype="float32"), 16000, format="WAV")
    return buf.getvalue()


def _collect(coro):
    async def _run():
        return [f async for f in coro]
    return asyncio.run(_run())


def test_emotion_emitted_before_transcription():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|HAPPY|>hello"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    types = [type(f).__name__ for f in frames]
    assert types.index("EmotionFrame") < types.index("TranscriptionFrame")


def test_happy_emotion_parsed():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|HAPPY|>hello"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    efs = [f for f in frames if isinstance(f, EmotionFrame)]
    assert efs[0].emotion == "happy"


def test_emo_unk_normalizes_to_neutral():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|EMO_UNK|>hello"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    efs = [f for f in frames if isinstance(f, EmotionFrame)]
    assert efs[0].emotion == "neutral"


def test_lang_token_parsed():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|zh|><|NEUTRAL|>你好"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    efs = [f for f in frames if isinstance(f, EmotionFrame)]
    assert efs[0].lang == "zh"


def test_audio_event_emitted_when_present():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|NEUTRAL|><|Laughter|>ha ha"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    aefs = [f for f in frames if isinstance(f, AudioEventFrame)]
    assert len(aefs) == 1
    assert "Laughter" in aefs[0].events


def test_audio_event_absent_when_none():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|NEUTRAL|>hello"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    aefs = [f for f in frames if isinstance(f, AudioEventFrame)]
    assert len(aefs) == 0


def test_empty_audio_yields_no_frames():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|NEUTRAL|>"))
    buf = io.BytesIO()
    sf.write(buf, np.zeros(0, dtype="float32"), 16000, format="WAV")
    frames = _collect(svc.run_stt(buf.getvalue()))
    assert len(frames) == 0


def test_speaker_verified_false_by_default():
    mod = _reload_stt("sensevoice")
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|HAPPY|>hi"))
    frames = _collect(svc.run_stt(_minimal_wav()))
    efs = [f for f in frames if isinstance(f, EmotionFrame)]
    assert efs[0].speaker_verified is False


def test_audio_bytes_carried_in_emotion_frame():
    mod = _reload_stt("sensevoice")
    wav = _minimal_wav()
    svc = mod.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|HAPPY|>hi"))
    frames = _collect(svc.run_stt(wav))
    efs = [f for f in frames if isinstance(f, EmotionFrame)]
    assert efs[0].audio_bytes == wav
```

- [ ] **Step 6: Run all STT tests**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_stt.py -v
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /home/ava/dev/ORBIS && git add voice/stt.py tests/test_stt.py && git commit -m "feat(stt): implement SenseVoiceSTT inference and frame emission"
```

---

## Task 5: AudioTagsTap — skeleton + owner tracking

**Files:**
- Create: `/home/ava/dev/ORBIS/agent/audio_tags.py`
- Create: `/home/ava/dev/ORBIS/tests/test_audio_tags_tap.py`

- [ ] **Step 1: Write the failing tests**

```bash
cat > /home/ava/dev/ORBIS/tests/test_audio_tags_tap.py << 'EOF'
"""Tests for AudioTagsTap FrameProcessor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.audio_tags import AudioTagsTap
from agent.frames import AudioEventFrame, EmotionFrame
from memory import Memory
from pipecat.frames.frames import Frame, SystemFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mem(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "orbis.sqlite")


def _build_tap(mem, **kwargs):
    tap = AudioTagsTap(mem=mem, **kwargs)
    captured: list[Frame] = []

    async def _capture(frame, direction=None):
        captured.append(frame)

    tap.push_frame = _capture  # type: ignore[method-assign]
    return tap, captured


async def _send(tap, frame):
    await tap.process_frame(frame, FrameDirection.DOWNSTREAM)


# ---------------------------------------------------------------------------
# Speaker tracking
# ---------------------------------------------------------------------------

async def test_owner_verified_sets_speaker_verified(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    OwnerVerifiedFrame = type("OwnerVerifiedFrame", (Frame,), {})
    StrangerDetectedFrame = type("StrangerDetectedFrame", (Frame,), {})
    import sys, unittest.mock as mock
    with mock.patch.dict("sys.modules", {"agent.speaker_gate": mock.MagicMock(
        OwnerVerifiedFrame=OwnerVerifiedFrame,
        StrangerDetectedFrame=StrangerDetectedFrame,
    )}):
        await _send(tap, OwnerVerifiedFrame())
    assert tap._speaker_verified is True


async def test_stranger_detected_clears_speaker_verified(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = True
    OwnerVerifiedFrame = type("OwnerVerifiedFrame", (Frame,), {})
    StrangerDetectedFrame = type("StrangerDetectedFrame", (Frame,), {})
    import unittest.mock as mock
    with mock.patch.dict("sys.modules", {"agent.speaker_gate": mock.MagicMock(
        OwnerVerifiedFrame=OwnerVerifiedFrame,
        StrangerDetectedFrame=StrangerDetectedFrame,
    )}):
        await _send(tap, StrangerDetectedFrame())
    assert tap._speaker_verified is False


async def test_irrelevant_frames_pass_through(tmp_path):
    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem)
    frame = UserStartedSpeakingFrame()
    await _send(tap, frame)
    assert frame in captured


async def test_emotion_frame_passes_through(tmp_path):
    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem)
    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    assert ef in captured
EOF
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_audio_tags_tap.py -v
```
Expected: `ImportError` — `agent.audio_tags` does not exist yet.

- [ ] **Step 3: Create `agent/audio_tags.py`**

```bash
cat > /home/ava/dev/ORBIS/agent/audio_tags.py << 'EOF'
"""Audio perception tap for ORBIS's voice pipeline.

AudioTagsTap is a FrameProcessor that:
  1. Tracks speaker identity from SpeakerGate output frames
  2. On EmotionFrame: applies a per-emotion mood nudge (owner audio only)
     and injects a [audio] annotation SystemFrame per turn
  3. On AudioEventFrame: stores events for the [audio] line
  4. Optionally runs audio-tags v5 non-emotion heads (SNR/env/rate)
  5. Passes all frames through unchanged after side effects

R15 fix: mood writes are always delta-based (read → delta → clamp → write).
The neglect baseline survives; emotion nudges it per turn.
"""

from __future__ import annotations

import logging
import os

from pipecat.frames.frames import Frame, SystemFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.frames import AudioEventFrame, EmotionFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Emotion → mood delta mapping (valence_delta, arousal_delta)
# ---------------------------------------------------------------------------
_EMOTION_DELTAS: dict[str, tuple[float, float]] = {
    "happy":     (+0.10, +0.05),
    "surprised": ( 0.00, +0.10),
    "neutral":   ( 0.00,  0.00),
    "sad":       (-0.10, -0.05),
    "fearful":   (-0.05, +0.10),
    "angry":     (-0.15, +0.15),
    "disgusted": (-0.10, +0.05),
}

# ---------------------------------------------------------------------------
# v5 head config
# ---------------------------------------------------------------------------
_AUDIO_TAGS_ENABLED = os.environ.get("AUDIO_TAGS", "on").lower() == "on"
_AUDIO_TAGS_MODEL = os.environ.get("AUDIO_TAGS_MODEL", "protoLabsAI/orbis-audio-tags-v5-soft")

_v5_model = None


def _get_v5_model(loader_factory=None):
    """Return the audio-tags v5 model, loading on first call."""
    global _v5_model
    if _v5_model is not None:
        return _v5_model
    if loader_factory is not None:
        _v5_model = loader_factory()
        return _v5_model
    from transformers import pipeline as hf_pipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading audio-tags v5 {_AUDIO_TAGS_MODEL} on {device}")
    _v5_model = hf_pipeline("audio-classification", model=_AUDIO_TAGS_MODEL, device=device)
    return _v5_model


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


# ---------------------------------------------------------------------------
# AudioTagsTap
# ---------------------------------------------------------------------------

class AudioTagsTap(FrameProcessor):
    """Observes EmotionFrame; writes mood nudges and injects [audio] SystemFrame."""

    def __init__(self, *, mem, v5_loader_factory=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mem = mem
        self._v5_loader_factory = v5_loader_factory
        self._speaker_verified: bool = False
        self._last_events: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # Track speaker identity from SpeakerGate (guarded — branch not yet merged)
        try:
            from agent.speaker_gate import OwnerVerifiedFrame, StrangerDetectedFrame
            if isinstance(frame, OwnerVerifiedFrame):
                self._speaker_verified = True
                await self.push_frame(frame, direction)
                return
            if isinstance(frame, StrangerDetectedFrame):
                self._speaker_verified = False
                await self.push_frame(frame, direction)
                return
        except ImportError:
            pass

        # Reset per-turn events on new user speech
        if isinstance(frame, UserStartedSpeakingFrame):
            self._last_events = []
            await self.push_frame(frame, direction)
            return

        # Cache audio events for [audio] line
        if isinstance(frame, AudioEventFrame):
            self._last_events = list(frame.events)
            await self.push_frame(frame, direction)
            return

        # Emotion: apply mood nudge + inject SystemFrame
        if isinstance(frame, EmotionFrame):
            await self._handle_emotion(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _handle_emotion(self, frame: EmotionFrame, direction: FrameDirection) -> None:
        """Apply mood nudge (owner-only, delta-write) and emit [audio] SystemFrame."""
        dv, da = _EMOTION_DELTAS.get(frame.emotion, (0.0, 0.0))

        if frame.speaker_verified and (dv != 0.0 or da != 0.0):
            try:
                current = self._mem.personality.get_mood()
                new_valence = _clamp(current.valence + dv)
                new_arousal = _clamp(current.arousal + da)
                self._mem.personality.set_mood(valence=new_valence, arousal=new_arousal)
                logger.info(
                    f"[audio_tags] mood nudge emotion={frame.emotion} "
                    f"dv={dv:+.2f} da={da:+.2f} "
                    f"valence={new_valence:.3f} arousal={new_arousal:.3f}"
                )
            except Exception as e:
                logger.warning(f"[audio_tags] mood write failed: {e}")

        # v5 non-emotion heads
        snr_label = env_label = rate_label = None
        if _AUDIO_TAGS_ENABLED and frame.audio_bytes:
            try:
                v5 = _get_v5_model(self._v5_loader_factory)
                preds = v5(frame.audio_bytes)
                snr_label = preds.get("snr")
                env_label = preds.get("env")
                rate_label = preds.get("rate")
                logger.info(
                    f"[audio_tags] v5 heads snr={snr_label} env={env_label} rate={rate_label}"
                )
            except Exception as e:
                logger.warning(f"[audio_tags] v5 head inference failed: {e}")

        # Build [audio] annotation line
        speaker_label = "owner" if frame.speaker_verified else "stranger"
        audio_line = (
            f"[audio] emotion={frame.emotion} lang={frame.lang} "
            f"events={self._last_events} speaker={speaker_label}"
        )
        if snr_label:
            audio_line += f" snr={snr_label}"
        if env_label:
            audio_line += f" env={env_label}"
        if rate_label:
            audio_line += f" rate={rate_label}"

        logger.debug(f"[audio_tags] injecting SystemFrame: {audio_line}")

        # Emit SystemFrame BEFORE EmotionFrame so [audio] context lands
        # in LLMContextAggregator before the user text arrives.
        await self.push_frame(SystemFrame(text=audio_line), direction)
        await self.push_frame(frame, direction)
EOF
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_audio_tags_tap.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/ava/dev/ORBIS && git add agent/audio_tags.py tests/test_audio_tags_tap.py && git commit -m "feat(audio-tags): add AudioTagsTap with speaker tracking, mood nudge, and [audio] SystemFrame"
```

---

## Task 6: AudioTagsTap — mood nudge tests + R15 fix verification

**Files:**
- Modify: `/home/ava/dev/ORBIS/tests/test_audio_tags_tap.py`
- Modify: `/home/ava/dev/ORBIS/tests/test_neglect.py`

- [ ] **Step 1: Append mood nudge tests to `test_audio_tags_tap.py`**

```python
# ---------------------------------------------------------------------------
# Mood nudge tests
# ---------------------------------------------------------------------------

async def test_owner_happy_writes_mood(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = True
    mem.personality.set_mood(valence=0.0, arousal=0.0)
    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    updated = mem.personality.get_mood()
    assert abs(updated.valence - 0.10) < 1e-6
    assert abs(updated.arousal - 0.05) < 1e-6


async def test_stranger_suppresses_mood_write(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = False
    mem.personality.set_mood(valence=0.0, arousal=0.0)
    ef = EmotionFrame(emotion="angry", confidence="medium", lang="en", speaker_verified=False, audio_bytes=b"")
    await _send(tap, ef)
    updated = mem.personality.get_mood()
    assert updated.valence == 0.0 and updated.arousal == 0.0


async def test_neutral_emotion_no_mood_write(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = True
    mem.personality.set_mood(valence=0.0, arousal=0.0)
    ef = EmotionFrame(emotion="neutral", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    updated = mem.personality.get_mood()
    assert updated.valence == 0.0 and updated.arousal == 0.0


@pytest.mark.parametrize("emotion,dv,da", [
    ("happy",     +0.10, +0.05),
    ("surprised",  0.00, +0.10),
    ("neutral",    0.00,  0.00),
    ("sad",       -0.10, -0.05),
    ("fearful",   -0.05, +0.10),
    ("angry",     -0.15, +0.15),
    ("disgusted", -0.10, +0.05),
])
async def test_all_emotion_deltas(tmp_path, emotion, dv, da):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = True
    mem.personality.set_mood(valence=0.0, arousal=0.0)
    ef = EmotionFrame(emotion=emotion, confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    updated = mem.personality.get_mood()
    if dv == 0.0 and da == 0.0:
        assert updated.valence == 0.0 and updated.arousal == 0.0
    else:
        assert abs(updated.valence - dv) < 1e-6
        assert abs(updated.arousal - da) < 1e-6


async def test_valence_clamped_at_positive_limit(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = True
    mem.personality.set_mood(valence=0.95, arousal=0.0)
    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    assert mem.personality.get_mood().valence == 1.0


async def test_arousal_clamped_at_negative_limit(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._speaker_verified = True
    mem.personality.set_mood(valence=0.0, arousal=-0.98)
    ef = EmotionFrame(emotion="sad", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    assert mem.personality.get_mood().arousal == -1.0


# ---------------------------------------------------------------------------
# SystemFrame tests
# ---------------------------------------------------------------------------

async def test_system_frame_emitted_with_owner_label(tmp_path):
    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem)
    tap._speaker_verified = True
    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    sys_frames = [f for f in captured if isinstance(f, SystemFrame)]
    assert len(sys_frames) == 1
    assert "speaker=owner" in sys_frames[0].text


async def test_system_frame_emitted_with_stranger_label(tmp_path):
    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem)
    tap._speaker_verified = False
    ef = EmotionFrame(emotion="angry", confidence="medium", lang="en", speaker_verified=False, audio_bytes=b"")
    await _send(tap, ef)
    sys_frames = [f for f in captured if isinstance(f, SystemFrame)]
    assert len(sys_frames) == 1
    assert "speaker=stranger" in sys_frames[0].text


async def test_system_frame_includes_cached_events(tmp_path):
    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem)
    tap._speaker_verified = True
    await _send(tap, AudioEventFrame(events=["Laughter"]))
    captured.clear()
    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    sys_frames = [f for f in captured if isinstance(f, SystemFrame)]
    assert "Laughter" in sys_frames[0].text


async def test_system_frame_emitted_before_emotion_frame(tmp_path):
    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem)
    tap._speaker_verified = True
    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await _send(tap, ef)
    types = [type(f).__name__ for f in captured]
    assert types.index("SystemFrame") < types.index("EmotionFrame")


async def test_user_started_speaking_resets_last_events(tmp_path):
    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem)
    tap._last_events = ["Laughter"]
    await _send(tap, UserStartedSpeakingFrame())
    assert tap._last_events == []
```

- [ ] **Step 2: Add R15 test to `tests/test_neglect.py`**

Append to `tests/test_neglect.py`:

```python
# ---------------------------------------------------------------------------
# R15: neglect baseline + emotion nudge interaction
# ---------------------------------------------------------------------------

async def test_r15_emotion_nudge_preserves_neglect_baseline(tmp_path):
    """apply_soft_neglect sets a baseline; AudioTagsTap nudges from it.
    Result must be baseline + delta, not just the delta."""
    from agent.audio_tags import AudioTagsTap
    from agent.frames import EmotionFrame

    mem = Memory(tmp_path / "orbis.sqlite")
    _seed_session(mem, days_ago=5.0)
    apply_soft_neglect(mem)
    baseline = mem.personality.get_mood()
    assert baseline.valence < 0.0  # neglect should have depressed valence

    tap = AudioTagsTap(mem=mem)
    captured: list = []
    async def _capture(frame, direction=None):
        captured.append(frame)
    tap.push_frame = _capture  # type: ignore[method-assign]
    tap._speaker_verified = True

    ef = EmotionFrame(emotion="angry", confidence="medium", lang="en", speaker_verified=True, audio_bytes=b"")
    await tap.process_frame(ef, None)

    final = mem.personality.get_mood()
    expected_valence = max(-1.0, baseline.valence + (-0.15))
    expected_arousal = max(-1.0, baseline.arousal + 0.15)
    assert abs(final.valence - expected_valence) < 1e-6
    assert abs(final.arousal - expected_arousal) < 1e-6
```

- [ ] **Step 3: Run all audio-tags + neglect tests**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_audio_tags_tap.py tests/test_neglect.py -v
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/ava/dev/ORBIS && git add tests/test_audio_tags_tap.py tests/test_neglect.py && git commit -m "test: mood nudge, SystemFrame ordering, and R15 neglect-baseline tests"
```

---

## Task 7: AudioTagsTap — audio-tags v5 head tests

**Files:**
- Modify: `/home/ava/dev/ORBIS/tests/test_audio_tags_tap.py`

- [ ] **Step 1: Append v5 head tests**

```python
# ---------------------------------------------------------------------------
# v5 non-emotion head tests
# ---------------------------------------------------------------------------

def _make_v5_stub(snr="high", env="indoor", rate="normal"):
    stub = MagicMock()
    stub.return_value = {"snr": snr, "env": env, "rate": rate}
    return lambda: stub


import io as _io
import numpy as _np
import soundfile as _sf


def _minimal_wav(seconds: float = 0.5) -> bytes:
    buf = _io.BytesIO()
    _sf.write(buf, _np.zeros(int(16000 * seconds), dtype="float32"), 16000, format="WAV")
    return buf.getvalue()


async def test_v5_heads_appear_in_audio_line(tmp_path, monkeypatch):
    import agent.audio_tags as at_mod
    at_mod._v5_model = None
    at_mod._AUDIO_TAGS_ENABLED = True

    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem, v5_loader_factory=_make_v5_stub())
    tap._speaker_verified = True

    ef = EmotionFrame(
        emotion="happy", confidence="medium", lang="en",
        speaker_verified=True, audio_bytes=_minimal_wav()
    )
    await _send(tap, ef)
    sys_frames = [f for f in captured if isinstance(f, SystemFrame)]
    assert "snr=high" in sys_frames[0].text
    assert "env=indoor" in sys_frames[0].text
    assert "rate=normal" in sys_frames[0].text


async def test_v5_heads_absent_when_disabled(tmp_path):
    import agent.audio_tags as at_mod
    at_mod._v5_model = None
    at_mod._AUDIO_TAGS_ENABLED = False

    mem = _make_mem(tmp_path)
    tap, captured = _build_tap(mem, v5_loader_factory=_make_v5_stub())
    tap._speaker_verified = True

    ef = EmotionFrame(
        emotion="happy", confidence="medium", lang="en",
        speaker_verified=True, audio_bytes=_minimal_wav()
    )
    await _send(tap, ef)
    sys_frames = [f for f in captured if isinstance(f, SystemFrame)]
    assert "snr=" not in sys_frames[0].text


async def test_v5_loader_called_once(tmp_path):
    import agent.audio_tags as at_mod
    at_mod._v5_model = None
    at_mod._AUDIO_TAGS_ENABLED = True

    call_count = {"n": 0}
    def _counting_factory():
        call_count["n"] += 1
        stub = MagicMock()
        stub.return_value = {"snr": "high", "env": "indoor", "rate": "normal"}
        return stub

    mem = _make_mem(tmp_path)
    tap, _ = _build_tap(mem, v5_loader_factory=_counting_factory)
    tap._speaker_verified = True
    wav = _minimal_wav()

    for _ in range(3):
        ef = EmotionFrame(
            emotion="happy", confidence="medium", lang="en",
            speaker_verified=True, audio_bytes=wav
        )
        await _send(tap, ef)

    assert call_count["n"] == 1
```

- [ ] **Step 2: Run tests**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_audio_tags_tap.py -v
```
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
cd /home/ava/dev/ORBIS && git add tests/test_audio_tags_tap.py && git commit -m "test: audio-tags v5 head inference tests (SNR/env/rate)"
```

---

## Task 8: audio_context_block + _effective_prompt wiring

**Files:**
- Modify: `/home/ava/dev/ORBIS/app.py`

- [ ] **Step 1: Find the block-function section**

```bash
grep -n "^def.*_block" /home/ava/dev/ORBIS/app.py
```
Note the line number of `repair_block`. Add `audio_context_block` immediately after it.

- [ ] **Step 2: Add `audio_context_block()` to `app.py`**

After `repair_block()`:

```python
def audio_context_block() -> str:
    """Explain the [audio] per-turn annotation to the orb.

    Static — same text every session. The per-turn emotion value arrives
    via SystemFrame injected by AudioTagsTap, not here.
    """
    return (
        "## AUDIO CONTEXT\n\n"
        "Each turn may begin with an [audio] line describing how the user "
        "sounded: emotion, language, and background events detected from "
        "their voice before transcription. Use this as additional signal — "
        "if emotion=angry, lean toward de-escalation; if emotion=happy, "
        "match the energy. Never quote or paraphrase the [audio] line back "
        "to the user. It is your perception, not their statement."
    )
```

- [ ] **Step 3: Add `audio_context_block()` to `_effective_prompt()` return block**

Find the return statement in `_effective_prompt()`. Add `+ "\n\n" + audio_context_block()` after `+ repair_block()`. Final block:

```python
    return (
        base
        + "\n\n"
        + tool_use_block(verbosity, tts_backend)
        + "\n\n"
        + tool_response_block(verbosity)
        + (("\n\n" + plan) if plan else "")
        + "\n\n"
        + repair_block()
        + "\n\n"
        + audio_context_block()
        + (("\n\n" + user_block) if user_block else "")
        + (("\n\n" + personality) if personality else "")
        + (("\n\n## RETURN\n\n" + neglect_nudge) if neglect_nudge else "")
        + (("\n\n" + recall) if recall else "")
    )
```

- [ ] **Step 4: Verify `audio_context_block` appears in composed prompt**

```bash
cd /home/ava/dev/ORBIS && uv run python -c "
from app import _effective_prompt, audio_context_block

class FakeSkill:
    system_prompt = 'You are a test orb.'
    user_name = ''

result = _effective_prompt(FakeSkill(), 'kokoro', verbosity='brief', user_id='test')
assert '## AUDIO CONTEXT' in result, 'AUDIO CONTEXT block missing from prompt'
print('OK — AUDIO CONTEXT found in composed prompt')
"
```
Expected: `OK — AUDIO CONTEXT found in composed prompt`

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/ava/dev/ORBIS && git add app.py && git commit -m "feat(app): add audio_context_block and wire into _effective_prompt"
```

---

## Task 9: Wire AudioTagsTap into the pipeline

**Files:**
- Modify: `/home/ava/dev/ORBIS/app.py`

- [ ] **Step 1: Add import**

Near the top of `app.py`, alongside other `agent/` imports:

```python
from agent.audio_tags import AudioTagsTap
```

- [ ] **Step 2: Instantiate `AudioTagsTap` in `run_bot()`**

In `run_bot()`, after `stt = make_stt()`:

```python
audio_tags_tap = AudioTagsTap(mem=get_memory())
```

- [ ] **Step 3: Insert into Pipeline**

Find the `Pipeline([...])` call. Change the order so:

```python
pipeline = Pipeline([
    transport.input(),
    EchoGuardSuppressor(_ECHO_STATE),
    rtvi,
    # TODO(#35-pr1): insert SpeakerGate here once feat/speaker-gate-foundation-issue-35 merges
    stt,            # SenseVoiceSTT: emits EmotionFrame + AudioEventFrame + TranscriptionFrame
    audio_tags_tap, # consumes EmotionFrame → mood nudge + [audio] SystemFrame
    user_agg,       # sees SystemFrame + TranscriptionFrame for LLM context
    BargeInGate(...),
    # ... rest unchanged
])
```

- [ ] **Step 4: Smoke test import**

```bash
cd /home/ava/dev/ORBIS && STT_BACKEND=sensevoice AUDIO_TAGS=off uv run python -c "import app; print('app import OK')"
```
Expected: `app import OK`

- [ ] **Step 5: Run full test suite**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/ava/dev/ORBIS && git add app.py && git commit -m "feat(app): wire AudioTagsTap into pipeline after SenseVoiceSTT"
```

---

## Task 10: Integration smoke test

**Files:**
- Create: `/home/ava/dev/ORBIS/tests/test_integration_smoke.py`

- [ ] **Step 1: Write the tests**

```bash
cat > /home/ava/dev/ORBIS/tests/test_integration_smoke.py << 'EOF'
"""End-to-end frame ordering smoke test.

Verifies the full turn sequence:
  STT output: EmotionFrame → (AudioEventFrame?) → TranscriptionFrame
  After tap:  SystemFrame  → EmotionFrame        → TranscriptionFrame
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from agent.audio_tags import AudioTagsTap
from agent.frames import AudioEventFrame, EmotionFrame
from memory import Memory
from pipecat.frames.frames import Frame, SystemFrame, TranscriptionFrame


def _make_mem(tmp_path: Path) -> Memory:
    return Memory(tmp_path / "orbis.sqlite")


def _minimal_wav(seconds: float = 0.5) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype="float32"), 16000, format="WAV")
    return buf.getvalue()


def _stub_model(raw_text: str):
    stub = MagicMock()
    stub.generate.return_value = [{"text": raw_text}]
    return lambda: stub


async def test_stt_emotion_before_transcription():
    import voice.stt as s
    svc = s.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|HAPPY|>hello"))
    frames = [f async for f in svc.run_stt(_minimal_wav())]
    types = [type(f).__name__ for f in frames]
    assert types.index("EmotionFrame") < types.index("TranscriptionFrame")


async def test_stt_event_before_transcription():
    import voice.stt as s
    svc = s.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|NEUTRAL|><|Laughter|>ha"))
    frames = [f async for f in svc.run_stt(_minimal_wav())]
    types = [type(f).__name__ for f in frames]
    assert types.index("AudioEventFrame") < types.index("TranscriptionFrame")


async def test_tap_system_frame_before_emotion_frame(tmp_path):
    mem = _make_mem(tmp_path)
    tap = AudioTagsTap(mem=mem)
    captured: list[Frame] = []
    async def _capture(frame, direction=None):
        captured.append(frame)
    tap.push_frame = _capture  # type: ignore[method-assign]
    tap._speaker_verified = True

    ef = EmotionFrame(emotion="happy", confidence="medium", lang="en",
                      speaker_verified=True, audio_bytes=_minimal_wav())
    await tap.process_frame(ef, None)
    types = [type(f).__name__ for f in captured]
    assert types.index("SystemFrame") < types.index("EmotionFrame")


async def test_full_turn_sequence(tmp_path):
    """STT frames fed through tap produce: SystemFrame → EmotionFrame → TranscriptionFrame."""
    import voice.stt as s
    mem = _make_mem(tmp_path)
    wav = _minimal_wav()

    svc = s.SenseVoiceSTT(loader_factory=_stub_model("<|en|><|HAPPY|>hello"))
    stt_frames = [f async for f in svc.run_stt(wav)]

    tap = AudioTagsTap(mem=mem)
    final_frames: list[Frame] = []
    async def _capture(frame, direction=None):
        final_frames.append(frame)
    tap.push_frame = _capture  # type: ignore[method-assign]
    tap._speaker_verified = True

    for frame in stt_frames:
        await tap.process_frame(frame, None)

    types = [type(f).__name__ for f in final_frames]
    assert "SystemFrame" in types
    assert "EmotionFrame" in types
    assert "TranscriptionFrame" in types
    assert types.index("SystemFrame") < types.index("EmotionFrame")
    assert types.index("EmotionFrame") < types.index("TranscriptionFrame")
EOF
```

- [ ] **Step 2: Run integration smoke tests**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/test_integration_smoke.py -v
```
Expected: all 4 tests pass.

- [ ] **Step 3: Run full suite**

```bash
cd /home/ava/dev/ORBIS && uv run pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/ava/dev/ORBIS && git add tests/test_integration_smoke.py && git commit -m "test: integration smoke test for SenseVoice + AudioTagsTap frame ordering"
```

---

## Manual live verification (after Task 9)

Once all tests pass, verify with a real session:

1. Start ORBIS:
   ```bash
   cd /home/ava/dev/ORBIS
   STT_BACKEND=sensevoice AUDIO_TAGS=on uv run python app.py
   ```

2. Speak for 2-3 seconds and verify log output:
   ```
   [stt.sensevoice] 2.1s → 0.07s emotion=happy lang=en events=[] chars=12
   [audio_tags] mood nudge emotion=happy dv=+0.10 da=+0.05 valence=0.100 arousal=0.050
   [audio_tags] injecting SystemFrame: [audio] emotion=happy lang=en events=[] speaker=owner snr=high env=indoor rate=normal
   ```

3. Verify the `[audio]` line appears in the LLM context by enabling debug logging:
   ```bash
   LOG_LEVEL=DEBUG STT_BACKEND=sensevoice AUDIO_TAGS=on uv run python app.py
   ```

---

## Commit plan summary

```
Task 1:  deps: add funasr>=1.1 for SenseVoice STT backend
Task 2:  feat(frames): add EmotionFrame and AudioEventFrame
Task 3:  feat(stt): add SenseVoiceSTT skeleton and make_stt routing
Task 4:  feat(stt): implement SenseVoiceSTT inference and frame emission
Task 5:  feat(audio-tags): add AudioTagsTap with speaker tracking, mood nudge, and [audio] SystemFrame
Task 6:  test: mood nudge, SystemFrame ordering, and R15 neglect-baseline tests
Task 7:  test: audio-tags v5 head inference tests (SNR/env/rate)
Task 8:  feat(app): add audio_context_block and wire into _effective_prompt
Task 9:  feat(app): wire AudioTagsTap into pipeline after SenseVoiceSTT
Task 10: test: integration smoke test for SenseVoice + AudioTagsTap frame ordering
```

---

## Risks

| Risk | Mitigation |
|---|---|
| FunASR raw output token format differs from spec | `_EMOTION_RE` is case-insensitive, no crash on non-match; `emo_unk` normalized. Verify against real output in manual live test. |
| `process_frame` override in `SenseVoiceSTT` interferes with superclass audio buffering | Override calls `await super().process_frame(frame, direction)` first — superclass logic runs before gate-frame tracking. |
| `speaker_gate` import inside `process_frame` causing overhead | `ImportError` path only runs until branch merges; after merge, `sys.modules` dict lookup. Negligible. |
| `SystemFrame(text=...)` constructor signature differs between pipecat versions | Verify with `uv run python -c "from pipecat.frames.frames import SystemFrame; import inspect; print(inspect.signature(SystemFrame))"`. Adjust if needed. |
| `_AUDIO_TAGS_ENABLED` read at module import time — `monkeypatch.setenv` ineffective in tests | Tests in Task 7 directly patch `at_mod._AUDIO_TAGS_ENABLED` after env change. Documented in test comments. |
| PR 1+2 (speaker-gate) not merged before Task 9 | All `speaker_gate` imports are `try/except ImportError`. `_speaker_verified` defaults to `False` (stranger-safe). TODO comment left in pipeline for SpeakerGate insertion point. |
