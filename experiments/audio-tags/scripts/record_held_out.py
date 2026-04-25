"""Interactive recorder for the held-out whisper evaluation set.

Records ~30 paired voiced/whispered utterances from the user with prompts.
Used as the OUT-OF-DISTRIBUTION test for v2 — independent of any
synthetic training data, the real check on whether the model
generalizes from DSP-whisperized speech to actual whispered speech.

Saves 16 kHz mono WAV + a manifest jsonl for join with labels.

Usage:
  python scripts/record_held_out.py --out-dir /mnt/data/audio-tags/held_out/

Controls:
  ENTER       — start recording
  ENTER again — stop recording
  's'         — skip prompt
  'q'         — quit (saves what's recorded so far)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

PROMPTS = [
    "The weather has been strange this week.",
    "Tell me what you remember about yesterday.",
    "I'd like to set a reminder for tomorrow morning.",
    "Could you help me find my keys?",
    "What's the latest from the team?",
    "Did you hear what happened at the office?",
    "I'm thinking about taking a walk later.",
    "Show me the calendar for next week.",
    "My favorite book is sitting on the desk.",
    "Read me the headlines from this morning.",
    "Send a message to Alex about the project.",
    "Set a timer for fifteen minutes please.",
    "Remind me to call my mother on Sunday.",
    "What's on the agenda for the meeting?",
    "Play some quiet music for the evening.",
]

SR = 16000


def record_until_stop(prompt_id: int, label: str, out_path: Path) -> dict:
    print(f"\n  >>> Press ENTER to start recording, ENTER again to stop", flush=True)
    input()
    print(f"  ... recording (press ENTER to stop)", flush=True)
    chunks: list[np.ndarray] = []
    started = time.time()

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
        # Read in 100ms chunks; stop when stdin sees a newline (non-blocking-ish)
        import select
        while True:
            data, _ = stream.read(SR // 10)
            chunks.append(data.copy())
            r, _, _ = select.select([sys.stdin], [], [], 0.01)
            if r:
                sys.stdin.readline()
                break

    audio = np.concatenate(chunks).flatten()
    duration = len(audio) / SR
    print(f"  ... captured {duration:.1f}s, RMS={np.sqrt(np.mean(audio**2)):.4f}", flush=True)
    sf.write(out_path, audio, SR)
    return {
        "audio_path": str(out_path),
        "voice_quality": label,
        "duration": duration,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/data/audio-tags/held_out/")
    ap.add_argument("--n", type=int, default=15, help="prompts to record (each is voiced + whispered)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    print("=" * 60)
    print("AUDIO-TAGS HELD-OUT RECORDER")
    print("=" * 60)
    print(f"Output: {out_dir}")
    print(f"Will record {args.n} prompts × 2 styles (voiced + whispered) = {args.n*2} clips.")
    print("Sit close to the mic. Speak each prompt twice — once normally,")
    print("once whispered. Try to vary intensity in the whispered takes.")
    print()

    # Resume support
    done: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["audio_path"])
                except Exception:
                    pass
    print(f"Resuming: {len(done)} clips already recorded.\n")

    rows: list[dict] = []
    with manifest_path.open("a") as mf:
        for i, prompt in enumerate(PROMPTS[: args.n]):
            print("-" * 60)
            print(f"Prompt {i+1}/{args.n}:")
            print(f'  "{prompt}"')
            for label in ("voiced", "whispered"):
                stem = f"prompt_{i:02d}_{label}.wav"
                out_path = out_dir / stem
                if str(out_path) in done:
                    print(f"  [{label}] skipping — already recorded")
                    continue
                print(f"  [{label} take]")
                try:
                    row = record_until_stop(i, label, out_path)
                    row["prompt_id"] = i
                    row["prompt_text"] = prompt
                    mf.write(json.dumps(row) + "\n")
                    mf.flush()
                    rows.append(row)
                except KeyboardInterrupt:
                    print("\n  ABORTED — saving and exiting.")
                    return

    print("\n=" * 60)
    print(f"Done. {len(rows)} clips saved → {manifest_path}")


if __name__ == "__main__":
    main()
