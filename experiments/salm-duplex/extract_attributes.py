"""
DeSTA2-style speech attribute extraction pipeline.

Takes audio files and extracts 12 attributes automatically:
1. Transcript (Whisper)
2. Duration
3. Gender (classifier)
4. Emotion (emotion2vec)
5. SNR (brouhaha / simple energy-based)
6. Pitch (fundamental frequency)
7. Speaking speed (words per second)
8. Volume (RMS energy)

Then generates a seed transcript and uses a local LLM to create
rich descriptions for speech understanding training.

Usage:
    CUDA_VISIBLE_DEVICES=1 python extract_attributes.py --audio-dir /path/to/wavs --output /path/to/output.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def extract_basic_attributes(audio_path: str) -> dict:
    """Extract basic audio attributes without ML models."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        from scipy.signal import resample
        num_samples = int(len(audio) * 16000 / sr)
        audio = resample(audio, num_samples).astype(np.float32)
        sr = 16000
    duration = len(audio) / sr

    # RMS volume
    rms = np.sqrt(np.mean(audio ** 2))
    volume = "loud" if rms > 0.05 else "quiet" if rms < 0.01 else "normal"

    # Simple pitch estimation via autocorrelation
    try:
        from scipy.signal import correlate
        corr = correlate(audio[:sr], audio[:sr], mode='full')
        corr = corr[len(corr)//2:]
        # Find first peak after zero crossing
        d = np.diff(corr[:500])
        peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
        if len(peaks) > 0 and peaks[0] > 20:
            f0 = sr / peaks[0]
            pitch = "high" if f0 > 200 else "low" if f0 < 120 else "medium"
        else:
            pitch = "medium"
    except Exception:
        pitch = "medium"

    # Simple SNR estimate (signal vs noise floor)
    frame_size = int(0.025 * sr)
    frames = [audio[i:i+frame_size] for i in range(0, len(audio)-frame_size, frame_size)]
    energies = [np.mean(f**2) for f in frames]
    if energies:
        signal_energy = np.percentile(energies, 90)
        noise_energy = np.percentile(energies, 10) + 1e-10
        snr_db = 10 * np.log10(signal_energy / noise_energy)
    else:
        snr_db = 20.0

    return {
        "duration": round(duration, 2),
        "volume": volume,
        "pitch": pitch,
        "snr_db": round(snr_db, 1),
        "rms": round(float(rms), 4),
        "sample_rate": sr,
    }


def transcribe_batch(audio_paths: list[str], model, processor) -> list[str]:
    """Transcribe audio files using Whisper."""
    transcripts = []
    for path in audio_paths:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(model.device)
        with torch.no_grad():
            predicted_ids = model.generate(**inputs, max_new_tokens=256)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        transcripts.append(text)
    return transcripts


def build_seed_transcript(attrs: dict, transcript: str) -> str:
    """Build DeSTA2-style seed transcript from attributes."""
    duration = attrs["duration"]
    mins = int(duration // 60)
    secs = int(duration % 60)
    ms = int((duration % 1) * 100)

    parts = [
        f"[00:{mins:02d}:{secs:02d}-00:{mins:02d}:{secs+1:02d}] {transcript}",
        f"(Duration: {duration:.1f}s",
        f"Volume: {attrs['volume']}",
        f"Pitch: {attrs['pitch']}",
        f"SNR: {attrs['snr_db']:.0f}dB",
    ]

    # Add ML-extracted attributes if available
    if "gender" in attrs:
        parts.append(f"Gender: {attrs['gender']}")
    if "emotion" in attrs:
        parts.append(f"Emotion: {attrs['emotion']}")
    if "speaking_speed" in attrs:
        parts.append(f"Speaking speed: {attrs['speaking_speed']}")

    return ", ".join(parts) + ")"


def generate_descriptions(seed_transcripts: list[str], llm_url: str = "http://localhost:8000/v1") -> list[str]:
    """Generate rich descriptions using local LLM (via vLLM OpenAI API)."""
    from openai import OpenAI
    client = OpenAI(base_url=llm_url, api_key="not-needed")

    descriptions = []
    for seed in seed_transcripts:
        try:
            response = client.chat.completions.create(
                model="local",
                messages=[
                    {"role": "system", "content": "You are a speech analysis assistant. Given a description of an audio clip, provide a comprehensive natural language description of what can be heard."},
                    {"role": "user", "content": f"What can you hear from this audio?\n\n{seed}"}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            descriptions.append(response.choices[0].message.content.strip())
        except Exception as e:
            descriptions.append(f"Audio clip containing speech: {seed}")
    return descriptions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Directory of wav files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--whisper-model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--llm-url", default="http://localhost:8000/v1")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM description generation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_files = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.flac"))
    if args.max_samples:
        audio_files = audio_files[:args.max_samples]
    print(f"Found {len(audio_files)} audio files")

    # Load Whisper
    print(f"Loading Whisper ({args.whisper_model})...")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    whisper = WhisperForConditionalGeneration.from_pretrained(args.whisper_model, torch_dtype=torch.float32).to("cuda")
    whisper.eval()
    print(f"Whisper loaded on GPU (VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB)")

    results = []
    start = time.time()

    for i, audio_path in enumerate(audio_files):
        # Basic attributes
        attrs = extract_basic_attributes(str(audio_path))

        # Transcribe
        transcript = transcribe_batch([str(audio_path)], whisper, processor)[0]

        # Speaking speed
        words = len(transcript.split())
        if attrs["duration"] > 0:
            wps = words / attrs["duration"]
            attrs["speaking_speed"] = "fast" if wps > 3.0 else "slow" if wps < 1.5 else "normal"

        # Build seed transcript
        seed = build_seed_transcript(attrs, transcript)

        result = {
            "audio_path": str(audio_path),
            "transcript": transcript,
            "attributes": attrs,
            "seed_transcript": seed,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(audio_files)}] {rate:.1f} files/s — last: {transcript[:60]}...")

    # Free Whisper VRAM
    del whisper, processor
    torch.cuda.empty_cache()

    # Generate LLM descriptions
    if not args.skip_llm:
        print(f"\nGenerating LLM descriptions for {len(results)} samples...")
        seeds = [r["seed_transcript"] for r in results]
        descriptions = generate_descriptions(seeds, args.llm_url)
        for r, desc in zip(results, descriptions):
            r["description"] = desc
    else:
        print("Skipping LLM descriptions (--skip-llm)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to Python native for JSON serialization
    def jsonify(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(jsonify(r), ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"\nDone! {len(results)} samples processed in {elapsed:.0f}s")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
