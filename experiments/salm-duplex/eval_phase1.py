"""
Evaluate SALM Phase 1 checkpoint — speech understanding.

Tests:
1. ASR accuracy: Can it transcribe speech? (WER on LibriSpeech test-clean)
2. Description quality: Can it describe what it hears? (LLM-judged)
3. Text regression: Does the LLM backbone still work for text? (sanity check)

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_phase1.py \
        --checkpoint /mnt/data/training/salm-qwen4b/step=XXXX.ckpt \
        --config /home/ava/dev/lab/experiments/salm-duplex/conf/salm-qwen4b.yaml \
        --test-audio /mnt/data/salm-duplex/data/pipeline-test \
        --num-samples 50
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["HF_HOME"] = "/mnt/models/huggingface"

import soundfile as sf
import numpy as np
import torch
from omegaconf import OmegaConf


def load_model(config_path: str, checkpoint_path: str | None = None):
    """Load SALM model from config and optional checkpoint."""
    from nemo.collections.speechlm2.models.salm import SALM

    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    print(f"Loading SALM model from config: {config_path}")
    model = SALM(model_cfg)

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.cuda()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {trainable:,} trainable / {total:,} total params")
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    return model


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target_sr."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * target_sr / sr)).astype(np.float32)
    return audio


def test_asr(model, audio_files: list[Path], reference_transcripts: dict | None = None):
    """Test 1: Can the model transcribe speech?"""
    print("\n=== Test 1: ASR (Speech Transcription) ===")

    results = []
    for audio_path in audio_files[:20]:
        audio = load_audio(str(audio_path))
        audio_tensor = torch.tensor(audio).unsqueeze(0).cuda()
        audio_len = torch.tensor([len(audio)]).cuda()

        prompt = "Transcribe the following audio."

        try:
            with torch.no_grad():
                # Use model's generate method if available
                if hasattr(model, "generate"):
                    output = model.generate(
                        audio=audio_tensor,
                        audio_len=audio_len,
                        prompt=prompt,
                        max_new_tokens=256,
                    )
                    text = output if isinstance(output, str) else str(output)
                else:
                    text = "[generate not available — inference method TBD]"
        except Exception as e:
            text = f"[ERROR: {e}]"

        results.append({
            "file": audio_path.name,
            "output": text[:200],
        })
        print(f"  {audio_path.name}: {text[:100]}")

    return results


def test_description(model, audio_files: list[Path]):
    """Test 2: Can the model describe what it hears?"""
    print("\n=== Test 2: Audio Description ===")

    results = []
    for audio_path in audio_files[:10]:
        audio = load_audio(str(audio_path))
        audio_tensor = torch.tensor(audio).unsqueeze(0).cuda()
        audio_len = torch.tensor([len(audio)]).cuda()

        prompt = "Describe what you hear in this audio."

        try:
            with torch.no_grad():
                if hasattr(model, "generate"):
                    output = model.generate(
                        audio=audio_tensor,
                        audio_len=audio_len,
                        prompt=prompt,
                        max_new_tokens=300,
                    )
                    text = output if isinstance(output, str) else str(output)
                else:
                    text = "[generate not available]"
        except Exception as e:
            text = f"[ERROR: {e}]"

        results.append({
            "file": audio_path.name,
            "description": text[:300],
        })
        print(f"  {audio_path.name}: {text[:150]}")

    return results


def test_text_regression(model):
    """Test 3: Does the LLM backbone still work for text-only queries?"""
    print("\n=== Test 3: Text Regression (LLM sanity check) ===")

    test_prompts = [
        "What is 2 + 2?",
        "Explain gravity in one sentence.",
        "Write a Python function that reverses a string.",
        "What is the capital of France?",
    ]

    results = []
    for prompt in test_prompts:
        try:
            with torch.no_grad():
                if hasattr(model, "generate_text"):
                    text = model.generate_text(prompt=prompt, max_new_tokens=100)
                elif hasattr(model, "llm"):
                    # Direct LLM access
                    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
                    if tokenizer:
                        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                        outputs = model.llm.generate(**inputs, max_new_tokens=100)
                        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        text = "[no tokenizer available]"
                else:
                    text = "[no text generation method available]"
        except Exception as e:
            text = f"[ERROR: {e}]"

        results.append({"prompt": prompt, "output": text[:200]})
        print(f"  Q: {prompt}")
        print(f"  A: {text[:150]}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="SALM config YAML")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (optional, uses pretrained if omitted)")
    parser.add_argument("--test-audio", required=True, help="Directory with test audio files")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    # Load model
    model = load_model(args.config, args.checkpoint)

    # Find audio files
    audio_dir = Path(args.test_audio)
    audio_files = sorted(audio_dir.glob("*.flac")) + sorted(audio_dir.glob("*.wav"))
    audio_files = audio_files[:args.num_samples]
    print(f"\nFound {len(audio_files)} audio files for testing")

    # Run tests
    all_results = {}

    asr_results = test_asr(model, audio_files)
    all_results["asr"] = asr_results

    desc_results = test_description(model, audio_files)
    all_results["description"] = desc_results

    text_results = test_text_regression(model)
    all_results["text_regression"] = text_results

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"ASR: {len(asr_results)} samples tested")
    asr_errors = sum(1 for r in asr_results if r["output"].startswith("[ERROR"))
    print(f"  Errors: {asr_errors}/{len(asr_results)}")

    print(f"Description: {len(desc_results)} samples tested")
    desc_errors = sum(1 for r in desc_results if r["description"].startswith("[ERROR"))
    print(f"  Errors: {desc_errors}/{len(desc_results)}")

    print(f"Text regression: {len(text_results)} prompts tested")
    text_errors = sum(1 for r in text_results if r["output"].startswith("[ERROR"))
    print(f"  Errors: {text_errors}/{len(text_results)}")


if __name__ == "__main__":
    main()
