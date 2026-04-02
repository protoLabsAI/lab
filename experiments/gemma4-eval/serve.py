"""Lightweight OpenAI-compatible server for Gemma 4 models.

Serves Gemma 4 via transformers (vLLM doesn't support gemma4 arch yet).
Provides /v1/chat/completions for eval suite compatibility.

Usage:
    # MoE (26B-A4B, 4B active — fast)
    CUDA_VISIBLE_DEVICES=0 ~/dev/gemma4-env/bin/python experiments/gemma4-eval/serve.py \
      --model google/gemma-4-26B-A4B-it --port 8000

    # Dense 31B (TP=2 via device_map=auto)
    ~/dev/gemma4-env/bin/python experiments/gemma4-eval/serve.py \
      --model google/gemma-4-31B-it --port 8000

    # Edge 4B
    CUDA_VISIBLE_DEVICES=1 ~/dev/gemma4-env/bin/python experiments/gemma4-eval/serve.py \
      --model google/gemma-4-E4B-it --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

app = FastAPI(title="Gemma 4 Server")

_model = None
_processor = None
_tokenizer = None
_model_name = None
_device = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = 4096
    temperature: float = 0.0
    stop: list[str] | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Format messages — use tokenizer directly (avoid processor multimodal parsing)
    chat = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    text = _tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=max(req.temperature, 0.01),
            do_sample=req.temperature > 0,
        )
    gen_time = time.time() - t0

    # Decode only new tokens
    new_tokens = outputs[0][input_len:]
    text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
    output_tokens = len(new_tokens)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=_model_name,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": input_len,
            "completion_tokens": output_tokens,
            "total_tokens": input_len + output_tokens,
            "gen_time_s": round(gen_time, 2),
            "tok_per_s": round(output_tokens / gen_time, 1) if gen_time > 0 else 0,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": _model_name, "object": "model"}]}


def main():
    global _model, _processor, _tokenizer, _model_name, _device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    _model_name = args.model
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading {args.model}...")
    _tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        _processor = AutoProcessor.from_pretrained(args.model)
    except Exception:
        _processor = None

    _model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto",
    )
    _model.eval()

    params_b = sum(p.numel() for p in _model.parameters()) / 1e9
    print(f"Model loaded: {params_b:.1f}B params, dtype={dtype}")
    print(f"Serving on :{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
