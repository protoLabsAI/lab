"""Lightweight embedding + reranker server using transformers directly.

Serves Qwen3-Embedding and Qwen3-Reranker via OpenAI-compatible endpoints.
Much simpler than vLLM pooling (which has serialization bugs in 0.18).

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python experiments/rag-bench/serve_embed.py
    CUDA_VISIBLE_DEVICES=1 uv run python experiments/rag-bench/serve_embed.py --model Qwen/Qwen3-Embedding-4B
"""

from __future__ import annotations

import argparse
import time
import os

os.environ.setdefault("HF_HOME", "/mnt/models/huggingface")

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI(title="Embedding Server")

# Globals set at startup
_model = None
_tokenizer = None
_model_name = None
_device = None


class EmbedRequest(BaseModel):
    model: str = ""
    input: str | list[str] = ""


class EmbedResponse(BaseModel):
    object: str = "list"
    data: list[dict]
    model: str
    usage: dict


class RerankRequest(BaseModel):
    model: str = ""
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankResponse(BaseModel):
    results: list[dict]
    model: str


def _encode(texts: list[str], max_length: int = 2048) -> np.ndarray:
    """Encode texts to normalized embeddings."""
    inputs = _tokenizer(
        texts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        # Use last hidden state at EOS token position
        embeddings = outputs.last_hidden_state[:, -1, :]

    # Normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().float().numpy()


def _rerank_score(query: str, documents: list[str]) -> list[float]:
    """Score query-document pairs for reranking."""
    pairs = [[query, doc] for doc in documents]
    # Tokenize as pairs
    inputs = _tokenizer(
        pairs, padding=True, truncation=True, max_length=2048,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        # Use CLS/last token logit as relevance score
        scores = outputs.last_hidden_state[:, -1, 0]  # First dim of last token

    return scores.cpu().float().tolist()


@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input

    t0 = time.time()
    embs = _encode(texts)
    elapsed = time.time() - t0

    data = []
    for i, emb in enumerate(embs):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": emb.tolist(),
        })

    total_tokens = sum(len(_tokenizer.encode(t)) for t in texts)

    return EmbedResponse(
        data=data,
        model=_model_name,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens,
               "elapsed_ms": int(elapsed * 1000)},
    )


@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    scores = _rerank_score(req.query, req.documents)

    results = []
    for i, (score, doc) in enumerate(zip(scores, req.documents)):
        results.append({
            "index": i,
            "relevance_score": score,
            "document": {"text": doc[:200]},
        })

    # Sort by score descending
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if req.top_n:
        results = results[:req.top_n]

    return RerankResponse(results=results, model=_model_name)


@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_name}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": _model_name, "object": "model"}]}


def main():
    global _model, _tokenizer, _model_name, _device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    _model_name = args.model
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} on {_device}...")
    _tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16,
    ).to(_device).eval()
    print(f"Model loaded. Serving on :{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
