"""Shared helpers: chat completion, embeddings, prompt de-tokenization."""
from __future__ import annotations
import json, re, time, urllib.request

def normalize_prompt(t: str) -> str:
    """Undo Reddit-WritingPrompts tokenization (spaced punctuation/clitics)."""
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"\s+([,.!?;:’'])", r"\1", t)
    for cl in ["'ve", "'s", "'re", "'ll", "'d", "'m", "n't"]:
        t = t.replace(" " + cl, cl)
    return re.sub(r"[ \t]+", " ", t).strip()

def chat(endpoint: str, model: str, content: str, max_tokens=600,
         temperature=0.9, no_think=False, timeout=600):
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": temperature}
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    dt = time.time() - t0
    m = out["choices"][0]["message"]
    txt = (m.get("content") or m.get("reasoning_content") or "").strip()
    ctok = out.get("usage", {}).get("completion_tokens", 0)
    return txt, ctok, dt

def embed(texts, endpoint="http://localhost:8001/v1/embeddings",
          model="Qwen/Qwen3-Embedding-0.6B", batch=32):
    import numpy as np
    vecs = []
    for i in range(0, len(texts), batch):
        chunk = [t[:8000] for t in texts[i:i + batch]]
        body = {"model": model, "input": chunk}
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)
        vecs.extend([d["embedding"] for d in out["data"]])
    a = np.asarray(vecs, dtype="float64")
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)   # L2-normalize for RBF
    return a
