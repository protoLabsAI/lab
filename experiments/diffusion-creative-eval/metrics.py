"""Rosmine-style text-distribution metrics: Token-L2, MMD, slop signs, self-BLEU.

- Token-L2: L2 distance between 1-gram token frequency distributions (word choice).
- MMD: Maximum Mean Discrepancy over embeddings (content/distribution similarity).
- slop signs: deterministic count of em-dashes + stock phrases + "not X, it's Y".
- self-BLEU: intra-set 4-gram overlap (diversity; lower = more diverse).
"""
from __future__ import annotations
import itertools, re
from collections import Counter
import numpy as np

WORD = re.compile(r"[a-z0-9']+")
BANNED = ["it's not just", "isn't just", "a testament to", "the void", "carved out",
          "tapestry", "delve", "in a world", "little did", "sent shivers",
          "couldn't help", "palpable", "symphony of", "dance of", "tang of",
          "whispered promises", "kaleidoscope", "indelible", "myriad"]

def toks(t): return WORD.findall(t.lower())

def token_l2(texts_a, texts_b):
    """L2 distance between normalized 1-gram frequency distributions."""
    ca, cb = Counter(), Counter()
    for t in texts_a: ca.update(toks(t))
    for t in texts_b: cb.update(toks(t))
    na, nb = sum(ca.values()) or 1, sum(cb.values()) or 1
    vocab = set(ca) | set(cb)
    return float(np.sqrt(sum((ca[w]/na - cb[w]/nb) ** 2 for w in vocab)))

def slop_rate(texts):
    """Slop signs per 1000 words (lower = cleaner)."""
    em = notxy = phr = words = 0
    for t in texts:
        tl = t.lower()
        em += t.count("—") + len(re.findall(r"\s-\s", t))
        notxy += len(re.findall(r"\bnot [^,.\n]{1,40}[,—]\s*(it'?s|but|just)", tl))
        phr += sum(tl.count(p) for p in BANNED)
        words += len(toks(t))
    per1k = 1000.0 / max(words, 1)
    return {"em_dash": em, "not_x_y": notxy, "phrases": phr,
            "total_per_1k": round((em + notxy + phr) * per1k, 2)}

def self_bleu(texts, n=4):
    def ng(t):
        w = toks(t); return set(tuple(w[i:i+n]) for i in range(len(w)-n+1))
    gs = [ng(t) for t in texts]
    ov = []
    for a, b in itertools.combinations(gs, 2):
        u = len(a | b); ov.append(len(a & b) / u if u else 0.0)
    return round(float(np.mean(ov)) if ov else 0.0, 4)

def _sq_dists(X, Y):
    xn = (X ** 2).sum(1)[:, None]; yn = (Y ** 2).sum(1)[None, :]
    d = xn + yn - 2 * X @ Y.T
    return np.maximum(d, 0)

def mmd2(X, Y):
    """Unbiased MMD^2 with RBF kernel, median-heuristic bandwidth. ~0 = same dist."""
    Kxx, Kyy, Kxy = _sq_dists(X, X), _sq_dists(Y, Y), _sq_dists(X, Y)
    med = np.median(np.concatenate([Kxx[np.triu_indices_from(Kxx, 1)],
                                    Kyy[np.triu_indices_from(Kyy, 1)],
                                    Kxy.ravel()]))
    gamma = 1.0 / (med + 1e-12)
    kxx, kyy, kxy = np.exp(-gamma * Kxx), np.exp(-gamma * Kyy), np.exp(-gamma * Kxy)
    m, n = len(X), len(Y)
    np.fill_diagonal(kxx, 0); np.fill_diagonal(kyy, 0)
    term = (kxx.sum() / (m*(m-1)) + kyy.sum() / (n*(n-1)) - 2 * kxy.mean())
    return float(max(term, 0.0))
