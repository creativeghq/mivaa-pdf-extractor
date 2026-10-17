"""Request planning and response checking for Voyage contextualized chunk embeddings.

Import-free so the unit suite can load it by path.
"""

from typing import Dict, List, Optional, Tuple

#: voyage-context-4 takes at most 32K tokens per document and does not truncate.
WINDOW_TOKENS = 12000
REQUEST_TOKENS = 60000
MAX_CHUNK_CHARS = 24000
DIMENSIONS = 1024
MAX_ATTEMPTS = 3


def is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def backoff_seconds(attempt: int) -> float:
    return 2.0 * (2 ** attempt)


def estimate_tokens(text: str) -> int:
    # Deliberately pessimistic: Greek runs ~2 chars/token, and an underestimate is a 400.
    return len(text) // 2 + 20


def prepare(texts: List[Optional[str]]) -> List[str]:
    return [((t or "").strip() or "no text content")[:MAX_CHUNK_CHARS] for t in texts]


def windows(texts: List[str]) -> List[List[int]]:
    """Consecutive runs of chunk indices, each under the per-document token budget."""
    out: List[List[int]] = []
    cur: List[int] = []
    cur_tokens = 0
    for i, text in enumerate(texts):
        t = estimate_tokens(text)
        if cur and cur_tokens + t > WINDOW_TOKENS:
            out.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += t
    if cur:
        out.append(cur)
    return out


def plan_requests(texts: List[str]) -> List[List[List[int]]]:
    """Group windows into requests under the per-request token budget."""
    requests: List[List[List[int]]] = []
    cur: List[List[int]] = []
    cur_tokens = 0
    for w in windows(texts):
        wt = sum(estimate_tokens(texts[i]) for i in w)
        if cur and cur_tokens + wt > REQUEST_TOKENS:
            requests.append(cur)
            cur, cur_tokens = [], 0
        cur.append(w)
        cur_tokens += wt
    if cur:
        requests.append(cur)
    return requests


def assign_vectors(request: List[List[int]], data: Dict) -> List[Tuple[int, List[float]]]:
    """Map a response onto chunk indices, or raise if any window does not line up."""
    docs = sorted(data.get("data") or [], key=lambda d: int(d["index"]))
    if [int(d["index"]) for d in docs] != list(range(len(request))):
        raise ValueError("contextualized response documents are not a complete 0..n-1 run")
    pairs: List[Tuple[int, List[float]]] = []
    for w, doc in zip(request, docs):
        items = sorted(doc.get("data") or [], key=lambda it: int(it["index"]))
        if [int(it["index"]) for it in items] != list(range(len(w))):
            raise ValueError(
                f"contextualized response returned {len(items)} chunks for a "
                f"{len(w)}-chunk window -- refusing to misalign them"
            )
        for i, it in zip(w, items):
            emb = it.get("embedding")
            if not isinstance(emb, list) or len(emb) != DIMENSIONS:
                raise ValueError(f"contextualized response vector is not {DIMENSIONS}D")
            pairs.append((i, emb))
    return pairs
