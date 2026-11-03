"""Search relevance evaluation: does a query find the product it describes?

Text cases live in `search_text_cases.json`; photo cases are one image per product, degraded the
same fixed way, so a hit is a near-duplicate found rather than an exact vector looked up. A
capture runs each case once through the app's path with a wide pool and no threshold, keeping
every candidate's per-channel scores, so fusion variants are compared offline for free.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

CHANNELS = ("visual", "chunk", "understanding", "product", "keyword", "color", "texture", "style", "material", "page")
CAPTURE_POOL = 60


# ── Cases ───────────────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: str
    kind: str
    query: str
    expected: List[str]
    acceptable: List[str]
    image_id: Optional[str] = None
    image_url: Optional[str] = None


def load_text_cases(path: str) -> List[Case]:
    data = json.load(open(path, encoding="utf-8"))
    return [Case(c["id"], c["kind"], c["query"], list(c["expected"]), list(c.get("acceptable") or [])) for c in data["cases"]]


def degrade(image_bytes: bytes) -> str:
    """The fixed 'phone photo' treatment, as base64 JPEG."""
    from PIL import Image
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = im.size
    im = im.crop((int(w * .08), int(h * .10), int(w * .90), int(h * .88))).rotate(4, fillcolor=(200, 200, 200))
    im.thumbnail((640, 640))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode()


def build_photo_cases(db: Any, workspace_id: str, limit: int) -> List[Case]:
    """One live, embedded image per product, picked deterministically, products in a stable shuffled order."""
    rows = (db.table("image_product_associations")
            .select("product_id, image_id, document_images!inner(image_url, has_slig_embedding, metadata)")
            .eq("workspace_id", workspace_id).eq("document_images.has_slig_embedding", True)
            .is_("document_images.metadata->excluded_from_products", "null")
            .limit(10000).execute().data or [])
    by_product: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if (r.get("document_images") or {}).get("image_url"):
            by_product.setdefault(r["product_id"], []).append(r)
    picks = []
    for pid, imgs in by_product.items():
        img = min(imgs, key=lambda r: hashlib.sha1(f"{pid}:{r['image_id']}".encode()).hexdigest())
        picks.append((hashlib.sha1(pid.encode()).hexdigest(), pid, img))
    picks.sort()
    return [Case(f"p{i:03d}", "photo", "", [pid], [], image_id=img["image_id"], image_url=img["document_images"]["image_url"])
            for i, (_, pid, img) in enumerate(picks[:limit])]


# ── Capture (live, paid) ────────────────────────────────────────────────────────

async def capture_case(rag: Any, case: Case, workspace_id: str, extra_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from app.services.search.unified_search_service import UnifiedSearchService
    from app.services.search.weight_profiles import text_query_source_weights

    query, filters, profile, weights = case.query, None, "balanced", None
    image_b64 = None
    if case.kind == "photo":
        import httpx
        async with httpx.AsyncClient(timeout=30) as http:
            image_b64 = degrade((await http.get(case.image_url)).content)
    else:
        _, _, profile, _ = await UnifiedSearchService()._parse_query_with_ai(case.query, workspace_id=workspace_id)
        weights = text_query_source_weights()  # what /api/rag/search uses for words with no image
    sc = {**({"weights": weights} if weights else {}), **(extra_config or {})} or None
    out = await rag.multi_vector_search(
        query=query, workspace_id=workspace_id, top_k=CAPTURE_POOL, material_filters=filters or None,
        similarity_threshold=0.0, search_config=sc, image_base64=image_b64,
    )
    weights_used = (out.get("search_config") or {}).get("weights") or weights
    return {
        "case": case.__dict__, "rewritten_query": query, "filters": filters, "profile": profile,
        "weights": weights_used, "channels_failed": out.get("channels_failed"),
        "candidates": [
            {"id": r["id"], "name": r.get("product_name"), "score": r.get("score"),
             **{c: float(r.get(f"{c}_score") or 0.0) for c in CHANNELS}}
            for r in out.get("results", [])
        ],
    }


# ── Fusion (offline) ────────────────────────────────────────────────────────────

Fusion = Callable[[List[Dict[str, Any]], Dict[str, float]], List[str]]


def fuse_active_mean(cands: List[Dict[str, Any]], w: Dict[str, float]) -> List[str]:
    """Production today: weighted mean over the channels where the product scored > 0."""
    def score(c):
        num = sum(c[k] * w.get(k, 0) for k in CHANNELS if c[k] > 0)
        den = sum(w.get(k, 0) for k in CHANNELS if c[k] > 0)
        return num / den if den else 0.0
    return [c["id"] for c in sorted(cands, key=score, reverse=True)]


def fuse_weighted_sum(cands: List[Dict[str, Any]], w: Dict[str, float]) -> List[str]:
    """A channel that found nothing contributes 0: more independent evidence ranks higher."""
    return [c["id"] for c in sorted(cands, key=lambda c: sum(c[k] * w.get(k, 0) for k in CHANNELS), reverse=True)]


def fuse_rrf(cands: List[Dict[str, Any]], w: Dict[str, float], k: int = 60) -> List[str]:
    """Weighted reciprocal-rank fusion: robust to channels whose raw scores are on different scales."""
    total: Dict[str, float] = {c["id"]: 0.0 for c in cands}
    for ch in CHANNELS:
        ranked = sorted((c for c in cands if c[ch] > 0), key=lambda c: c[ch], reverse=True)
        for rank, c in enumerate(ranked, 1):
            total[c["id"]] += w.get(ch, 0) / (k + rank)
    return sorted(total, key=total.get, reverse=True)


FUSIONS: Dict[str, Fusion] = {"active_mean": fuse_active_mean, "weighted_sum": fuse_weighted_sum, "rrf": fuse_rrf}


# ── Metrics ─────────────────────────────────────────────────────────────────────

def first_hit(ranked: Sequence[str], good: Sequence[str]) -> Optional[int]:
    goods = set(good)
    return next((i + 1 for i, pid in enumerate(ranked) if pid in goods), None)


def metrics(ranks: Sequence[Optional[int]]) -> Dict[str, float]:
    n = len(ranks) or 1
    return {
        "n": len(ranks),
        "hit@1": round(sum(1 for r in ranks if r and r <= 1) / n, 3),
        "hit@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
        "hit@10": round(sum(1 for r in ranks if r and r <= 10) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
    }


def evaluate(captures: List[Dict[str, Any]], fusion: str, weights_override: Optional[Callable] = None) -> Dict[str, Any]:
    """Rank every capture with `fusion`; a hit is an EXPECTED id (acceptable counts for hit@10 'lenient')."""
    fn = FUSIONS[fusion]
    per_kind: Dict[str, List[Optional[int]]] = {}
    lenient: List[Optional[int]] = []
    rows = []
    for cap in captures:
        case = cap["case"]
        w = (weights_override(cap) if weights_override else None) or cap.get("weights") or {}
        ranked = fn(cap["candidates"], w)
        r = first_hit(ranked, case["expected"])
        per_kind.setdefault(case["kind"], []).append(r)
        lenient.append(first_hit(ranked, case["expected"] + case["acceptable"]))
        rows.append({"id": case["id"], "kind": case["kind"], "query": case["query"], "rank": r,
                     "top3": [next((c["name"] for c in cap["candidates"] if c["id"] == pid), pid) for pid in ranked[:3]]})
    all_ranks = [r for rs in per_kind.values() for r in rs]
    return {"fusion": fusion, "overall": metrics(all_ranks), "lenient_overall": metrics(lenient),
            "by_kind": {k: metrics(v) for k, v in sorted(per_kind.items())}, "cases": rows}


def recall_ceiling(captures: List[Dict[str, Any]]) -> Dict[str, float]:
    """Share of cases whose expected product is in the candidate pool at all: no fusion can beat it."""
    found = [any(c["id"] in cap["case"]["expected"] for c in cap["candidates"]) for cap in captures]
    return {"in_pool": round(sum(found) / (len(found) or 1), 3), "n": len(found)}

