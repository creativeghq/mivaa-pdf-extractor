"""One look over the discovered list before any money is spent on it (#513).

A model sees names, page ranges and printed codes only, and says which entries repeat a product, are
accessories, or wrongly fold several models into one. It suggests; code decides what applies. With a
review gate every suggestion waits for the operator; without one only a sure repeat merges.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from app.services.discovery.series_merge import absorb, series_key

logger = logging.getLogger(__name__)

TASK = "discovery_dedupe"
AUTO_MERGE_CONFIDENCE = 0.9
SAME_PRODUCT = "same_product"
ACCESSORY = "accessory"
MISLABELLED = "mislabelled_merge"
OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "verdict", "same_as", "confidence", "reason"],
        "properties": {
            "name": {"type": "string"},
            "verdict": {"type": "string", "enum": ["keep", SAME_PRODUCT, ACCESSORY, MISLABELLED]},
            "same_as": {"type": ["string", "null"]},
            "confidence": {"type": "number"},
            "reason": {"type": "string"},
        },
    }}},
}
BRACKETS = re.compile(r"\([^)]*\)")


def _model_tokens(name: str) -> List[str]:
    return series_key(BRACKETS.sub(" ", name or "")).split()


def distinct_models(a: str, b: str) -> bool:
    """LINEAR MAX and LINEAR MAX - M differ by a model letter: only a person may call those the same."""
    x, y = set(_model_tokens(a)), set(_model_tokens(b))
    return any(len(t) <= 3 for t in x ^ y)


def excluded_by_rules(name: str, exclude_names: Sequence[str]) -> bool:
    folded = series_key(name)
    return any(series_key(rule) and series_key(rule) in folded for rule in exclude_names)


def listing(products: Sequence[Any]) -> str:
    rows = []
    for p in products:
        meta = p.metadata or {}
        rows.append({"name": p.name, "pages": sorted(p.page_range or [])[:40],
                     "codes": (meta.get("product_codes") or meta.get("sku_codes") or [])[:12] if isinstance(meta.get("product_codes") or meta.get("sku_codes") or [], list) else []})
    return json.dumps(rows, ensure_ascii=False)


def apply_verdicts(products: List[Any], verdicts: Sequence[Dict[str, Any]], reviewed: bool) -> Dict[str, Any]:
    """Mark every suggestion on its product; merge in place only when nobody will review the list."""
    by_name = {p.name: p for p in products}
    summary: Dict[str, Any] = {"merged": [], "accessories": [], "mislabelled": [], "suggested_merges": []}
    for v in verdicts:
        product = by_name.get(v.get("name"))
        if not product or v.get("verdict") == "keep":
            continue
        meta = dict(product.metadata or {})
        kind, into = v["verdict"], by_name.get(v.get("same_as") or "")
        if kind == SAME_PRODUCT and (into is None or into is product):
            continue
        meta["discovery_suggestion"] = {"verdict": kind, "same_as": v.get("same_as"), "confidence": v.get("confidence"), "reason": v.get("reason")}
        product.metadata = meta
        if kind == ACCESSORY:
            summary["accessories"].append(product.name)
            meta["product_type"] = "accessory"
        elif kind == MISLABELLED:
            summary["mislabelled"].append(product.name)
        elif not reviewed and float(v.get("confidence") or 0) >= AUTO_MERGE_CONFIDENCE and not distinct_models(product.name, into.name):
            absorb(into, product)
            products.remove(product)
            del by_name[product.name]
            summary["merged"].append({"name": product.name, "into": into.name})
        else:
            summary["suggested_merges"].append({"name": product.name, "into": into.name})
    return summary


async def dedupe_discovered(products: List[Any], *, reviewed: bool, exclude_names: Sequence[str] = (), job_id: Optional[str] = None,
                            user_id: Optional[str] = None, workspace_id: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """Fills in suggestions on `products`, in place. A failed call changes nothing and says so."""
    from app.services.core.claude_models import SONNET
    from app.services.core.claude_tool_call import call_with_schema
    from app.services.utilities.prompt_registry import load_prompt, render

    excluded = [p.name for p in products if excluded_by_rules(p.name, exclude_names)]
    products[:] = [p for p in products if p.name not in excluded]
    summary: Dict[str, Any] = {"excluded_by_rules": excluded, "merged": [], "accessories": [], "mislabelled": [], "suggested_merges": []}
    if len(products) < 2:
        return summary
    try:
        prompt = await load_prompt("extraction", TASK, stage="discovery")
        result = await call_with_schema(
            task=TASK, model=model or SONNET, max_tokens=8000, job_id=job_id, user_id=user_id, workspace_id=workspace_id,
            messages=[{"role": "user", "content": render(prompt, products=listing(products))}],
            output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
        summary.update(apply_verdicts(products, result.data.get("verdicts") or [], reviewed))
    except Exception as e:
        logger.warning(f"discovery dedupe did not run: {type(e).__name__}: {e}")
        summary["failed"] = f"{type(e).__name__}: {str(e)[:200]}"
    return summary
