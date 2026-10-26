"""Which registry fields of a draft product are still empty, and where could they be read from."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

#: Why a product is in the report. One verdict per product, never a bare number.
NOTHING_MISSING = "nothing_missing"
GAPS_WITH_SOURCE = "gaps_with_source"
GAPS_NO_SOURCE = "gaps_no_source"


def empty_fields(
    flat: Dict[str, Any],
    specs: Iterable[Any],
    category: Optional[str],
    is_empty: Callable[[Any], bool],
    is_internal: Callable[[str], bool],
) -> List[str]:
    """Descriptive, public, metadata-destined registry fields with no value in `flat`.

    Identity fields (colour, size, finish) are excluded: a product page often describes a
    sibling variant, and a confident wrong identity value is the one nothing downstream catches.
    Column-destined fields are excluded because they are not metadata keys.
    """
    out: List[str] = []
    for spec in specs:
        if spec.destination != "metadata" or spec.sensitivity != "public":
            continue
        if spec.role_in(category) != "descriptive" or is_internal(spec.name):
            continue
        if is_empty(flat.get(spec.name)):
            out.append(spec.name)
    return out


def source_for(product_url: Optional[str], brand_website: Optional[str]) -> Optional[Dict[str, str]]:
    """The URL we already hold. Never a search by name: that is a guessed match."""
    for kind, url in (("product_url", product_url), ("brand_site", brand_website)):
        if isinstance(url, str) and url.strip().lower().startswith("https://"):
            return {"kind": kind, "url": url.strip()}
    return None


def verdict(gaps: List[str], source: Optional[Dict[str, str]]) -> str:
    if not gaps:
        return NOTHING_MISSING
    return GAPS_WITH_SOURCE if source else GAPS_NO_SOURCE


def _chunks(ids: List[str], size: int = 100) -> Iterable[List[str]]:
    """An `in_` list rides in the URL, so a few hundred ids overflows it."""
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


async def _internal_pattern(supabase: Any) -> "re.Pattern[str]":
    """The cost-field name floor, fetched from SQL so it is never restated here."""
    resp = await asyncio.to_thread(supabase.client.rpc("internal_product_field_pattern").execute)
    pattern = resp.data
    if not isinstance(pattern, str) or not pattern:
        raise RuntimeError("internal_product_field_pattern returned nothing")
    return re.compile(pattern)


async def product_gaps(supabase: Any, product_ids: List[str], workspace_id: str) -> List[Dict[str, Any]]:
    """One verdict row per product in `workspace_id`. Raises rather than guessing on a failed read."""
    from app.api.pdf_processing.stage_4_products import (
        _category_of,
        _flatten_for_canonicalization,
        _is_empty_value,
    )
    from app.services.metadata.field_registry import field_registry

    if not product_ids:
        return []
    await field_registry.ensure_loaded()
    internal = await _internal_pattern(supabase)

    products: List[Dict[str, Any]] = []
    for chunk in _chunks(product_ids):
        rows = await asyncio.to_thread(
            supabase.client.table("products")
            .select("id, name, metadata, product_url, brand_company_id")
            .eq("workspace_id", workspace_id)
            .in_("id", chunk)
            .execute
        )
        products.extend(rows.data or [])

    brand_ids = sorted({p["brand_company_id"] for p in products if p.get("brand_company_id")})
    websites: Dict[str, str] = {}
    for chunk in _chunks(brand_ids):
        comp = await asyncio.to_thread(
            supabase.client.table("crm_companies").select("id, website").in_("id", chunk).execute
        )
        websites.update({c["id"]: c.get("website") for c in (comp.data or [])})

    report: List[Dict[str, Any]] = []
    for p in products:
        metadata = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}
        category = field_registry.resolve_category(_category_of(metadata))
        if not category:
            report.append({
                "product_id": p["id"], "name": p.get("name"), "category": None, "gaps": [],
                "source": None, "verdict": "unclassified",
            })
            continue
        flat = _flatten_for_canonicalization(metadata, category)
        gaps = empty_fields(
            flat, field_registry.fields_for_category(category), category,
            _is_empty_value, lambda name: bool(internal.search(name)),
        )
        source = source_for(p.get("product_url"), websites.get(p.get("brand_company_id")))
        report.append({
            "product_id": p["id"], "name": p.get("name"), "category": category, "gaps": gaps,
            "source": source, "verdict": verdict(gaps, source),
        })
    return report


def summarize(report: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"products": len(report), NOTHING_MISSING: 0, GAPS_WITH_SOURCE: 0, GAPS_NO_SOURCE: 0, "unclassified": 0}
    for row in report:
        counts[row["verdict"]] += 1
    return counts
