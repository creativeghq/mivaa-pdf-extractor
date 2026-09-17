"""Market price resolution. The derivation itself lives in SQL.

`resolve_market_price_from_hits` is the one place that decides what a product is
worth; this module only shapes hits for it and records that somebody asked.
"""

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

EMPTY: Dict[str, Any] = {
    "status": "no_data",
    "chosen_price": None,
    "chosen_basis": None,
    "chosen_retailer": None,
    "chosen_url": None,
    "chosen_title": None,
    "min": None,
    "max": None,
    "median": None,
    "currency": None,
    "sample_size": 0,
    "verified_count": 0,
    "in_stock_count": 0,
    "confidence": "none",
}


def _failed() -> Dict[str, Any]:
    """Unknown, not zero: a caller must not read a DB outage as 'no market price'."""
    out = dict(EMPTY)
    out["status"] = "collector_failed"
    return out


def hits_to_payload(hits: Iterable[Any]) -> List[Dict[str, Any]]:
    """Flatten PriceHit objects (or plain dicts) into the resolver's input shape.

    Args:
        hits: PriceHit models or mappings carrying at least a `price`.

    Returns:
        A list of plain dicts safe to pass as jsonb.
    """
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        get = h.get if isinstance(h, dict) else lambda k, _h=h: getattr(_h, k, None)
        price = get("price")
        if price is None:
            continue
        out.append({
            "price": float(price),
            "currency": get("currency"),
            "availability": get("availability"),
            "verified": bool(get("verified") or False),
            "match_kind": get("match_kind"),
            "is_anomaly": bool(get("is_anomaly") or False),
            "retailer_name": get("retailer_name"),
            "product_url": get("product_url"),
            "product_title": get("product_title"),
        })
    return out


def resolve_from_hits(sb, hits: Iterable[Any]) -> Dict[str, Any]:
    """Derive the market answer for a set of hits via the SQL resolver.

    Args:
        sb: A supabase client.
        hits: PriceHit models or mappings.

    Returns:
        The resolver dict. Status is `no_data` when there was nothing to derive from and
        `collector_failed` when the derivation could not be run.
    """
    payload = hits_to_payload(hits)
    if not payload:
        return dict(EMPTY)
    try:
        res = sb.rpc("resolve_market_price_from_hits", {"p_hits": payload}).execute()
        if isinstance(res.data, dict):
            return res.data
        logger.warning("resolve_market_price_from_hits returned %r", type(res.data))
    except Exception as e:
        logger.error("market price resolver failed: %s", e)
    return _failed()


def record_demand(
    sb,
    *,
    tracked_query_id: Optional[str] = None,
    product_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    surface: str = "internal",
    served_from: Optional[str] = None,
) -> None:
    """Record that this price was asked for. Never raises: demand is telemetry."""
    if not tracked_query_id and not product_id:
        return
    try:
        sb.rpc("record_price_demand", {
            "p_tracked_query_id": tracked_query_id,
            "p_product_id": product_id,
            "p_workspace_id": workspace_id,
            "p_user_id": user_id,
            "p_api_key_id": api_key_id,
            "p_surface": surface,
            "p_served_from": served_from,
        }).execute()
    except Exception as e:
        logger.debug("record_price_demand failed (non-fatal): %s", e)
