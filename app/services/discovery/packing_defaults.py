"""Packing a catalogue does not print: tile geometry, the same brand's size, other brands' size, the brand's kg/m².

A filled value is listed in the row's `estimated.fields` with its basis, a stated value is never
replaced, an estimate is never a source for another, and what stays unknown goes in `not_available`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SIBLING_KEYS = ("pieces_per_box", "m2_per_box", "weight_per_box_kg", "boxes_per_pallet", "weight_per_pallet_kg")
PALLET_KEYS = ("boxes_per_pallet", "weight_per_pallet_kg")
REPORTED_KEYS = ("pieces_per_box", "m2_per_box", "boxes_per_pallet", "m2_per_pallet", "weight_per_pallet_kg")
BOX_KEYS = ("pieces_per_box", "m2_per_box")
MIN_SOURCES = 2
MIN_SHARE = 0.75
MIN_PALLET_BRANDS = 2
GEOMETRY_TOLERANCE = 0.02
DENSITY_SPREAD = 0.15


def _sr() -> Any:
    from app.services.discovery import series_reconcile
    return series_reconcile


def _num(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("value")
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(str(value).replace(",", ".").strip())
    except ValueError:
        return None
    return out if out > 0 else None


def _clean(value: float) -> Any:
    return int(value) if value == int(value) else round(value, 2)


def format_key(text: Any) -> Optional[str]:
    sizes = sorted(_sr().sizes_in(str(text or "")))
    return sizes[0] if sizes else None


def same_size(a: str, b: str) -> bool:
    sr = _sr()
    return a == b or sr.nominal_of(a, b) or sr.nominal_of(b, a)


def _stated(row: Dict[str, Any], key: str) -> Optional[float]:
    if key in ((row.get("estimated") or {}).get("fields") or []):
        return None
    return _num(row.get(key))


def packing_rows(product: Dict[str, Any], formats: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Per-format rows; flat packing on a product with exactly one format is that format's row."""
    metadata = product.get("metadata") or {}
    packaging = metadata.get("packaging") or {}
    rows = [dict(r) for r in packaging.get("per_variant") or [] if isinstance(r, dict)]
    if rows or len(formats) != 1:
        return rows
    flat = {k: v for k in (*SIBLING_KEYS, "m2_per_pallet") if (v := packaging.get(k, metadata.get(k))) is not None and _num(v) is not None}
    return [{"format": formats[0], **{k: _num(v) for k, v in flat.items()}}] if flat else []


Pool = Sequence[Tuple[str, Dict[str, Any], str, Any]]


def sibling_pool(products: Iterable[Dict[str, Any]], formats_of=lambda p: ()) -> List[Tuple[str, Dict[str, Any], str, Any]]:
    pool = []
    for product in products:
        for row in packing_rows(product, formats_of(product)):
            fmt = format_key(row.get("format"))
            if fmt:
                pool.append((fmt, row, product["id"], product.get("brand_company_id")))
    return pool


def _agreed(values: Sequence[float]) -> Optional[Tuple[float, int]]:
    if not values:
        return None
    top, count = Counter(values).most_common(1)[0]
    return (top, count) if count / len(values) >= MIN_SHARE else None


def _product_votes(rows: Iterable[Tuple[Dict[str, Any], str]], key: str) -> Dict[str, set]:
    votes: Dict[str, set] = {}
    for r, pid in rows:
        if _stated(r, key) is not None:
            votes.setdefault(pid, set()).add((_stated(r, key), r.get("pallet")))
    return votes


def _single(votes: Dict[str, set]) -> List[float]:
    return [next(iter(vs))[0] for vs in votes.values() if len({v for v, _ in vs}) == 1]


def _mixed_pallets(votes: Dict[str, set]) -> bool:
    return len({pallet for vs in votes.values() for _, pallet in vs}) > 1


def brand_density(pool: Pool, brand: Any) -> Optional[float]:
    """The brand's kg per m², when its stated pallets agree within DENSITY_SPREAD."""
    per: Dict[str, float] = {}
    for _, r, pid, b in pool:
        kg, m2 = _stated(r, "weight_per_pallet_kg"), _stated(r, "m2_per_pallet")
        if b == brand and kg and m2:
            per[pid] = kg / m2
    ratios = sorted(per.values())
    if len(ratios) < MIN_SOURCES:
        return None
    mid = ratios[len(ratios) // 2]
    return mid if (ratios[-1] - ratios[0]) / mid <= DENSITY_SPREAD else None


def _matches_printed(n: int, area: float, m2: float) -> bool:
    if abs(n * area - m2) / m2 <= GEOMETRY_TOLERANCE:
        return True
    text = repr(m2)
    places = len(text.split(".")[1]) if "." in text else 0
    return places <= 1 and round(n * area, places) == m2


def plan_row(fmt: str, row: Dict[str, Any], pool: Pool, own_id: str, own_brand: Any = None) -> Tuple[Dict[str, Any], List[str]]:
    """The row with every value the evidence agrees on; returns it and the basis of each fill."""
    new = dict(row)
    new.setdefault("format", fmt)
    x, y = (float(v) for v in fmt.split("x"))
    area = x * y / 10000
    filled: List[str] = []
    basis: List[str] = []

    def _fill(key: str, value: float, why: str) -> None:
        new[key] = _clean(value)
        filled.append(key)
        basis.append(why)

    def geometry() -> None:
        pieces, m2 = _num(new.get("pieces_per_box")), _num(new.get("m2_per_box"))
        if m2 is None and pieces:
            _fill("m2_per_box", round(pieces * area, 2), f"m² per box: {_clean(pieces)} pieces × {round(area, 4)} m² per {fmt} tile")
        elif pieces is None and m2:
            n = round(m2 / area)
            if n >= 1 and _matches_printed(n, area, m2):
                _fill("pieces_per_box", n, f"pieces per box: {_clean(m2)} m² ÷ {round(area, 4)} m² per {fmt} tile")

    geometry()
    sized = [(r, pid, b) for f, r, pid, b in pool if pid != own_id and same_size(f, fmt)]
    m2 = _num(new.get("m2_per_box"))
    if m2 is not None and "m2_per_box" not in filled:
        narrowed = [x for x in sized if _stated(x[0], "m2_per_box") == m2]
        sized = narrowed or sized
    own = [(r, pid) for r, pid, b in sized if b == own_brand]
    for key in SIBLING_KEYS:
        if _num(new.get(key)) is not None:
            continue
        votes = _product_votes(own, key)
        if key in PALLET_KEYS and _mixed_pallets(votes):
            continue
        values = _single(votes)
        hit = _agreed(values) if len(values) >= MIN_SOURCES else None
        if hit:
            _fill(key, hit[0], f"{key.replace('_', ' ')}: {hit[1]} of {len(values)} same-brand {fmt} series")
    for key in (*BOX_KEYS, *PALLET_KEYS):
        if _num(new.get(key)) is not None:
            continue
        brand_votes: List[float] = []
        for brand in {b for _, _, b in sized if b != own_brand}:
            votes = _product_votes([(r, pid) for r, pid, b in sized if b == brand], key)
            if key in PALLET_KEYS and _mixed_pallets(votes):
                continue
            hit = _agreed(_single(votes))
            if hit:
                brand_votes.append(hit[0])
        need = MIN_PALLET_BRANDS if key in PALLET_KEYS else 1
        hit = _agreed(brand_votes) if len(brand_votes) >= need else None
        if hit:
            _fill(key, hit[0], f"{key.replace('_', ' ')}: {hit[1]} of {len(brand_votes)} other brands' {fmt}")
    geometry()
    box, boxes = _num(new.get("m2_per_box")), _num(new.get("boxes_per_pallet"))
    if _num(new.get("m2_per_pallet")) is None and box and boxes:
        _fill("m2_per_pallet", round(box * boxes, 2), "m² per pallet: m² per box × boxes per pallet")
    density = brand_density(pool, own_brand)
    if density:
        for key, base in (("weight_per_pallet_kg", "m2_per_pallet"), ("weight_per_box_kg", "m2_per_box")):
            if _num(new.get(key)) is None and _num(new.get(base)):
                _fill(key, round(_num(new.get(base)) * density, 1 if key == "weight_per_box_kg" else 0),
                      f"{key.replace('_', ' ')}: {base.replace('_', ' ')} × the brand's {round(density, 1)} kg/m²")
    return new, basis if filled else []


def plan_product(product: Dict[str, Any], formats: Sequence[str], pool: Pool) -> Optional[List[Dict[str, Any]]]:
    """The product's per-format packing with gaps filled, or None when nothing changes."""
    rows = packing_rows(product, formats)
    have = {format_key(r.get("format")) for r in rows}
    rows += [{"format": f} for f in formats if not any(h and same_size(h, f) for h in have)]
    changed = False
    out = []
    for row in rows:
        fmt = format_key(row.get("format"))
        if not fmt:
            out.append(row)
            continue
        new, basis = plan_row(fmt, row, pool, product["id"], product.get("brand_company_id"))
        if not basis:
            out.append(row)
            continue
        prior = row.get("estimated") or {}
        fields = list(dict.fromkeys(list(prior.get("fields") or []) + [k for k in new if k in (*SIBLING_KEYS, *REPORTED_KEYS) and new.get(k) != row.get(k)]))
        new["estimated"] = {"fields": fields, "basis": "; ".join(filter(None, [prior.get("basis"), *basis])),
                            "method": "tile geometry, same brand's size, other brands' size, brand kg/m²",
                            "at": datetime.now(timezone.utc).isoformat()}
        missing = [k for k in REPORTED_KEYS if _num(new.get(k)) is None]
        if missing:
            new["not_available"] = missing
        else:
            new.pop("not_available", None)
        changed = True
        out.append(new)
    out = [r for r in out if any(_num(r.get(k)) is not None for k in (*SIBLING_KEYS, *REPORTED_KEYS))]
    return out if changed and out else None


async def fill_packing_defaults(supabase: Any, document_id: str, *, apply: bool = True, job_id: Optional[str] = None) -> Dict[str, Any]:
    from app.services.discovery.series_facts import _all, verified_formats
    from app.services.pdf.ingestion_corrections import CORRECTED, Correction, record

    db = getattr(supabase, "client", supabase)
    columns = "id, name, workspace_id, brand_company_id, metadata"
    products = _all(lambda: db.table("products").select(columns).eq("source_document_id", document_id).order("id"))
    summary: Dict[str, Any] = {"products": len(products), "filled": 0, "per_product": []}
    if not products:
        return summary
    workspace = products[0].get("workspace_id")
    siblings = {p["id"]: p for p in products}
    for p in _all(lambda: db.table("products").select(columns).eq("workspace_id", workspace)
                  .filter("metadata->packaging", "not.is", "null").order("id")):
        siblings.setdefault(p["id"], p)

    def formats_of(p: Dict[str, Any]) -> List[str]:
        return verified_formats(p.get("metadata") or {}, [])

    pool = sibling_pool(siblings.values(), formats_of)
    for product in products:
        planned = plan_product(product, formats_of(product), pool)
        if planned is None:
            continue
        summary["filled"] += 1
        summary["per_product"].append(product["name"])
        if apply:
            metadata = dict(product.get("metadata") or {})
            metadata["packaging"] = {**(metadata.get("packaging") or {}), "per_variant": planned}
            db.table("products").update({"metadata": metadata}).eq("id", product["id"]).execute()
    if apply and summary["filled"]:
        await record(db, document_id, workspace, "series_facts", [Correction(
            "packing_filled_from_same_size", CORRECTED,
            f"{summary['filled']} product(s) took missing packing from tile geometry, the same or other brands' identical size, or the brand's kg/m², each marked estimated",
            summary["filled"], summary["per_product"])], job_id)
    return summary
