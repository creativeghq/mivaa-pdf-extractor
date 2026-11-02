"""One discovered product per series, across discovery batches.

A catalogue organised by format names a series once per chapter ("Amalfi 30x60", "Amalfi 33x33"),
and a batch that meets a series again only knows its own pages. Both fold into the first product
found: pages and list fields are unioned, and a size carried in a name becomes one of its sizes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List

SIZE = re.compile(r"\b\d{1,3}(?:[.,]\d+)?\s*[x×X]\s*\d{1,3}(?:[.,]\d+)?(?:\s*[x×X]\s*\d{1,3}(?:[.,]\d+)?)?(?:\s*cm)?\b")
LIST_FIELDS = ("dimensions", "variants", "colors", "finishes", "available_sizes")


def sizes_in(name: str) -> List[str]:
    return [re.sub(r"\s*[x×X]\s*", "x", m.group(0).replace("cm", "").strip()) for m in SIZE.finditer(name or "")]


def clean_name(name: str) -> str:
    """'Amalfi 30x60' -> 'Amalfi'; a name that is only a size stays as it is."""
    cleaned = re.sub(r"\s{2,}", " ", SIZE.sub(" ", name or "")).strip(" -–,/")
    return cleaned or (name or "").strip()


def series_key(name: str) -> str:
    folded = unicodedata.normalize("NFKD", clean_name(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def absorb(into: Any, other: Any) -> None:
    """Fold `other` into `into` (ProductInfo-shaped: name, page_range, description, metadata)."""
    into.page_range = sorted(set(into.page_range or []) | set(other.page_range or []))
    if not into.description and other.description:
        into.description = other.description
    meta: Dict[str, Any] = dict(into.metadata or {})
    theirs: Dict[str, Any] = dict(other.metadata or {})
    for key in LIST_FIELDS:
        a, b = meta.get(key), theirs.get(key)
        if isinstance(a, list) or isinstance(b, list):
            merged = list(a or []) if isinstance(a, list) else ([a] if a else [])
            for v in (b if isinstance(b, list) else ([b] if b else [])):
                if v not in merged:
                    merged.append(v)
            meta[key] = merged
    for key, value in theirs.items():
        if key not in meta or meta[key] in (None, "", [], {}):
            meta[key] = value
    names = list(meta.get("discovered_names") or [into.name])
    for n in theirs.get("discovered_names") or [other.name]:
        if n not in names:
            names.append(n)
    meta["discovered_names"] = names
    into.metadata = meta


def name_series(product: Any) -> None:
    """Strip a size out of the product's name and keep it as a size."""
    sizes = sizes_in(product.name)
    if not sizes:
        return
    meta: Dict[str, Any] = dict(product.metadata or {})
    dims = list(meta.get("dimensions") or []) if isinstance(meta.get("dimensions"), list) else []
    for s in sizes:
        if s not in dims:
            dims.append(s)
    meta["dimensions"] = dims
    names = list(meta.get("discovered_names") or [])
    if product.name not in names:
        names.append(product.name)
    meta["discovered_names"] = names
    product.metadata = meta
    product.name = clean_name(product.name)
