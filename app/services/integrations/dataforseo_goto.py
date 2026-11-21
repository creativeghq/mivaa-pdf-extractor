"""Google GoTo passthrough URLs in DataForSEO SERP items. Stdlib only, so CI can test it.

Since late October 2026 a Live SERP returns most result URLs as `https://google.com/goto?url=<opaque>`
with `domain: "google.com"`. The breadcrumb still starts with the real origin, so the site is
recoverable even though the page path is not. Twin: supabase/functions/_shared/seo/serp-goto.ts.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import urlparse

_GOTO = re.compile(r"^https?://(?:www\.)?google\.[a-z.]+/goto\?", re.IGNORECASE)
_ORIGIN = re.compile(r"^\s*(https?://[^\s/›>]+)", re.IGNORECASE)
_MAX_DEPTH = 12  # envelope → tasks → result → items → ai_overview.references is already 8


def is_goto_url(value: Any) -> bool:
    return isinstance(value, str) and bool(_GOTO.match(value))


def breadcrumb_host(breadcrumb: Any) -> Optional[str]:
    if not isinstance(breadcrumb, str):
        return None
    m = _ORIGIN.match(breadcrumb)
    if not m:
        return None
    host = (urlparse(m.group(1)).hostname or "").lower()
    if not host or host.endswith(".google.com") or host == "google.com":
        return None
    return host


def _fix(node: dict) -> int:
    if not is_goto_url(node.get("url")):
        return 0
    node["goto_url"] = node["url"]
    node["url"] = None
    node["url_unresolved"] = True
    domain = str(node.get("domain") or "").lower()
    if not domain or domain == "google.com" or domain.endswith(".google.com"):
        node["domain"] = breadcrumb_host(node.get("breadcrumb"))
    return 1


def normalize_in_place(node: Any, depth: int = 0) -> int:
    """An unresolved GoTo item keeps its real domain (from the breadcrumb, else None) and a NULL url;
    the redirect moves to `goto_url`. Returns how many items it rewrote."""
    if depth > _MAX_DEPTH:
        return 0
    if isinstance(node, list):
        return sum(normalize_in_place(n, depth + 1) for n in node)
    if not isinstance(node, dict):
        return 0
    fixed = _fix(node)
    for v in node.values():
        if isinstance(v, (dict, list)):
            fixed += normalize_in_place(v, depth + 1)
    return fixed
