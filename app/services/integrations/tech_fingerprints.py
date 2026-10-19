"""Technologies a homepage gives away (#462). Pure; the patterns come from technology_dictionary."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

_SCRIPT_SRC = re.compile(r"<script\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I)
_META_GENERATOR = re.compile(
    r"<meta\b(?=[^>]*\bname\s*=\s*[\"']generator[\"'])[^>]*\bcontent\s*=\s*[\"']([^\"']*)[\"']", re.I)


def detect(page_html: str, fingerprints: Iterable[Dict[str, Any]]) -> List[str]:
    """Slugs whose detect_html matches. Each list is OR'd; an empty or missing list never matches."""
    page = page_html or ""
    lower = page.lower()
    scripts = [s.lower() for s in _SCRIPT_SRC.findall(page)]
    generators = [g.lower() for g in _META_GENERATOR.findall(page)]
    found: List[str] = []
    for fp in fingerprints:
        rules = fp.get("detect_html") or {}
        hit = (
            any(p.lower() in s for p in rules.get("script_src") or [] if p for s in scripts)
            or any(p.lower() in g for p in rules.get("meta_generator") or [] if p for g in generators)
            or any(p.lower() in lower for p in rules.get("html") or [] if p)
        )
        if hit and fp.get("slug") not in found:
            found.append(fp["slug"])
    return found
