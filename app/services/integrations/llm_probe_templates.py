"""
The questions we ask AI assistants about a subject.

Split out of `llm_mention_probe_service` for the same two reasons as
`llm_visibility_math`: this is a DERIVATION (subject facets in, prompts out) and
the service it came from cannot be imported without a Supabase client, so a
template built there is a template no test can exercise. Stdlib only.

WHAT WAS WRONG
--------------
`build_probes` carried four hardcoded prompts and a docstring promising "Caller
may add more via source_config". Nothing read `source_config` — anywhere. So the
promise was false and every workspace asked the same four generic questions,
including this one:

    "What are the best products brands? Give a ranked list..."

`product_type` defaults to the literal string "products", so for a subject with
no product type the sentence degrades to "the best products brands", which is not
English and is not a question anyone would ask an assistant. A probe that reads
like that is measuring the wrong thing even when the plumbing works.

Custom prompts are now real, validated, and rendered from the same facets.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

__all__ = [
    "DEFAULT_TEMPLATES",
    "PLACEHOLDERS",
    "render_template",
    "normalize_custom_probes",
    "build_probes",
]

# Placeholders a prompt may use. Anything else is left untouched rather than
# raising — a stray brace in a merchant's prose must not break their probe run.
PLACEHOLDERS = ("label", "brand", "product_type", "competitors", "site")

DEFAULT_TEMPLATES: List[Dict[str, str]] = [
    {
        "key": "generic_recommendation",
        "prompt": "What are the best {product_type} brands? Give a ranked list with one short reason per entry.",
    },
    {
        "key": "use_case",
        "prompt": "Recommend 5 {product_type} for use in a high-traffic commercial space. Name them and briefly explain each.",
    },
    {
        "key": "comparison",
        "prompt": "Compare {brand} with {competitors}. Cover product range, quality, and typical price tier.",
    },
    {
        "key": "direct_lookup",
        "prompt": "Tell me about {label}. What do they make and what are they known for?",
    },
]

_KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def render_template(template: str, values: Dict[str, str]) -> str:
    """Substitute `{placeholder}` tokens.

    Deliberately NOT `str.format`: a merchant's prompt is untrusted prose, and
    `format` raises KeyError on any brace it does not recognise — including a
    perfectly reasonable "what does {this} mean". Unknown tokens are left as
    written so the prompt still runs and the author can see what happened.
    """
    out = template or ""
    for name in PLACEHOLDERS:
        out = out.replace("{" + name + "}", str(values.get(name) or "").strip())
    # Collapse the whitespace an emptied placeholder leaves behind.
    return re.sub(r"\s{2,}", " ", out).strip()


def normalize_custom_probes(raw: Any, *, limit: int = 12) -> List[Dict[str, str]]:
    """Validate workspace-authored probes out of `source_config.custom_probes`.

    Anything malformed is DROPPED rather than raising: this runs inside a nightly
    cron over every subject, and one workspace with a bad row must not stop the
    sweep for everyone else. Capped, because each probe is a paid model call per
    engine and an unbounded list is an unbounded bill.
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, str]] = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        prompt = str(entry.get("prompt") or "").strip()
        if len(prompt) < 8:
            continue
        key = str(entry.get("key") or "").strip().lower() or f"custom_{len(out) + 1}"
        if not _KEY_RE.match(key):
            key = f"custom_{len(out) + 1}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "prompt": prompt[:600]})
        if len(out) >= limit:
            break
    return out


def build_probes(
    facets: Any,
    *,
    custom_probes: Any = None,
    include_defaults: bool = True,
    site: Optional[str] = None,
) -> List[Dict[str, str]]:
    """The probe set for one subject.

    `include_defaults=False` lets a workspace replace the stock questions entirely
    rather than only appending to them — asking four irrelevant questions
    alongside four good ones dilutes share of voice with noise the merchant did
    not choose to measure.

    Falls back to the defaults when a caller disables them AND supplies nothing
    usable, because a subject with zero probes would silently stop being measured
    while still looking tracked.
    """
    product_type = getattr(facets, "product_type", None) or "products"
    label = getattr(facets, "label", "") or ""
    brand = getattr(facets, "brand", None) or label
    competitor_brands = list(getattr(facets, "competitor_brands", None) or [])
    values = {
        "label": label,
        "brand": brand,
        "product_type": product_type,
        "competitors": ", ".join(competitor_brands[:3]) if competitor_brands else "leading alternatives",
        "site": site or "",
    }

    probes: List[Dict[str, str]] = []
    if include_defaults:
        probes.extend(
            {"key": t["key"], "prompt": render_template(t["prompt"], values)}
            for t in DEFAULT_TEMPLATES
        )

    custom = normalize_custom_probes(custom_probes)
    existing = {p["key"] for p in probes}
    for c in custom:
        key = c["key"] if c["key"] not in existing else f"{c['key']}_custom"
        probes.append({"key": key, "prompt": render_template(c["prompt"], values)})
        existing.add(key)

    if not probes:
        probes = [
            {"key": t["key"], "prompt": render_template(t["prompt"], values)}
            for t in DEFAULT_TEMPLATES
        ]
    return probes
