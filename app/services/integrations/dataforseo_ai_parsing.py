"""Readers for the DataForSEO AI Optimization payloads. Stdlib only, so CI can test them."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "first_ai_result",
    "response_text",
    "response_annotations",
    "response_urls",
    "response_usage",
    "response_fan_out_queries",
    "mentions_rows",
    "scraper_text",
    "scraper_source_urls",
    "scraper_brands",
    "scraper_check_url",
    "LLM_MODEL_PREFERENCE",
    "pick_llm_model",
]

#: LLM Responses REQUIRES `model_name` (40501 without it). Small, current, web-search
#: models first: the probe asks who gets named, which needs search, not depth.
LLM_MODEL_PREFERENCE: Dict[str, List[str]] = {
    "chat_gpt": ["gpt-5.4-mini", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini"],
    "claude": ["claude-haiku-5-5", "claude-haiku-4-5", "claude-sonnet-5"],
    "gemini": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"],
    "perplexity": ["sonar", "sonar-pro"],
}


def pick_llm_model(family: str, models_result: Any) -> Optional[str]:
    """The model to name for `family`, from the live `/llm_responses/models` result.

    A preferred model that is listed wins; otherwise the first listed model that
    supports web search; with no usable list, the first preference (the call then
    fails loudly if it is gone, rather than silently asking a model without search).
    """
    prefs = LLM_MODEL_PREFERENCE.get(family.lower(), [])
    rows = [r for r in (models_result or []) if isinstance(r, dict) and r.get("model_name")]
    searchable = [str(r["model_name"]) for r in rows if r.get("web_search_supported")]
    for name in prefs:
        if name in searchable:
            return name
    if searchable:
        return searchable[0]
    return prefs[0] if prefs else None


def first_ai_result(raw: Any) -> Dict[str, Any]:
    """The AI result object out of a raw DataForSEO envelope.

    Read from `raw`, never from the client's flattened `items`: `_call` hoists any
    result carrying an `items` key, so the AI endpoints - whose result object IS
    `{model_name, input_tokens, items: [...]}` - arrive flattened to their message
    items with the tokens and the model name gone. Parsing that yields an empty answer
    on a perfectly good response.
    """
    tasks = (raw or {}).get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return {}
    results = (tasks[0] or {}).get("result")
    if not isinstance(results, list) or not results:
        return {}
    return results[0] if isinstance(results[0], dict) else {}


def _sections(result: Any) -> List[Dict[str, Any]]:
    """Every section of every item. Text and annotations live on the SECTION, not the
    item - an extractor reading `item["annotations"]` finds None on every answer."""
    out: List[Dict[str, Any]] = []
    for item in (result or {}).get("items") or []:
        if not isinstance(item, dict):
            continue
        for section in item.get("sections") or []:
            if isinstance(section, dict):
                out.append(section)
    return out


def response_text(result: Any) -> str:
    """The answer, as one string."""
    parts = [str(s.get("text") or "") for s in _sections(result) if s.get("type") == "text"]
    return "\n".join(p for p in parts if p).strip()


def response_annotations(result: Any) -> List[Dict[str, Any]]:
    """Sources, each bound to the span of the answer it supports.

    Richer than a flat URL list: `start_index`/`end_index` say WHICH claim came from
    the page, which is what lets a report name the sentence a competitor won. The
    array is null unless the request set `web_search`, and empty when the engine
    searched and found nothing - those are different facts and neither is a citation.
    """
    out: List[Dict[str, Any]] = []
    for section in _sections(result):
        for a in section.get("annotations") or []:
            if not isinstance(a, dict) or not a.get("url"):
                continue
            out.append({
                "url": str(a["url"]),
                "title": a.get("title"),
                "text": a.get("text"),
                "start_index": a.get("start_index"),
                "end_index": a.get("end_index"),
            })
    return out


def response_urls(result: Any) -> List[str]:
    """Cited URLs, order-preserving and deduped. Duplicates are expected - one source
    can support several spans, and the rebuilt annotations emit a row per span."""
    seen: set = set()
    urls: List[str] = []
    for a in response_annotations(result):
        key = a["url"].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(a["url"])
    return urls


def response_usage(result: Any) -> Dict[str, Any]:
    """Tokens and what DataForSEO paid the model. `money_spent` is the LLM API charge
    only; the task base price sits on the envelope, so the two must not be added twice."""
    r = result or {}
    return {
        "model_name": r.get("model_name"),
        "input_tokens": int(r.get("input_tokens") or 0),
        "output_tokens": int(r.get("output_tokens") or 0),
        "reasoning_tokens": int(r.get("reasoning_tokens") or 0),
        "money_spent": r.get("money_spent"),
        "web_search": bool(r.get("web_search")),
    }


def response_fan_out_queries(result: Any) -> List[str]:
    """The searches the engine actually ran to answer. These are the real queries to
    rank for - they are rarely the question the buyer typed."""
    raw = (result or {}).get("fan_out_queries")
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for q in raw:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
        elif isinstance(q, dict):
            v = q.get("query") or q.get("keyword")
            if v:
                out.append(str(v))
    return out


def mentions_rows(result: Any) -> List[Dict[str, Any]]:
    """The rows of any LLM Mentions endpoint. They all wrap their payload in `items`,
    but the lite tier returns the row list directly - handle both or lite reads empty."""
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    items = (result or {}).get("items")
    if isinstance(items, list):
        return [r for r in items if isinstance(r, dict)]
    return []


def scraper_text(result: Any) -> Optional[str]:
    """The answer as the consumer surface rendered it.

    The scraper speaks `markdown`, not `text`, and not the `sections` that
    llm_responses uses — three shapes for "the answer" across one vendor. The
    top-level field is the whole reply; the per-item ones are its blocks, joined
    only when the top level is absent.
    """
    r = result or {}
    top = str(r.get("markdown") or "").strip()
    if top:
        return top
    parts: List[str] = []
    for item in r.get("items") or []:
        if isinstance(item, dict) and item.get("markdown"):
            parts.append(str(item["markdown"]))
    joined = "\n".join(parts).strip()
    return joined or None


def scraper_source_urls(result: Any) -> List[str]:
    """Cited URLs from a scraped answer — the links the consumer surface actually shows.

    Deduped on the URL MINUS its tracking query: every ChatGPT source carries
    `?utm_source=chatgpt.com`, so a raw dedupe keeps the same page twice when one copy
    happens to arrive without it.
    """
    seen: set = set()
    out: List[str] = []
    for src in _scraper_sources(result):
        url = str(src.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        key = url.split("?")[0].rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(url)
    return out


def scraper_brands(result: Any) -> List[str]:
    """Brands the scraper itself identified in the answer.

    Ground truth rather than an LLM re-reading the prose: DataForSEO marks the entity
    while it has the rendered page, including the ones a text pass would miss.
    """
    out: List[str] = []
    seen: set = set()
    for item in [(result or {}), *((result or {}).get("items") or [])]:
        if not isinstance(item, dict):
            continue
        for b in item.get("brand_entities") or []:
            if not isinstance(b, dict):
                continue
            name = str(b.get("title") or "").strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append(name)
    return out


def scraper_check_url(result: Any) -> Optional[str]:
    """The link that reproduces this answer on the consumer surface."""
    v = (result or {}).get("check_url")
    return str(v) if v else None


def _scraper_sources(result: Any) -> List[Dict[str, Any]]:
    """Sources live top-level AND on each item; the item ones are a subset but not
    always, so both are read and the caller dedupes."""
    out: List[Dict[str, Any]] = []
    for holder in [(result or {}), *((result or {}).get("items") or [])]:
        if not isinstance(holder, dict):
            continue
        for src in holder.get("sources") or []:
            if isinstance(src, dict):
                out.append(src)
    return out
