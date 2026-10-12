"""Grammar matches from the LanguageTool server running on this host (systemd unit `languagetool`).

Stdlib only and loadable by path, so the unit tests run under MIVAA CI's bare pytest.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List

DEFAULT_URL = "http://127.0.0.1:8081"
MAX_TEXT_CHARS = 60000
MAX_REPLACEMENTS = 3

#: A bare language picks the variant LanguageTool needs for its full rule set.
_VARIANTS = {"el": "el-GR", "en": "en-US", "de": "de-DE", "pt": "pt-PT", "nl": "nl-NL", "ca": "ca-ES"}
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{2,4})?$")


class GrammarCheckUnavailable(Exception):
    """LanguageTool did not answer, or answered with something that is not a check result."""


def server_url() -> str:
    return (os.getenv("LANGUAGETOOL_URL") or DEFAULT_URL).rstrip("/")


def resolve_language(language: str) -> str:
    code = (language or "").strip().replace("_", "-")
    if not _LANGUAGE_RE.match(code):
        raise ValueError(f"not a language code: {language!r}")
    return _VARIANTS.get(code.lower(), code)


def _post_form(url: str, form: Dict[str, str], timeout: float) -> Dict[str, Any]:
    data = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — fixed local URL
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            raise ValueError(f"LanguageTool refused the request: {exc.read()[:200]!r}") from exc
        raise GrammarCheckUnavailable(f"LanguageTool answered HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise GrammarCheckUnavailable(f"LanguageTool unreachable: {exc}") from exc


def _utf16_slice(text: str, offset: int, length: int) -> str:
    units = text.encode("utf-16-le")
    return units[offset * 2:(offset + length) * 2].decode("utf-16-le", errors="replace")


def parse_matches(text: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """LanguageTool's matches, flattened.

    `offset`/`length` are UTF-16 code units — LanguageTool's own unit and a JavaScript string's,
    so the edge caller indexes its text with them unchanged.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
        raise GrammarCheckUnavailable("LanguageTool reply carries no matches list")
    out: List[Dict[str, Any]] = []
    for m in payload["matches"]:
        rule = m.get("rule") or {}
        offset, length = int(m.get("offset", 0)), int(m.get("length", 0))
        out.append({
            "offset": offset,
            "length": length,
            "text": _utf16_slice(text, offset, length),
            "message": m.get("message") or m.get("shortMessage") or "",
            "rule_id": rule.get("id") or "",
            "category": (rule.get("category") or {}).get("id") or "",
            "issue_type": rule.get("issueType") or "",
            "replacements": [r.get("value", "") for r in (m.get("replacements") or [])[:MAX_REPLACEMENTS]],
            "sentence": m.get("sentence") or "",
        })
    return out


def check(
    language: str,
    text: str,
    *,
    post: Callable[[str, Dict[str, str], float], Dict[str, Any]] = _post_form,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Raises:
        ValueError: bad language code or text over MAX_TEXT_CHARS.
        GrammarCheckUnavailable: the server could not be reached or replied with garbage.
    """
    lang = resolve_language(language)
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is {len(text)} characters; the limit is {MAX_TEXT_CHARS}")
    payload = post(f"{server_url()}/v2/check", {"language": lang, "text": text}, timeout)
    return {
        "language": lang,
        "matches": parse_matches(text, payload),
        "software": ((payload.get("software") or {}).get("version") or ""),
    }
