"""Public ATS job boards: where they live, how to read them, how to spot them on a website.

Pure functions, no I/O. A parser raises `AtsShapeError` when the top-level shape is not the
one the provider documents: a board that changed shape must read as a failure, never as a
company that stopped hiring.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urljoin, urlparse

from app.utils.html_to_markdown import html_to_markdown

ENUMERABLE = ("greenhouse", "lever", "ashby", "workable", "teamtailor")
_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$")


class AtsShapeError(ValueError):
    """The board answered, but not in the shape its provider documents."""


def board_url(provider: str, token: str) -> str:
    """`eu:<slug>` is a Lever board on its EU data centre, which the US API 404s."""
    eu = token.startswith("eu:")
    t = quote(unquote(token[3:] if eu else token), safe="")
    if provider == "greenhouse":
        return f"https://boards-api.greenhouse.io/v1/boards/{t}/jobs?content=true"
    if provider == "lever":
        return f"https://api{'.eu' if eu else ''}.lever.co/v0/postings/{t}?mode=json"
    if provider == "ashby":
        return f"https://api.ashbyhq.com/posting-api/job-board/{t}"
    if provider == "workable":
        return f"https://apply.workable.com/api/v1/widget/accounts/{t}?details=true"
    if provider == "teamtailor":
        host = token.lower()
        if not _HOST_RE.match(host):
            raise ValueError(f"teamtailor token must be a careers host, got {token[:80]!r}")
        return f"https://{host}/jobs.rss"
    raise ValueError(f"no enumerator for provider {provider!r}")


def _iso(v: Any) -> Optional[str]:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        try:
            return datetime.fromtimestamp(int(v) / 1000.0, tz=timezone.utc).isoformat()
        except (ValueError, OverflowError):
            return None
    s = str(v).strip()
    try:
        return parsedate_to_datetime(s).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return None


def _posting(external_id, title, url, *, description_md="", location=None, is_remote=None,
             employment_type=None, department=None, posted_at=None) -> Optional[Dict[str, Any]]:
    if not external_id or not title or not url:
        return None
    return {
        "external_id": str(external_id),
        "title": str(title).strip(),
        "canonical_url": str(url).strip(),
        "description_md": description_md or "",
        "location": location or None,
        "is_remote": is_remote,
        "employment_type": employment_type or None,
        "department": department or None,
        "posted_at": _iso(posted_at),
    }


def _greenhouse(data: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise AtsShapeError("greenhouse: expected an object with a 'jobs' array")
    out, company = [], None
    for r in data["jobs"]:
        if not isinstance(r, dict):
            continue
        company = company or r.get("company_name")
        loc = (r.get("location") or {}).get("name") if isinstance(r.get("location"), dict) else None
        depts = [d.get("name") for d in (r.get("departments") or []) if isinstance(d, dict) and d.get("name")]
        out.append(_posting(
            r.get("id"), r.get("title"), r.get("absolute_url"),
            description_md=html_to_markdown(r.get("content"), unescape_first=True),
            location=loc, is_remote=("remote" in (loc or "").lower()) or None,
            department=", ".join(depts) or None,
            posted_at=r.get("first_published") or r.get("updated_at"),
        ))
    return company, [p for p in out if p]


def _lever(data: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not isinstance(data, list):
        raise AtsShapeError("lever: expected a postings array")
    out = []
    for r in data:
        if not isinstance(r, dict):
            continue
        parts = [html_to_markdown(r.get("description"))]
        for lst in r.get("lists") or []:
            if isinstance(lst, dict):
                parts.append(f"## {(lst.get('text') or '').strip()}\n\n"
                             + html_to_markdown(f"<ul>{lst.get('content') or ''}</ul>"))
        parts.append(html_to_markdown(r.get("additional")))
        cats = r.get("categories") or {}
        wt = (r.get("workplaceType") or "").lower()
        out.append(_posting(
            r.get("id"), r.get("text"), r.get("hostedUrl") or r.get("applyUrl"),
            description_md="\n\n".join(p for p in parts if p.strip()),
            location=cats.get("location"),
            is_remote=True if wt == "remote" else (False if wt in ("onsite", "on-site") else None),
            employment_type=cats.get("commitment"),
            department=cats.get("department") or cats.get("team"),
            posted_at=r.get("createdAt"),
        ))
    return None, [p for p in out if p]


def _ashby(data: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise AtsShapeError("ashby: expected an object with a 'jobs' array")
    out = []
    for r in data["jobs"]:
        if not isinstance(r, dict) or r.get("isListed") is False:
            continue
        out.append(_posting(
            r.get("id"), r.get("title"), r.get("jobUrl") or r.get("applyUrl"),
            description_md=html_to_markdown(r.get("descriptionHtml")) or (r.get("descriptionPlain") or ""),
            location=r.get("location"), is_remote=r.get("isRemote"),
            employment_type=r.get("employmentType"),
            department=r.get("department") or r.get("team"),
            posted_at=r.get("publishedAt"),
        ))
    return None, [p for p in out if p]


def _workable(data: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise AtsShapeError("workable: expected an object with a 'jobs' array")
    out = []
    for r in data["jobs"]:
        if not isinstance(r, dict):
            continue
        loc = ", ".join(x for x in (r.get("city"), r.get("state"), r.get("country")) if x) or None
        out.append(_posting(
            r.get("shortcode"), r.get("title"), r.get("url") or r.get("shortlink"),
            description_md=html_to_markdown(r.get("description")),
            location=loc, is_remote=bool(r.get("telecommuting")) if r.get("telecommuting") is not None else None,
            employment_type=r.get("employment_type"), department=r.get("department"),
            posted_at=r.get("published_on") or r.get("created_at"),
        ))
    return data.get("name") or None, [p for p in out if p]


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1].lower() if "}" in tag else tag.lower()


def _teamtailor(body: bytes) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    import defusedxml.ElementTree as _ET  # lazy: CI installs pytest alone

    try:
        root = _ET.fromstring(body)
    except _ET.ParseError as e:
        raise AtsShapeError(f"teamtailor: rss did not parse: {e}") from e
    channel = next((el for el in root.iter() if _local(el.tag) == "channel"), None)
    if channel is None:
        raise AtsShapeError("teamtailor: no rss channel")
    company = next((el.text for el in channel if _local(el.tag) == "title" and el.text), None)
    out = []
    for item in channel:
        if _local(item.tag) != "item":
            continue
        f: Dict[str, Any] = {}
        city = country = None
        for ch in item.iter():
            t = _local(ch.tag)
            if t in ("title", "description", "link", "guid", "pubdate", "department", "remotestatus") and t not in f:
                f[t] = (ch.text or "").strip()
            elif t == "city" and not city:
                city = (ch.text or "").strip() or None
            elif t == "country" and not country:
                country = (ch.text or "").strip() or None
        remote = (f.get("remotestatus") or "").lower()
        out.append(_posting(
            f.get("guid") or f.get("link"), f.get("title"), f.get("link"),
            description_md=html_to_markdown(f.get("description")),
            location=", ".join(x for x in (city, country) if x) or None,
            is_remote=True if remote in ("fully", "remote") else (False if remote in ("none", "onsite") else None),
            department=f.get("department"), posted_at=f.get("pubdate"),
        ))
    return company, [p for p in out if p]


def parse_board(provider: str, body: bytes) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """(company name if the board states one, postings). Raises AtsShapeError on a wrong shape."""
    if provider == "teamtailor":
        return _teamtailor(body)
    try:
        data = json.loads(body)
    except ValueError as e:
        raise AtsShapeError(f"{provider}: body is not JSON: {e}") from e
    parser = {"greenhouse": _greenhouse, "lever": _lever, "ashby": _ashby, "workable": _workable}.get(provider)
    if parser is None:
        raise ValueError(f"no parser for provider {provider!r}")
    return parser(data)


# ── Spotting a board on a company website ───────────────────────────────────

_SKIP_TOKENS = {"embed", "v1", "v0", "api", "jobs", "j", "boards", "posting-api", "job-board", "widget",
                "accounts", "companies", "apply", "www", "careers", "help", "resources"}
_LINK_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("greenhouse", re.compile(r"greenhouse\.io/embed/job_(?:board(?:/js)?|app)\?(?:[^\"'\s<>]*&(?:amp;)?)?for=([a-z0-9_-]{1,100})", re.I)),
    ("greenhouse", re.compile(r"(?:job-)?boards(\.eu)?(?:-api)?\.greenhouse\.io/(?:v1/boards/)?([a-z0-9_-]{1,100})", re.I)),
    ("lever", re.compile(r"jobs(\.eu)?\.lever\.co/([a-z0-9_.-]{1,100})", re.I)),
    ("ashby", re.compile(r"jobs\.ashbyhq\.com/([a-z0-9_.%-]{1,100})", re.I)),
    ("workable", re.compile(r"apply\.workable\.com/(?:api/v\d/widget/accounts/)?([a-z0-9_-]+)", re.I)),
    ("workable", re.compile(r"//([a-z0-9-]+)\.workable\.com", re.I)),
    ("recruitee", re.compile(r"//([a-z0-9-]+)\.recruitee\.com", re.I)),
    ("personio", re.compile(r"//([a-z0-9-]+)\.jobs\.personio\.(?:de|com)", re.I)),
    ("smartrecruiters", re.compile(r"(?:careers|jobs)\.smartrecruiters\.com/([a-z0-9_-]+)", re.I)),
    ("successfactors", re.compile(r"//(career\d*\.successfactors\.(?:com|eu))/career\?company=([a-z0-9_]+)", re.I)),
]
_TEAMTAILOR_HOST = re.compile(r"utm_content=([a-z0-9.-]+\.[a-z]{2,})", re.I)
_CAREER_WORDS = re.compile(
    r"career|jobs?\b|lavora|carriere|trabaja|empleo|karriere|stellen|kariera|θέσεις|εργασ|καριέρ|"
    r"work-with-us|join-us|recruit|posizioni|vacanc", re.I)
_ANCHOR = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"'#]+)[\"'][^>]*>(.{0,200}?)</a>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")


def find_ats_accounts(page_html: str) -> List[Tuple[str, str]]:
    """Every (provider, token) a page links to, in first-seen order."""
    found: List[Tuple[str, str]] = []
    for provider, pat in _LINK_PATTERNS:
        for m in pat.finditer(page_html or ""):
            if provider == "successfactors" or m.lastindex == 2:
                region, token = (None, m.group(2)) if provider == "successfactors" else m.groups()
            else:
                region, token = None, m.group(1)
            token = token.strip("/").lower()
            if not token or token in _SKIP_TOKENS or len(token) > 100:
                continue
            if region and provider == "lever":
                token = f"eu:{token}"
            if (provider, token) not in found:
                found.append((provider, token))
    if "teamtailor.com" in (page_html or "").lower():
        for m in _TEAMTAILOR_HOST.finditer(page_html):
            host = m.group(1).lower()
            if _HOST_RE.match(host) and ("teamtailor", host) not in found:
                found.append(("teamtailor", host))
    return found


def find_careers_links(page_html: str, base_url: str, limit: int = 3) -> List[str]:
    """Same-site links whose href or text reads like a careers page."""
    base_host = (urlparse(base_url).hostname or "").removeprefix("www.")
    out: List[str] = []
    for href, text in _ANCHOR.findall(page_html or ""):
        label = _TAGS.sub(" ", text)
        if not (_CAREER_WORDS.search(href) or _CAREER_WORDS.search(label)):
            continue
        u = urljoin(base_url, html.unescape(href.strip()))
        p = urlparse(u)
        host = (p.hostname or "").removeprefix("www.")
        if p.scheme not in ("http", "https") or not host:
            continue
        if host != base_host and not host.endswith("." + base_host):
            continue
        if u not in out:
            out.append(u)
        if len(out) >= limit:
            break
    return out


def count_anchors(page_html: str) -> int:
    return len(_ANCHOR.findall(page_html or ""))
