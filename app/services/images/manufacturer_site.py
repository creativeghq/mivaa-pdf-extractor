"""Find a product's page on its manufacturer's website and the image files on it.

Every fetch goes through `safe_fetch_bytes` (invariant 7). A page found here is only a
CANDIDATE: image_sourcing confirms it by matching one of the product's own PDF crops to an
image on it before anything from the page is attached.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

from app.utils.ssrf_guard import safe_fetch_bytes

MAX_HTML_BYTES = 3 * 1024 * 1024
MAX_LISTING_PAGES = 4
LISTING_HINTS = ("collection", "collezion", "series", "serie", "product", "prodott", "catalog", "range", "gamma", "colecc", "kolle")
IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp)(\?|#|$)", re.I)
NOT_A_PAGE = re.compile(r"\.(pdf|zip|rar|docx?|xlsx?|dwg|dxf|rfa|skp|gif|svg|mp4)(\?|#|$)", re.I)
SKIP_IMAGE = re.compile(r"(thumb|logo|icon|favicon|sprite|banner-?nav|flag|placeholder|catalog(?:ue)?s?/)", re.I)
NAME_STOP = {"tile", "tiles", "wall", "floor", "collection", "series", "serie", "the", "and", "of", "for", "decor"}
SIZE = re.compile(r"\b\d{2,3}\s*[x×]\s*\d{2,3}\b", re.I)
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (compatible; MaterialsHub product-image lookup)"}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[Tuple[str, str]] = []
        self.images: List[str] = []
        self.base: Optional[str] = None
        self._href: Optional[str] = None
        self._text: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "base" and a.get("href") and self.base is None:
            self.base = a["href"]
        elif tag == "a" and a.get("href"):
            self._href, self._text = a["href"], []
            if IMAGE_EXT.search(a["href"]):
                self.images.append(a["href"])
        elif tag in ("img", "source"):
            for key in ("data-src", "data-lazy-src", "data-original", "src"):
                if a.get(key) and not a[key].startswith("data:"):
                    self.images.append(a[key])
                    break
            srcset = a.get("srcset") or a.get("data-srcset")
            if srcset:
                last = srcset.split(",")[-1].strip().split(" ")[0]
                if last:
                    self.images.append(last)
        elif tag == "meta" and a.get("property", "").lower() == "og:image" and a.get("content"):
            self.images.append(a["content"])

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join(t.strip() for t in self._text if t.strip())))
            self._href = None


def _site_host(url: str) -> str:
    return (urlparse(url).hostname or "").removeprefix("www.")


def parse_page(html: str, base_url: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    p = _PageParser()
    try:
        p.feed(html)
    except Exception:
        pass
    # Relative links resolve against <base href> when the page declares one, as a browser does.
    base_url = urljoin(base_url, p.base) if p.base else base_url
    host = _site_host(base_url)
    links, seen = [], set()
    for href, text in p.links:
        u = urljoin(base_url, href.strip()).split("#")[0]
        if _site_host(u) == host and u not in seen and not IMAGE_EXT.search(u) and not NOT_A_PAGE.search(u):
            seen.add(u)
            links.append((u, text))
    images, seen_i = [], set()
    for src in p.images:
        u = urljoin(base_url, src.strip())
        h = _site_host(u)
        if not (h == host or h.endswith("." + host)):
            continue
        if not IMAGE_EXT.search(u) or SKIP_IMAGE.search(u) or u in seen_i:
            continue
        seen_i.add(u)
        images.append(u)
    return links, images


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def name_tokens(product_name: str) -> Tuple[List[str], List[str]]:
    """('spatula'), ('30x90') from 'SPATULA (30x90 wall tile)': the name proper, then its sizes."""
    base = product_name.split("(")[0]
    sizes = [re.sub(r"\s*[x×]\s*", "x", m.lower()) for m in SIZE.findall(product_name)]
    tokens = [t for t in _norm(SIZE.sub(" ", base)).split() if len(t) >= 2 and t not in NAME_STOP]
    return tokens, sizes


def page_score(url: str, text: str, tokens: Sequence[str], sizes: Sequence[str]) -> float:
    hay = " " + _norm(urlparse(url).path + " " + text) + " "
    if not tokens or not all(f" {t} " in hay or f"{t} " in hay for t in tokens):
        return 0.0
    joined = hay.replace(" ", "")
    return 1.0 + sum(0.5 for s in sizes if s in joined)


async def fetch_html(url: str) -> Optional[Tuple[str, str]]:
    try:
        r = await safe_fetch_bytes(url, max_bytes=MAX_HTML_BYTES, timeout=25, headers=BROWSER_UA)
    except Exception:
        return None
    if not r.ok or "html" not in (r.content_type or "").lower():
        return None
    return r.final_url, r.content.decode("utf-8", errors="replace")


async def site_links(site: str) -> List[Tuple[str, str]]:
    """Every same-site link on the home page and on up to MAX_LISTING_PAGES collection listings."""
    home = await fetch_html(site)
    if not home:
        return []
    links, _ = parse_page(home[1], home[0])
    out: Dict[str, str] = {u: t for u, t in links}
    listings = [u for u, t in links if any(h in (u + " " + t).lower() for h in LISTING_HINTS)][:MAX_LISTING_PAGES]
    for u in listings:
        page = await fetch_html(u)
        if page:
            for lu, lt in parse_page(page[1], page[0])[0]:
                out.setdefault(lu, lt)
    return list(out.items())


def candidate_pages(links: Sequence[Tuple[str, str]], product_name: str, limit: int = 2) -> List[str]:
    tokens, sizes = name_tokens(product_name)
    scored = [(page_score(u, t, tokens, sizes), u) for u, t in links]
    scored = sorted((s for s in scored if s[0] > 0), reverse=True)
    if not scored:
        return []
    top = scored[0][0]
    return [u for s, u in scored if s == top][:limit]


async def page_images(url: str) -> List[str]:
    page = await fetch_html(url)
    if not page:
        return []
    return parse_page(page[1], page[0])[1]
