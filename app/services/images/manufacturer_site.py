"""Find a product's page on its manufacturer's website and the image files on it.

Every fetch goes through `safe_fetch_bytes` (invariant 7). A page found here is only a
CANDIDATE: image_sourcing confirms it by matching one of the product's own PDF crops to an
image on it before anything from the page is attached.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse

from app.utils.ssrf_guard import safe_fetch_bytes

MAX_HTML_BYTES = 3 * 1024 * 1024
MAX_LISTING_PAGES = 10
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
    folded = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


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
    # A page whose slug spells the product ('30-out-door' for OUTDOOR) wins outright.
    exact = [u for u, _ in links if page_names_product(u, product_name)]
    if exact:
        return exact[:limit]
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


WP_SIZE_COPY = re.compile(r"-(?:\d{2,4}x\d{2,4}|scaled)(?=\.(?:jpe?g|png|webp)$)", re.I)


def image_key(url: str) -> str:
    """One key per picture: host without www, no query, WordPress size copies folded."""
    u = urlparse(url.split("?")[0])
    return WP_SIZE_COPY.sub("", u._replace(netloc=(u.netloc or "").lower().removeprefix("www.")).geturl()).lower()


def page_furniture(own: Sequence[str], others: Sequence[Sequence[str]], markers: Iterable[str] = ()) -> Set[str]:
    """Images a product page shares with the home page or another product's page: menu, banner, blog and footer pictures.

    Izida's series pages carried 47 such files (another brand's tiles, blog photos, colour-filter chips)
    and each was attached to 33-95 series. A file named for this product (its name or an item code in
    `markers`) stays even when a neighbour shows it as a related product.
    """
    marks = [m for m in (_norm(x).replace(" ", "") for x in markers) if len(m) >= 4]

    def named(key: str) -> bool:
        return any(m in _norm(key.rsplit("/", 1)[-1]).replace(" ", "") for m in marks)

    mine = {image_key(u) for u in own}
    return {k for other in others for k in (image_key(u) for u in other) if k in mine and not named(k)}


def reference_pages(links: Sequence[Tuple[str, str]], page: str, limit: int = 2) -> List[str]:
    """Pages beside `page` (same parent path, another slug): what the site puts on every product page."""
    parent = page.rstrip("/").rsplit("/", 1)[0]
    out = []
    for u, _ in links:
        if u.rstrip("/").rsplit("/", 1)[0] == parent and u.rstrip("/") != page.rstrip("/") and u not in out:
            out.append(u)
        if len(out) >= limit:
            break
    return out


def _slug_tokens(page_url: str) -> List[str]:
    """'.../producto/ardenza-brillo-5/' -> ['ardenza', 'brillo']: the last path segment, numeric suffixes dropped."""
    parts = [p for p in urlparse(page_url).path.split("/") if p]
    tokens = _norm(parts[-1]).split() if parts else []
    while tokens and tokens[-1].isdigit():
        tokens.pop()
    return tokens


def page_names_product(page_url: str, product_name: str) -> bool:
    """'/collection/12-name' is NAME and '/13-name-30x90' is 'NAME (30x90 ...)': the
    slug, its leading catalogue id dropped, spells exactly the name and its sizes. Exact, so a
    sibling series ('spatula' vs 'spatula-30x90') never takes the other's page."""
    tokens, sizes = name_tokens(product_name)
    slug = _slug_tokens(page_url)
    while slug and slug[0].isdigit():
        slug.pop(0)
    return bool(tokens) and "".join(slug) == "".join(tokens + sizes)


def search_hits_for(hits: Sequence[Dict[str, Any]], site_root: str, product_name: str,
                    sibling_names: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """(image url, page url) for the image-search hits that are this product's, on its maker's own site.

    A hit counts when the image and its page are on the maker's host and the page slug starts with
    the product name. A sibling whose longer name also starts the slug takes the hit instead
    ("Wood Espiga" pages are not "Wood"). WordPress size copies collapse to the largest.
    """
    host = _site_host(site_root)
    tokens = _norm(product_name.split("(")[0]).split()
    if not host or not tokens:
        return []
    longer = [t for t in (_norm(n.split("(")[0]).split() for n in sibling_names)
              if len(t) > len(tokens) and t[:len(tokens)] == tokens]
    best: Dict[str, Tuple[int, str, str]] = {}
    for h in hits:
        img, page = h.get("imageUrl") or "", h.get("url") or ""
        if not IMAGE_EXT.search(img) or SKIP_IMAGE.search(img):
            continue
        if not all(_site_host(u) == host or _site_host(u).endswith("." + host) for u in (img, page)):
            continue
        slug = _slug_tokens(page)
        if slug[:len(tokens)] != tokens or any(slug[:len(t)] == t for t in longer):
            continue
        area = int(h.get("imageWidth") or 0) * int(h.get("imageHeight") or 0)
        key = WP_SIZE_COPY.sub("", img.split("?")[0])
        if key not in best or area > best[key][0]:
            best[key] = (area, img, page)
    return [(img, page) for _, img, page in sorted(best.values(), key=lambda v: -v[0])]
