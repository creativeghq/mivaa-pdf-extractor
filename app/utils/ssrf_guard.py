"""Shared SSRF guard (pentest #250 E6)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata",
    "instance-data",  # AWS/GCP metadata aliases
}


class SSRFError(ValueError):
    """Raised when a URL resolves to a disallowed (internal) target."""


def _ip_is_blocked(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable → block
    # 169.254.169.254 is link-local (covered by is_link_local) but call it out explicitly.
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or str(ip) == "169.254.169.254"
    )


def assert_safe_url(url: str, allow_schemes: tuple[str, ...] = ("http", "https")) -> str:
    """
    Return `url` unchanged if it is a safe outbound target, else raise SSRFError.

    Blocks: disallowed schemes (file/gopher/etc.), missing/blocked hostnames, and any host
    whose DNS resolution yields a private / loopback / link-local / metadata / reserved
    address (checks EVERY resolved address, so a mixed-record host can't sneak one through).
    Callers MUST also pass follow_redirects=False (or re-validate every hop).
    """
    if not url or not isinstance(url, str):
        raise SSRFError("empty url")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in allow_schemes:
        raise SSRFError(f"blocked url scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host or host.lower() in _BLOCKED_HOSTNAMES:
        raise SSRFError("blocked host")

    # A literal IP in the URL is checked directly; a hostname is resolved (all records).
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError(f"host resolution failed: {exc}") from exc
    if not infos:
        raise SSRFError("host did not resolve")
    for info in infos:
        ip_str = info[4][0]
        if _ip_is_blocked(ip_str):
            raise SSRFError("host resolves to an internal/blocked address")
    return url


# ─────────────────────────────────────────────────────────────────────────────
# The canonical guarded fetch.

#: An image, anywhere in this platform. Matches `aspect_query._MAX_IMAGE_BYTES`.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

#: A source PDF. Matches `Settings.max_file_size`, which is what upload accepts, so a
#: document we accepted by upload cannot be rejected on the re-download path.
MAX_PDF_BYTES = 100 * 1024 * 1024


class ResponseTooLarge(SSRFError):
    """Body exceeded the caller's cap. Subclasses SSRFError on purpose.

    Every existing `except SSRFError` handler is a "this fetch was refused for safety"
    handler, and a body that will not fit in memory is refused for the same reason.
    A separate exception type would need every one of those handlers edited to be
    caught, and the ones that were missed would surface as a 500.
    """


class SafeFetchResult:
    """What a guarded fetch returns. Deliberately not the httpx.Response.

    Handing back the response would let a caller reach for `.content` — which on a
    streamed response is the body we just bounded, read AGAIN and unbounded. The whole
    point of this helper is that the bytes in your hand are the bytes that fit.
    """

    __slots__ = ("content", "content_type", "status_code", "final_url")

    def __init__(self, content: bytes, content_type: str, status_code: int, final_url: str):
        self.content = content
        self.content_type = content_type
        self.status_code = status_code
        self.final_url = final_url

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


async def safe_fetch_bytes(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30.0,
    allow_schemes: tuple[str, ...] = ("https",),
    max_redirects: int = 3,
    headers=None,
    client=None,
) -> SafeFetchResult:
    """GET `url` with the full invariant-7 treatment, and return bounded bytes."""
    # httpx is imported lazily AND only when we have to build a client, for two
    # reasons that turn out to be the same reason. CI installs pytest and nothing else
    # (`deploy.yml`), so a module-level third-party import here would make every test
    # that loads this file uncollectable. And a caller who supplies its own client can
    # therefore drive this function with a stub — which is how the redirect-revalidation
    # behaviour below is actually tested rather than asserted in a comment.
    owns_client = client is None
    if client is None:
        import httpx

        client = httpx.AsyncClient(timeout=timeout)

    try:
        current = url
        for _hop in range(max_redirects + 1):
            assert_safe_url(current, allow_schemes=allow_schemes)

            async with client.stream(
                "GET",
                current,
                headers=headers,
                timeout=timeout,
                follow_redirects=False,
            ) as resp:
                if 300 <= resp.status_code < 400:
                    location = resp.headers.get("location")
                    if not location:
                        raise SSRFError(
                            f"redirect with no Location header from {current[:120]}"
                        )
                    # Relative Locations are legal and common; resolve against the hop
                    # we actually made, then re-validate at the top of the loop.
                    current = urljoin(current, location)
                    continue

                # Declared length is a fast fail, never the actual cap — an absent or
                # malformed header must not buy an unbounded read. `isdigit()` rather
                # than try/int(): ResponseTooLarge is a ValueError, so an `except
                # ValueError` around the parse would swallow the very rejection it is
                # wrapping.
                declared = resp.headers.get("content-length", "")
                if declared.isdigit() and int(declared) > max_bytes:
                    raise ResponseTooLarge(
                        f"declared {declared} bytes exceeds cap {max_bytes}: {current[:120]}"
                    )

                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ResponseTooLarge(
                            f"exceeded {max_bytes} bytes while streaming: {current[:120]}"
                        )
                    chunks.append(chunk)

                return SafeFetchResult(
                    content=b"".join(chunks),
                    content_type=resp.headers.get("content-type", ""),
                    status_code=resp.status_code,
                    final_url=current,
                )

        raise SSRFError(f"too many redirects (> {max_redirects}) starting at {url[:120]}")
    finally:
        if owns_client:
            await client.aclose()
