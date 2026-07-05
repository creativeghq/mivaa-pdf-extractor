"""
Shared SSRF guard (pentest #250 E6).

Every place the server fetches a URL whose host a user can influence must validate it
through `assert_safe_url()` first, and disable redirect-following (a permitted external
host can 302 into an internal target). This blocks the classic SSRF-to-cloud-metadata
(169.254.169.254) and internal-service reachability found in the audit (sam_routes,
image-analyze, feed URLs, alert webhooks, …).

Usage:
    from app.utils.ssrf_guard import assert_safe_url
    assert_safe_url(url)                      # raises SSRFError on a blocked target
    ... httpx.get(url, follow_redirects=False)
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

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
