"""Behavioural tests for the guarded fetch (`app.utils.ssrf_guard.safe_fetch_bytes`).

Why this file exists
--------------------
`test_ssrf_guard_coverage` proves nobody is making a raw fetch of a user-influenced URL.
It cannot prove that the thing they call instead is safe. After #15 that helper is the
single fetch path for eleven call sites across admin, internal, RAG, PDF, SAM, interior
design, image classification and aspect search — so a weakness in it is now a weakness in
all of them at once, which is the price of centralising and the reason it has to be
tested rather than reviewed.

The claim under test is specifically the one a reader is most likely to doubt: the helper
FOLLOWS redirects, where every hand-rolled site before it either banned them or followed
them blindly. Following is only defensible because every hop is re-validated, and
"is re-validated" is exactly the kind of statement that is true when written and false two
refactors later.

CI CONSTRAINT (same as test_workspace_resolution.py, which broke the deploy once):
CI installs pytest and NOTHING else. No httpx. So this drives the helper with a stub
client rather than `httpx.MockTransport` — which the helper permits precisely because its
httpx import is lazy and skipped when a client is supplied. The module under test imports
only stdlib, and is loaded by path because tests/unit/test_table_extraction.py registers a
bare ModuleType("app") in sys.modules and collects first.

Addresses in here are IP LITERALS on purpose: `assert_safe_url` calls `getaddrinfo`, which
for a literal parses locally and never touches the network, so these tests need no DNS and
cannot go flaky in a sandboxed runner.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _ROOT / "app" / "utils" / "ssrf_guard.py"
_spec = importlib.util.spec_from_file_location("ssrf_guard_under_test", _MODULE_PATH)
_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_guard)

safe_fetch_bytes = _guard.safe_fetch_bytes
SSRFError = _guard.SSRFError
ResponseTooLarge = _guard.ResponseTooLarge

#: A routable public literal (example.com). Parses as an ordinary global address, so the
#: guard permits it without asking a resolver.
PUBLIC = "https://93.184.216.34/thing.png"
#: The cloud metadata endpoint. The whole point of invariant 7.
METADATA = "https://169.254.169.254/latest/meta-data/iam/security-credentials/"
PRIVATE = "https://10.0.0.5/internal"


class _StubResponse:
    def __init__(self, status_code, headers=None, chunks=()):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._chunks = list(chunks)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _StubStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _StubClient:
    """Records every URL it is asked for, and replays a scripted list of responses.

    `requested` is the assertion surface: a hop that was fetched WITHOUT being validated
    would show up here, and a hop that was rejected before the fetch would not.
    """

    def __init__(self, *responses):
        self._responses = list(responses)
        self.requested = []
        self.follow_redirects_seen = []

    def stream(self, method, url, headers=None, timeout=None, follow_redirects=None):
        self.requested.append(url)
        self.follow_redirects_seen.append(follow_redirects)
        if not self._responses:
            raise AssertionError(f"stub ran out of responses at {url}")
        return _StubStream(self._responses.pop(0))


def _redirect_to(location):
    return _StubResponse(302, {"location": location})


def _body(payload, content_type="image/png"):
    return _StubResponse(
        200,
        {"content-type": content_type, "content-length": str(len(payload))},
        [payload],
    )


def _run(coro):
    return asyncio.run(coro)


# ── The happy path, so the failures below mean something ─────────────────────────


def test_returns_bounded_bytes_and_never_follows_redirects_implicitly():
    client = _StubClient(_body(b"PNGDATA"))
    result = _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))

    assert result.content == b"PNGDATA"
    assert result.content_type == "image/png"
    assert result.status_code == 200
    assert result.final_url == PUBLIC
    assert result.ok

    # Redirect-following is overridden per REQUEST, not left to the client's default, so
    # a client someone constructed with follow_redirects=True cannot smuggle an
    # unvalidated hop past the guard.
    assert client.follow_redirects_seen == [False]


# ── The claim that justifies following redirects at all ──────────────────────────


def test_a_redirect_into_link_local_is_rejected_and_never_fetched():
    """The headline case: hop 1 is fine, hop 2 is the metadata service.

    A blanket `follow_redirects=False` also stops this, but at the cost of breaking every
    provider URL that legitimately redirects — which is why sites kept following blindly
    instead. This asserts the third option actually works: follow, but re-validate.
    """
    client = _StubClient(_redirect_to(METADATA), _body(b"AWS-CREDENTIALS"))

    with pytest.raises(SSRFError):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))

    # Rejected BEFORE the request, not after reading it. If the metadata URL appears
    # here, the guard validated too late and the credentials already left the building.
    assert client.requested == [PUBLIC]


def test_a_redirect_into_rfc1918_is_rejected():
    client = _StubClient(_redirect_to(PRIVATE))
    with pytest.raises(SSRFError):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))
    assert client.requested == [PUBLIC]


def test_a_permitted_redirect_is_followed_and_reports_the_final_url():
    onward = "https://93.184.216.34/cdn/thing.png"
    client = _StubClient(_redirect_to(onward), _body(b"OK"))

    result = _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))

    assert result.content == b"OK"
    assert result.final_url == onward
    assert client.requested == [PUBLIC, onward]


def test_a_relative_location_resolves_against_the_hop_that_issued_it():
    """Relative Locations are legal and common. Resolving them against the ORIGINAL url
    instead of the current hop would validate one address and fetch another."""
    client = _StubClient(_redirect_to("/elsewhere/x.png"), _body(b"OK"))

    result = _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))

    assert result.final_url == "https://93.184.216.34/elsewhere/x.png"
    assert client.requested == [PUBLIC, "https://93.184.216.34/elsewhere/x.png"]


def test_a_redirect_loop_terminates():
    client = _StubClient(*[_redirect_to(PUBLIC) for _ in range(10)])
    with pytest.raises(SSRFError):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, max_redirects=2, client=client))
    # Bounded by max_redirects, not by the stub running dry.
    assert len(client.requested) == 3


def test_a_redirect_with_no_location_is_an_error_not_a_body_read():
    client = _StubClient(_StubResponse(302, {}, [b"whatever"]))
    with pytest.raises(SSRFError):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))


# ── The initial target ───────────────────────────────────────────────────────────


def test_the_first_hop_is_validated_too():
    client = _StubClient(_body(b"CREDS"))
    with pytest.raises(SSRFError):
        _run(safe_fetch_bytes(METADATA, max_bytes=1024, client=client))
    assert client.requested == []


def test_plaintext_http_is_refused_by_default():
    client = _StubClient(_body(b"x"))
    with pytest.raises(SSRFError):
        _run(safe_fetch_bytes("http://93.184.216.34/x", max_bytes=1024, client=client))
    assert client.requested == []


def test_http_can_be_opted_into_for_the_one_legacy_caller():
    """`rag_routes` resumes jobs from pre-migration `documents.file_path` rows, some of
    which are plaintext. Refusing them would make those jobs permanently unresumable."""
    client = _StubClient(_body(b"PDF"))
    result = _run(
        safe_fetch_bytes(
            "http://93.184.216.34/x.pdf",
            max_bytes=1024,
            allow_schemes=("http", "https"),
            client=client,
        )
    )
    assert result.content == b"PDF"


# ── The cap ──────────────────────────────────────────────────────────────────────


def test_the_cap_aborts_mid_stream_rather_than_after_the_whole_body():
    """The failure the hand-rolled copies shared: check `len(resp.content)` AFTER the
    body is already in memory, which rejects the request but not the cost."""
    chunks = [b"A" * 100 for _ in range(50)]
    resp = _StubResponse(200, {"content-type": "image/png"}, chunks)  # no content-length
    client = _StubClient(resp)

    with pytest.raises(ResponseTooLarge):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=250, client=client))


def test_a_declared_length_over_the_cap_fails_before_streaming():
    resp = _StubResponse(200, {"content-length": "999999"}, [b"A"])
    client = _StubClient(resp)
    with pytest.raises(ResponseTooLarge):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))


def test_a_malformed_content_length_does_not_buy_an_unbounded_read():
    """An absent or junk header must not skip the cap — that is how the original image
    downloader read a whole body before consulting its own limit."""
    resp = _StubResponse(200, {"content-length": "not-a-number"}, [b"A" * 5000])
    client = _StubClient(resp)
    with pytest.raises(ResponseTooLarge):
        _run(safe_fetch_bytes(PUBLIC, max_bytes=100, client=client))


def test_too_large_is_catchable_as_ssrf_error():
    """Every existing `except SSRFError` handler is a 'refused for safety' handler, and
    a body that will not fit is refused for the same reason. If ResponseTooLarge stopped
    subclassing it, those handlers would start letting 500s through instead."""
    assert issubclass(ResponseTooLarge, SSRFError)
    assert issubclass(SSRFError, ValueError)


# ── Non-2xx is data, not an exception ────────────────────────────────────────────


def test_a_404_is_returned_not_raised():
    """Call sites have their own messages and their own view of whether a miss is fatal
    — `_generate_sam2_mask` logs and returns None, `restart_job_from_checkpoint` 502s."""
    client = _StubClient(_StubResponse(404, {"content-type": "text/plain"}, [b"nope"]))
    result = _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))
    assert result.status_code == 404
    assert not result.ok
    assert result.content == b"nope"


# ── Guard the guard ──────────────────────────────────────────────────────────────


def test_a_caller_supplied_client_is_not_closed_underneath_the_caller():
    """`_generate_sam2_mask` and `_upload_to_storage` pass a client that is still inside
    its own `async with`. Closing it here would break the poll loop that follows."""
    closed = []

    class _ClosingStub(_StubClient):
        async def aclose(self):
            closed.append(True)

    client = _ClosingStub(_body(b"x"))
    _run(safe_fetch_bytes(PUBLIC, max_bytes=1024, client=client))
    assert closed == []
