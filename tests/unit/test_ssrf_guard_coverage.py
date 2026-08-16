"""Guard: every server-side fetch of a USER-INFLUENCED URL goes through the SSRF guard.

Why this file exists, and why it is not another entry in test_audit_12_gates_hold
--------------------------------------------------------------------------------
`test_audit_12_gates_hold::test_image_download_is_ssrf_guarded_and_bounded` reads as
though it covers invariant 7. It checks exactly ONE declared file
(`image_download_service.py`). Issue #15 measured that gap against the defects found in
#14 and #15 and it was the same story for every narrow guard in the repo — the guard
names the class, scans a list, and the defects live in the difference:

    test_paid_route_metering       3 hardcoded doors in 2 files
    test_no_silent_degradation     5 declared subtrees (now the whole tree)
    test_audit_12_gates_hold       1 declared file  <- this one
    test_kb_chunk_retrieval        3 declared filenames, one since deleted

The correlation that settles it: `test_workspace_binding_coverage`, the ONE guard using
full-tree `rglob`, covers the class with the least evidence of defects.

Widening this one immediately found a live hole the declared-file version could never
see: `material_visual_search_service._perform_database_search` fetched
`request.query_image` — a URL straight off the request body — with a bare
`requests.get(...)`, while `aspect_query._resolve_image_base64` did the same job for the
same field correctly, guard and all. That is now fixed.

What it checks
--------------
A call is in scope when all three hold:
  1. it is an HTTP fetch on an http-client receiver (httpx / aiohttp / requests / a
     session or client object), and
  2. its URL argument is an EXPRESSION, not a literal — a literal endpoint is a
     deployer's choice, not an attacker's, and
  3. that expression names a value that reaches us from outside: a request body, an LLM
     response, a supplier feed, or a DB column written from one of those.

The enclosing function must then mention `assert_safe_url`. Function-level rather than
line-level on purpose: the guard legitimately runs a few lines above the fetch, and
demanding adjacency would push people to satisfy the test rather than the invariant.

In practice almost nothing should trip this now, because almost nothing should be making
the raw call: `app.utils.ssrf_guard.safe_fetch_bytes` performs the fetch itself, so a
guarded site has no `client.get(...)` for this scanner to find. That is the intended end
state — the detector is here for the next person who reaches for httpx directly.

THERE IS NO ALLOWLIST
---------------------
#15 shipped `KNOWN_UNGUARDED` with nine named entries, and all nine are now fixed. The
set was deleted rather than emptied, for the reason set out where it used to live.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

_FETCH_METHODS = {"get", "post", "put", "head", "stream", "request"}

#: The receiver has to look like an HTTP client, or `dict.get()` would match everything.
_CLIENT_RECEIVER = re.compile(r"httpx|aiohttp|requests|session|client|_c\b", re.I)

#: Identifiers naming a URL that reaches us from outside our own config.
_USER_INFLUENCED = re.compile(
    r"(image_url|albedo_url|mask_url|file_url|file_path|storage_url|query_image|"
    r"feed_url|pdf_url|source_url|page_url|target_url|logo_url|photo_url|"
    r"document_url|media_url|thumbnail_url)",
    re.I,
)

# THERE IS NO ALLOWLIST ANY MORE.
#
# #15 shipped `KNOWN_UNGUARDED` with nine entries — the sites widening had surfaced,
# named honestly rather than hidden by a narrow scan. All nine are now fixed, and the
# set is DELETED rather than left empty, which is the same call the frontend made for
# `KNOWN_UNCLUSTERED` / `KNOWN_UNBOUND`: an empty allowlist is a furnished place to put
# the next exception, and the next exception is always "just this one, it's a provider
# URL". With the name gone, adding one means re-introducing the mechanism on purpose
# and writing down why.
#
# What the nine turned out to be, since this list was the only record of them:
#   - Five DB-column URLs (`document_images.image_url`, `documents.file_path`, a
#     `storage_url` on the classification path, a `pdf_url`) — attacker-influenced at
#     one remove, through PDF extraction or a supplier feed.
#   - Four provider-output URLs from Replicate / Gemini / Modal responses.
#   - And one that #15 recorded as NOT being a request-body URL, wrongly:
#     `interior_design_routes.derive_product_pbr_maps` fetches `body.image_url` straight
#     off the request. It also read AND wrote `products` on a body-supplied id with no
#     workspace check, against a service-role client. Both fixed.
#
# They are all fixed the same way, which is the other half of the lesson: nine sites
# held nine opinions about how much of invariant 7 to implement, and each one looked
# fine on its own. The guard now owns the FETCH as well as the check —
# `app.utils.ssrf_guard.safe_fetch_bytes` validates the scheme and resolved address of
# every redirect hop and caps the body while streaming — so a raw client call on a
# user-influenced URL is now the thing that is wrong, rather than the thing that needs
# a correct-looking comment in its enclosing function.

#: NOT matched, deliberately: a bare `url` identifier. Nearly every one in this tree is a
#: module constant or an f-string over a configured base (ANTHROPIC_API, DATAFORSEO_*,
#: `{self.platform_url}/...`) — 93 dynamic-URL fetches across 43 files, of which the
#: overwhelming majority are vendor endpoints. Including `url` would drown the signal and
#: the guard would be muted within a week, which is the failure mode this whole file
#: exists to argue against. The named forms above (`image_url`, `query_image`,
#: `file_url`, …) are the ones that carry outside data, and they are what invariant 7 is
#: about. If a bare `url` ever does carry caller input, name the variable for what it is.
#:
#: `test_the_allowlist_does_not_rot` enforces the consequence: an entry that stops firing
#: must be deleted, so this exclusion cannot quietly grow a shadow allowlist.


def _enclosing_function(tree: ast.AST, lineno: int):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= (node.end_lineno or node.lineno):
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def _scan() -> tuple[list[tuple[str, str, int, str]], int]:
    """Return (unguarded_sites, files_scanned)."""
    unguarded: list[tuple[str, str, int, str]] = []
    scanned = 0

    for path in sorted(APP.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            src = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Retry ONCE before failing. A file caught mid-write by an editor parses on
            # the second read; a genuinely broken one fails both times. Without this the
            # guard is flaky in a working copy someone is editing — and a flaky guard
            # gets deleted, which is the outcome this whole file argues against. It is
            # not a weakening: an unparseable file is still a hard failure, because a
            # file this scanner cannot read is a file it is not covering.
            try:
                src = path.read_text(encoding="utf-8")
                tree = ast.parse(src)
            except (OSError, SyntaxError):
                pytest.fail(f"{rel} does not parse; this guard is blind to it")
        scanned += 1

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr not in _FETCH_METHODS:
                continue
            if not _CLIENT_RECEIVER.search(ast.unparse(func.value)):
                continue
            if not node.args or isinstance(node.args[0], ast.Constant):
                continue

            url_expr = ast.unparse(node.args[0])
            if not _USER_INFLUENCED.search(url_expr):
                continue
            # A name that already says it was checked is the guard's own output.
            if "safe_url" in url_expr:
                continue

            enclosing = _enclosing_function(tree, node.lineno)
            body = (ast.get_source_segment(src, enclosing) if enclosing else src) or ""
            if "assert_safe_url" in body:
                continue

            unguarded.append(
                (rel, enclosing.name if enclosing else "<module>", node.lineno, url_expr)
            )

    return unguarded, scanned


def test_no_new_unguarded_user_url_fetch():
    unguarded, scanned = _scan()

    # A scanner that walked nothing reports clean. Refuse to be that — this is the
    # failure mode that let the one-file version pass for as long as it did.
    assert scanned > 100, (
        f"only {scanned} files scanned under app/ — the walk is broken and this guard "
        "is vacuous"
    )

    assert not unguarded, (
        "server-side fetch of a user-influenced URL that is neither guarded nor routed "
        "through the guarded fetch (invariant 7):\n"
        + "\n".join(
            f"  {rel}:{lineno} in {fn}() — {expr}" for rel, fn, lineno, expr in unguarded
        )
        + "\n\nReplace the raw client call with:\n"
        "    from app.utils.ssrf_guard import safe_fetch_bytes, MAX_IMAGE_BYTES\n"
        "    result = await safe_fetch_bytes(the_url, max_bytes=MAX_IMAGE_BYTES)\n"
        "It validates the scheme and resolved address of the URL AND of every redirect "
        "hop, and aborts mid-stream at the cap. Do NOT add an allowlist here — there is "
        "deliberately nowhere left to put one."
    )


def test_the_detector_catches_the_shape_it_was_built_for():
    """Guard the guard. An empty result from a detector never shown to fire proves nothing.

    This reconstructs the exact pre-fix expression from
    `material_visual_search_service._perform_database_search` and asserts it is flagged.
    """
    offending = (
        "import requests\n"
        "def _perform_database_search(request):\n"
        "    if request.query_image.startswith('http'):\n"
        "        response = requests.get(request.query_image, timeout=30)\n"
        "    return response\n"
    )
    tree = ast.parse(offending)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in _FETCH_METHODS:
            continue
        if not _CLIENT_RECEIVER.search(ast.unparse(func.value)):
            continue
        if not node.args or isinstance(node.args[0], ast.Constant):
            continue
        expr = ast.unparse(node.args[0])
        if _USER_INFLUENCED.search(expr) and "assert_safe_url" not in offending:
            found.append(expr)

    assert found == ["request.query_image"], (
        "the detector no longer catches a bare requests.get(request.query_image) — it "
        "has been narrowed into uselessness"
    )
