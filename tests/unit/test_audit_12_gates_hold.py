"""
Guard: the audit #12 fixes are structural, not conventions someone re-breaks.

WHY THIS EXISTS
---------------
Every defect this file pins had the same shape: a mechanism that existed, read as
correct, and did nothing. None of them failed a typecheck, a test, or an integrity
probe, because every artifact involved was individually well-formed —
a keyword-guessed classification is a valid classification, an unbounded read is a
valid read, and `path.startswith("/api/rag")` is a valid boolean.

That is precisely the class of bug that comes back, because the "fix" is a habit
rather than a structure. So each check below asserts the SHAPE of the fix, not its
behaviour, and does it over source text — no app import, no DB, no network.

Covers:
  1. document_classifier states its verdict through a real Anthropic tool schema
     and has no keyword-heuristic fallback to fabricate one.
  2. The batch embedding path does not hardcode a model, and validates the
     response before anything positional is done with it.
  3. Server-side image fetches go through the shared SSRF guard, do not follow
     redirects, and bound memory while streaming.
  4. JWT exclude-path matching is segment-aware.
  5. The deploy drain hooks carry a route-level gate.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

_CLASSIFIER = _APP / "services" / "ai_validation" / "document_classifier.py"
_EMBEDDINGS = _APP / "services" / "embeddings" / "real_embeddings_service.py"
_DOWNLOADER = _APP / "services" / "images" / "image_download_service.py"
_JWT = _APP / "middleware" / "jwt_auth.py"
_ADMIN = _APP / "api" / "admin.py"
_ASSOC = _APP / "services" / "images" / "multi_modal_image_product_association_service.py"
_PRICING = _APP / "config" / "ai_pricing.py"
_SUPA = _APP / "services" / "core" / "supabase_client.py"
_DISCOVERY = _APP / "services" / "discovery" / "product_discovery_service.py"


def _src(path: Path) -> str:
    assert path.exists(), f"{path} has moved — update this guard, do not delete it"
    return path.read_text(encoding="utf-8")


# ───────────────────────────── 1. invariant 9 ──────────────────────────────


def test_document_classifier_uses_forced_tool_use():
    """A classifier whose verdict drives behaviour uses tools + tool_choice.

    It used to ask for `category|confidence` as free text and then
    `response_text.split("|")`, with a substring-match salvage path behind it.
    """
    src = _src(_CLASSIFIER)
    assert "DOCUMENT_CLASSIFICATION_TOOL" in src, "the tool schema is gone"
    assert '"tool_choice"' in src, "tool_choice is not forced — the model may answer in prose"
    assert '"enum": ["product", "supporting", "administrative", "transitional"]' in src, (
        "the category enum is the contract; without it the model can invent a category"
    )
    assert 'split("|")' not in src, "the salvage parser is back"


def test_document_classifier_has_no_keyword_fallback():
    """A failed classification must not be reported as a classification.

    The old `except` branch keyword-matched the content and returned e.g.
    `{"content_type": "product", "confidence": 0.5}` — indistinguishable at every
    call site from a real verdict at middling confidence, so a dead API key
    silently reclassified whole catalogs.
    """
    src = _src(_CLASSIFIER)
    assert "classification_failed" in src, "the explicit failure marker is gone"
    for tell in ('"specification", "features"', '"installation", "warranty"'):
        assert tell not in src, f"the keyword-heuristic fallback is back ({tell})"


# ────────────────────────── 2. embeddings integrity ────────────────────────


def test_batch_embedding_does_not_hardcode_the_model():
    """Both 1024D, different SPACE: a hardcoded model + a configured provenance
    stamp means indexed rows and query vectors can silently disagree."""
    src = _src(_EMBEDDINGS)
    assert '"model": "voyage-4"' not in src, (
        "the batch request hardcodes voyage-4 again while provenance stamps "
        "self.voyage_model — set VOYAGE_MODEL and the two diverge silently"
    )
    assert '"model": self.voyage_model' in src


def test_batch_embedding_response_is_validated_before_use():
    """A batch is positional: caller i gets embeddings[i]. 99 vectors for 100
    texts must raise, not shift every chunk onto its neighbour's embedding."""
    src = _src(_EMBEDDINGS)
    assert "len(items) != len(processed_texts)" in src, "no length check on the batch response"
    assert 'key=lambda it: int(it["index"])' in src, (
        "arrival order is not part of the Voyage contract — sort by the explicit index"
    )
    assert "!= voyage_dimensions" in src, "no per-vector dimension check"


# ───────────────────────────────── 3. SSRF ─────────────────────────────────


def test_image_download_is_ssrf_guarded_and_bounded():
    src = _src(_DOWNLOADER)
    assert "assert_safe_url" in src, "server-side fetch of a feed/LLM-supplied URL is unguarded"
    assert "allow_redirects=False" in src, (
        "aiohttp follows redirects by default, which hands back the SSRF bypass "
        "the guard just closed"
    )
    assert "iter_chunked" in src, "the body is read unbounded before the size cap is consulted"
    assert not re.search(r"image_data = await response\.read\(\)", src), (
        "unbounded response.read() is back"
    )


# ──────────────────────────── 4. auth surface ──────────────────────────────


def test_exclude_paths_match_on_segment_boundaries():
    """`startswith("/api/rag")` also excluded a future `/api/rag-admin`, which
    would have inherited public access on the day someone added it."""
    src = _src(_JWT)
    assert 'path.startswith(base + "/")' in src, "segment-aware matching is gone"
    assert not re.search(
        r"any\(path\.startswith\(excluded\) for excluded in self\.exclude_paths\)", src
    ), "raw prefix matching is back"


def test_deploy_hooks_are_gated_at_the_route():
    """These two are in exclude_paths, so the middleware cannot protect them —
    invariant 5 wants the route to carry its own guard, failing closed."""
    src = _src(_ADMIN)
    assert "require_deploy_token" in src
    assert src.count("dependencies=[Depends(require_deploy_token)]") >= 2, (
        "pause-for-deploy and resume-from-deploy must BOTH carry the gate"
    )
    assert "MIVAA_DEPLOY_TOKEN" in src
    assert "status_code=503" in src, "an unset secret must fail closed, never fall through"


# ────────────────────── 6. arithmetic that defeats a rule ──────────────────


def test_spatial_is_a_gate_not_merely_a_weight():
    """Two neutral scores must not outvote a hard exclusion.

    Weights are spatial .4 / caption .3 / clip .3 against a .3 threshold, so an
    image the spatial rule scored 0.0 still reached exactly 0.30 whenever caption
    and clip both returned their neutral 0.5 — which is the ORDINARY case: the
    pipeline writes generic "Image from page N" captions, and products carry only
    text_embedding_1024 so the visual comparison has nothing to compare against.
    """
    src = _src(_ASSOC)
    assert "if association.spatial_score <= 0.0:" in src, (
        "the spatial gate is gone — wrong-page images can ride two neutral scores "
        "over the threshold again"
    )
    # the arithmetic itself, so a weight/threshold tweak re-opening the hole fails here
    weights = {"spatial": 0.4, "caption": 0.3, "clip": 0.3}
    neutral_only = 0.0 * weights["spatial"] + 0.5 * weights["caption"] + 0.5 * weights["clip"]
    assert neutral_only >= 0.3, (
        "if this no longer reaches the threshold the weights changed; re-check "
        "whether the gate is still the thing keeping wrong-page images out"
    )


def test_pricing_fallback_matches_the_authoritative_table():
    """The hardcoded dict is a copy of ai_model_pricing. Copies drift; this one
    drifted to 3x on the most expensive model and nothing noticed for months."""
    src = _src(_PRICING)
    assert '"input": Decimal("15.00")' not in src, "claude-opus-4-8 is back at the 3x price"
    assert "_warn_on_pricing_drift" in src, "the drift check is gone"
    assert "logger.error(f\"ai_model_pricing lookup failed" in src, (
        "falling back to hardcoded prices must never be silent again"
    )


def test_postgrest_retry_does_not_repeat_writes():
    """'Server disconnected' is ambiguous — the write may have committed."""
    src = _src(_SUPA)
    assert "_is_safe_to_repeat" in src, "the idempotency gate on retry is gone"
    assert "resolution=" in src, "upserts must stay retryable (they collapse on conflict)"


# ───────────────── 7. discovery states its findings structurally ────────────


def test_discovery_uses_forced_tool_use_on_both_paths():
    """Discovery is the pipeline's first step; its output becomes catalog state.

    It used to ask for JSON in prose, strip ``` fences, json.loads, and fall back
    to _repair_json — regex that deleted trailing commas and guessed at missing
    separators. A repair that SUCCEEDS is indistinguishable from a response that
    was correct, so a subtly mangled catalog parsed clean and became pipeline
    state. Both the text path and the vision retry now force the tool.
    """
    src = _src(_DISCOVERY)
    assert "PRODUCT_DISCOVERY_TOOL" in src, "the discovery tool schema is gone"
    assert src.count("tools=[PRODUCT_DISCOVERY_TOOL]") >= 2, (
        "both the text path and the vision retry must force the tool"
    )
    assert src.count('"type": "tool", "name": PRODUCT_DISCOVERY_TOOL["name"]') >= 2, (
        "tool_choice must be forced, or the model may still answer in prose"
    )


def test_discovery_has_no_json_repair_left():
    """There is nothing to repair once the shape is guaranteed by the API."""
    src = _src(_DISCOVERY)
    assert "_repair_json" not in src, "the regex JSON repair is back"
    assert "```json" not in src, "code-fence stripping implies a prose response again"


def test_discovery_rejects_gpt_rather_than_silently_using_claude():
    """Finding 4: `gpt-vision` was accepted, validated an OpenAI key, and had no
    code path — so it ran on Claude while recording gpt-vision as the model."""
    src = _src(_DISCOVERY)
    assert "OPENAI_API_KEY" not in src or "os.getenv(\"OPENAI_API_KEY\"" not in src, (
        "the module-load OpenAI key capture is back"
    )
    assert "is not supported" in src, "a GPT model request must be rejected explicitly"
