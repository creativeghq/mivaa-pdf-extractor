"""
Guard: the aspect query vector is derived from the IMAGE when there is one (#277).

The bug: the search page requires an image for its Color/Texture/Style/Material modes, then
sends the image's *filename* as the query text. `multi_vector_search` queried all four aspect
collections with the understanding embedding of that text, so `image_texture_embeddings` was
searched with `voyage("IMG_2831.jpg")` — while `aspect_bias_weights` put 0.55 of the ranking
weight on that channel and cut the SLIG visual channel (the only one that had actually seen
the image) from 0.19 to 0.10. Picking "Texture Pattern" made results worse than not picking it.

Nothing could catch it: a filename embeds to a valid 1024D vector, cosine-compares against
every row, and returns a confident ordering. Right shape, absent meaning.

These are source-level assertions on purpose. Exercising the real path needs Anthropic +
Voyage + VECS; this has to run in CI in a second with nothing but pytest, which is the
difference between a guard that runs on every push and one that quietly never runs. The
behaviour they pin is structural — WHICH derivation each caller reaches for.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ASPECT_QUERY = _ROOT / "app" / "services" / "search" / "aspect_query.py"
_RAG_SERVICE = _ROOT / "app" / "services" / "search" / "rag_service.py"
_SEARCH_API = _ROOT / "app" / "api" / "search.py"

ASPECT_QUERY_SRC = _ASPECT_QUERY.read_text(encoding="utf-8")
RAG_SERVICE_SRC = _RAG_SERVICE.read_text(encoding="utf-8")
SEARCH_API_SRC = _SEARCH_API.read_text(encoding="utf-8")


def test_files_exist():
    """A moved file must fail loudly, not make every assertion vacuously true."""
    for p in (_ASPECT_QUERY, _RAG_SERVICE, _SEARCH_API):
        assert p.exists(), f"{p} not found"


def test_aspects_match_the_serializer_registry():
    """
    ASPECTS is a hand-written mirror of ASPECT_SERIALIZERS' keys (it cannot import the model
    layer without dragging it into every consumer). An aspect added to one and not the other
    means a collection nobody queries, or a query against a collection that does not exist.
    """
    model_src = (_ROOT / "app" / "models" / "vision_analysis.py").read_text(encoding="utf-8")
    block = re.search(r"ASPECT_SERIALIZERS\s*=\s*\{(.*?)\}", model_src, re.DOTALL)
    assert block, "ASPECT_SERIALIZERS registry not found"
    registry_keys = set(re.findall(r"[\"']([a-z_]+)[\"']\s*:", block.group(1)))

    declared = re.search(r"ASPECTS:\s*Tuple\[str, \.\.\.\]\s*=\s*\(([^)]*)\)", ASPECT_QUERY_SRC)
    assert declared, "ASPECTS tuple not found in aspect_query.py"
    declared_keys = set(re.findall(r"[\"']([a-z_]+)[\"']", declared.group(1)))

    assert declared_keys == registry_keys, (
        f"aspect_query.ASPECTS {sorted(declared_keys)} != ASPECT_SERIALIZERS "
        f"{sorted(registry_keys)}"
    )


def test_fusion_derives_aspect_queries_from_the_image():
    """The fix itself: an image present => image-derived aspect vectors."""
    assert "image_query_vectors" in RAG_SERVICE_SRC, (
        "multi_vector_search no longer derives aspect queries from the image — it is back to "
        "searching the aspect collections with an embedding of whatever text was passed, "
        "which on the search page is the image's filename."
    )
    guard = re.search(
        r"if image_base64 and.{0,600}?image_query_vectors",
        RAG_SERVICE_SRC,
        re.DOTALL,
    )
    assert guard, "the image-derived derivation must be reached only when an image is present"


def test_text_only_queries_still_use_the_understanding_embedding():
    """
    The other half. Text queries describe every aspect at once, so one shared vector is the
    right approximation AND stays free — a guard that forced a vision call on every text
    search would 'fix' this bug by making search cost a Claude call per query.
    """
    assert "understanding_embedding" in RAG_SERVICE_SRC
    assert re.search(
        r"aspect_queries\.get\(emb_type\)\s*or\s*understanding_embedding", RAG_SERVICE_SRC
    ), "the per-aspect vector must fall back to the shared text embedding, not to nothing"


def test_one_vision_call_per_image_not_one_per_aspect():
    """
    A VisionAnalysis carries all four aspects' source fields, so four serializers share one
    Claude call. Calling per aspect would quadruple the cost and latency of every aspect
    search for identical output.
    """
    fn = ASPECT_QUERY_SRC[ASPECT_QUERY_SRC.index("async def image_query_vectors"):]
    assert fn.count("analyze_query_image") == 1, (
        "image_query_vectors must analyze the image exactly once"
    )


def test_a_missing_aspect_never_borrows_another_aspects_vector():
    """
    An image with no discernible pattern yields no texture string. That aspect must be absent
    from the result — substituting a neighbouring aspect's vector would answer a different
    question with full confidence, which is the whole failure family this module exists for.
    """
    fn = ASPECT_QUERY_SRC[ASPECT_QUERY_SRC.index("async def image_query_vectors"):]
    assert "continue" in fn, "a missing/failed channel must be skipped, not substituted"
    assert "embeddings[channel] = vec" in fn


def test_understanding_is_filled_from_the_image_too():
    """
    `understanding` is the heaviest single channel in the balanced profile (18%) and it was
    fed the caller's query TEXT — which on the search page is the image's filename. The vision
    analysis run for the aspects already describes the whole image, so this channel costs one
    extra Voyage embed and no extra Claude call.
    """
    assert "serialize_vision_analysis_to_text" in ASPECT_QUERY_SRC, (
        "the understanding query vector must use the SAME serializer ingestion used, or it "
        "lands outside image_understanding_embeddings' space"
    )
    assert re.search(
        r'image_query_vecs\.get\("understanding"\)', RAG_SERVICE_SRC
    ), "the understanding channel must prefer the image-derived vector when there is one"


def test_text_only_channels_stand_down_when_there_is_no_text():
    """
    The core of the fix. With no words, chunk/product/keyword have nothing to search WITH.
    Disabling beats embedding the empty-or-fabricated string: a filename embeds to a valid
    vector and ranks every row against it, which is confidently wrong rather than absent.
    """
    assert "has_text_query" in RAG_SERVICE_SRC, "the no-text case must be detected explicitly"
    block = re.search(
        r"if not has_text_query:(.{0,400}?)self\.logger\.info", RAG_SERVICE_SRC, re.DOTALL
    )
    assert block, "expected an `if not has_text_query:` block disabling the text channels"
    body = block.group(1)
    for channel in ("enable_chunk", "enable_product", "enable_keyword"):
        assert f"{channel} = False" in body, f"{channel} must be disabled when there is no text"


def test_the_page_channel_needs_words():
    """
    generate_page_query_embedding is a TEXT entry point into multimodal space. Handing it a
    filename produces a vector that ranks pages against nothing in particular — and page
    carries 10% of the balanced profile.
    """
    fn_at = RAG_SERVICE_SRC.index("async def _get_page_embedding")
    fn = RAG_SERVICE_SRC[fn_at:fn_at + 1200]
    assert "if not has_text_query" in fn, (
        "the page channel must return None rather than embed a fabricated query"
    )


def test_query_image_fetch_goes_through_the_ssrf_guard():
    """
    Invariant 7. This helper fetches a user-supplied https URL; it previously used a raw
    httpx.get with follow_redirects=True, so a permitted host could 302 into link-local
    metadata. Wiring these endpoints up made that reachable.
    """
    assert "assert_safe_url" in ASPECT_QUERY_SRC, "user-supplied query_image URL must be validated"
    assert "follow_redirects=False" in ASPECT_QUERY_SRC, (
        "redirects must be disabled — validating only the first URL is not enough"
    )
    assert "_MAX_IMAGE_BYTES" in ASPECT_QUERY_SRC, "an unbounded fetch is a memory DoS"


def test_the_endpoints_delegate_rather_than_keeping_a_second_copy():
    """
    Two callers need this chain. Two implementations of it would be two things to keep inside
    the collections' latent space, and the drift would be invisible — both produce valid 1024D
    vectors either way.
    """
    assert "from app.services.search.aspect_query import aspect_query_embedding" in SEARCH_API_SRC
    assert "api.anthropic.com" not in SEARCH_API_SRC, (
        "search.py re-grew its own vision call — the aspect derivation must have exactly one home"
    )


@pytest.mark.parametrize("forbidden", ["voyage-3", "openai", "text-embedding"])
def test_no_second_embedder_creeps_in(forbidden):
    """The aspect collections ARE Voyage 1024D space; a same-dimension vector from another
    model is stored, compared and ranked without anything raising."""
    assert forbidden not in ASPECT_QUERY_SRC.lower()


def test_image_hits_are_resolved_to_something_a_caller_can_use():
    """
    VECS answers an image search with a UUID and a score. That is a correct ranking in an
    unusable form — an agent cannot describe it and a UI cannot render it — and it reports
    success with a populated `results` array, so nothing looks wrong.

    `/api/rag/search?strategy=multi_vector` never had this problem: it resolves to products
    and the route enriches them, keying on `result.get('id')`. Image rows have no `id`, so
    that enrichment silently skipped every one of them.

    Both bare-row endpoints must go through the shared enrichment.
    """
    enrich_src = (_ROOT / "app" / "services" / "search" / "image_results.py").read_text(
        encoding="utf-8"
    )
    assert "def enrich_image_rows" in enrich_src

    # Tenancy: the ids come back from a vector store, so re-scoping here is what stops a
    # caller reading the caption and URL of another tenant's image (invariant 1).
    assert 'eq("workspace_id", workspace_id)' in enrich_src, (
        "enrichment must filter by workspace — an unscoped lookup leaks captions and URLs"
    )
    assert "if ids and not workspace_id" in enrich_src, (
        "a missing workspace_id must refuse loudly, not enrich everything"
    )

    for label, src in (("aspect endpoint", SEARCH_API_SRC), ("image_similarity_search", RAG_SERVICE_SRC)):
        assert "enrich_image_rows" in src, f"{label} still returns bare image rows"


def test_enrichment_is_batched_not_per_row():
    """Two queries for the whole result set. Per-row lookups would make a 50-result search
    101 round trips, which is how an enrichment gets removed again for being slow."""
    enrich_src = (_ROOT / "app" / "services" / "search" / "image_results.py").read_text(
        encoding="utf-8"
    )
    assert enrich_src.count('.in_(') == 2, "expected exactly two batched .in_() lookups"
    body = enrich_src[enrich_src.index("async def enrich_image_rows"):]
    assert "for r in ids" not in body and "for image_id in ids" not in body, (
        "no per-id query loop"
    )
