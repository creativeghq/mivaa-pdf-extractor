"""Every search route ranks on the multi-vector score and reports that score, never a blend."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _code(path: str) -> str:
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    return budget.blank_comments((_ROOT / path).read_text(encoding="utf-8"))


def test_the_material_visual_route_does_not_blend_product_relevance_into_the_score():
    src = _code("app/services/search/material_visual_search_service.py")
    assert "* 0.35" not in src and "*0.35" not in src
    enrich = src[src.index("enrich_image_results("):]
    assert "rerank=False" in enrich[:enrich.index(")")]
    assert "text_query=not (request.query_embedding or request.query_image)" in src


def test_the_knowledge_base_route_reports_the_score_multi_vector_search_returns():
    src = _code("app/api/rag_routes.py")
    assert "result.get('weighted_score'" not in src
    kb = src[src.index("Searching products with multi-vector search"):]
    kb = kb[:kb.index("relevance_score")]
    assert 'search_config={"weights": text_query_source_weights()}' in kb


def test_multi_vector_search_uses_an_image_url_through_the_ssrf_guard():
    src = _code("app/api/rag_routes.py")
    branch = src[src.index('if strategy == "multi_vector":'):src.index('elif strategy == "material":')]
    assert "safe_fetch_bytes(request.image_url" in branch
    assert "image_base64=query_image_b64" in branch
    assert "get_profile(DEFAULT_PROFILE)" in branch


def test_inferred_query_filters_reach_only_the_material_strategy():
    src = _code("app/api/rag_routes.py")
    start = src.index("_parse_query_with_ai(query_to_use")
    qu = src[start:src.index("except asyncio.TimeoutError:", start)]
    gate = qu.index('if strategy == "material":')
    assert "query_to_use = visual_query" in qu[gate:]
    assert "query_to_use = visual_query" not in qu[:gate]
    assert "request.material_filters =" in qu[gate:]
    assert "request.material_filters =" not in qu[:gate]
