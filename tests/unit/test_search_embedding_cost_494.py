"""Guards for #494 work package M5: search filters, embeddings, cost logging, backfills.

Source-based or import-free: CI installs pytest and nothing else, so nothing here
imports `app` (its package __init__ reaches for a Supabase client).
"""

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"
_RAG = _APP / "services" / "search" / "rag_service.py"
_UNIFIED = _APP / "services" / "search" / "unified_search_service.py"
_LOGGER = _APP / "services" / "core" / "ai_call_logger.py"
_EMB = _APP / "services" / "embeddings" / "real_embeddings_service.py"
_VECS = _APP / "services" / "embeddings" / "vecs_service.py"
_PAGE = _APP / "services" / "embeddings" / "page_embedding_service.py"
_UNDERSTANDING = _APP / "services" / "embeddings" / "understanding_backfill.py"
_ASPECT = _APP / "services" / "embeddings" / "aspect_backfill.py"
_TEXT = _APP / "services" / "embeddings" / "text_embedding_backfill.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def _code(path: Path) -> str:
    """Source with comments and docstrings blanked: a marker in prose satisfies nothing."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    blank = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            blank.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in blank or line.lstrip().startswith("#"):
            out.append("")
        else:
            out.append(line.split("  #", 1)[0])
    return "\n".join(out)


def _function(path: Path, name: str) -> str:
    code = _code(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(code.split("\n")[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"{name} not found in {path.name}")


filters = _load(_APP / "services" / "search" / "search_filters.py", "search_filters_494")
tool_call = _load(_APP / "services" / "core" / "claude_tool_call.py", "claude_tool_call_494")


# ── Search filters are predicates ────────────────────────────────────────────


def test_attribute_filters_become_predicates_on_attributes():
    preds, applied, ignored = filters.build_filter_predicates(
        {"attributes.color": ["beige", "white"], "attributes.finish": "matte"}
    )
    assert preds == [
        "attributes->>color.ilike.*beige*,attributes->>color.ilike.*white*",
        "attributes->>finish.ilike.*matte*",
    ]
    assert applied == {"attributes.color": ["beige", "white"], "attributes.finish": ["matte"]}
    assert ignored == {}


def test_category_filter_reads_the_column_the_attribute_and_the_registry_id():
    preds, applied, _ = filters.build_filter_predicates(
        {"categories": ["tiles"]}, {"tiles": "11111111-1111-1111-1111-111111111111"}
    )
    assert preds == [
        "category.ilike.tiles,attributes->>material_category.ilike.tiles,"
        "category_id.in.(11111111-1111-1111-1111-111111111111)"
    ]
    assert applied == {"categories": ["tiles"]}


def test_free_metadata_keys_are_reported_not_silently_boosted():
    preds, applied, ignored = filters.build_filter_predicates(
        {"design.designers": "Ponti", "dimensions": {"gte": 60}}
    )
    assert preds == [] and applied == {}
    assert set(ignored) == {"design.designers", "dimensions"}


def test_filter_values_cannot_break_out_of_the_logic_tree():
    assert filters.filter_values('red),id.not.is.null,(x') == ["red id.not.is.null x"]
    assert filters.filter_values("a*b") == ["a b"]
    assert filters.filter_values({"contains": ["oak", None, "oak"]}) == ["oak"]


def test_predicate_tree_requires_every_group():
    assert filters.predicate_tree([]) is None
    assert filters.predicate_tree(["a.eq.1,b.eq.2", "c.eq.3"]) == "and(or(a.eq.1,b.eq.2),or(c.eq.3))"


def test_the_score_boost_is_gone_and_attributes_are_selected():
    body = _function(_RAG, "multi_vector_search")
    assert "filter_boost" not in body
    assert "+= 0.15" not in body
    assert "_filtered_product_ids" in body
    assert "attributes" in body


def test_multi_vector_queries_the_slig_collection_on_its_own_vector():
    body = _function(_RAG, "multi_vector_search")
    assert "search_similar_images(" in body
    assert "query_embedding=visual_embedding" in body
    assert "if enable_visual and visual_embedding:" not in body


def test_search_failures_raise_rather_than_return_an_empty_list():
    for name in ("multi_vector_search", "material_property_search", "image_similarity_search"):
        tree = ast.parse(textwrap.dedent(_function(_RAG, name)))
        func = tree.body[0]
        outer_try = [n for n in func.body if isinstance(n, ast.Try)][-1]
        outer = outer_try.handlers[-1]
        raises = [n for n in ast.walk(outer) if isinstance(n, ast.Raise)]
        returns = [n for n in ast.walk(outer) if isinstance(n, ast.Return)]
        assert raises and not returns, f"{name} swallows its failure into a result"
    for name in ("_search_chunks_by_embedding", "_search_products_by_embedding",
                 "_search_products_fulltext"):
        assert "raise SearchFailedError" in _function(_RAG, name)


def test_query_understanding_has_a_client_for_alias_resolution():
    init = _function(_UNIFIED, "__init__")
    assert "get_supabase_client()" in init


# ── Embeddings ───────────────────────────────────────────────────────────────


def test_search_embeds_queries_as_queries():
    body = _function(_RAG, "multi_vector_search")
    assert 'input_type="query"' in body
    assert 'input_type="query"' in _function(_RAG, "_retrieve_context_chunks")
    gen = _function(_EMB, "generate_text_embedding")
    assert "input_type=input_type" in gen


def test_page_embedding_is_attributed():
    body = _function(_PAGE, "_embed_one_page")
    assert "workspace_id=str(workspace_id)" in body and "user_id=user_id" in body


def test_vecs_writes_run_off_the_event_loop():
    for name in ("upsert_image_embedding", "upsert_specialized_embeddings",
                 "upsert_understanding_embedding", "upsert_page_embedding"):
        body = _function(_VECS, name)
        assert "asyncio.to_thread(collection.upsert" in body, name
        assert "\n            collection.upsert(" not in body, name


# ── Cost logging ─────────────────────────────────────────────────────────────


def test_dedup_keys_on_the_invocation_not_on_its_figures():
    body = _function(_LOGGER, "log_ai_call")
    assert "hashlib" not in _code(_LOGGER)
    assert "idem_key = invocation_id" in body


def test_caller_priced_cost_and_pixels_reach_the_usage_row():
    body = _function(_LOGGER, "log_ai_call")
    assert "cost_breakdown or ai_pricing.calculate_cost" in body
    assert '"image_pixels"' in body
    mm = _function(_EMB, "_voyage_multimodal_embed")
    assert "cost_breakdown=" in mm and "image_pixels=image_pixels" in mm


def test_a_debited_call_completes_the_debit_row_instead_of_adding_one():
    body = _function(_LOGGER, "log_ai_call")
    assert "if usage_log_id:" in body and "_complete_debit_usage_row" in body
    for caller in ("log_claude_call", "log_time_based_call", "log_replicate_call"):
        assert "usage_log_id=usage_log_id" in _function(_LOGGER, caller), caller


def test_failed_gpu_calls_carry_their_error():
    assert "error_message=error_message" in _function(_LOGGER, "log_time_based_call")
    siglip = _function(_EMB, "_generate_siglip_embedding")
    assert siglip.count("error_message=") >= 2


def _tool_response(stop_reason="tool_use", tool_input=None, name="emit"):
    blocks = [] if tool_input is None else [
        SimpleNamespace(type="tool_use", name=name, input=tool_input)
    ]
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


def test_a_refusal_is_a_failed_call():
    assert tool_call.claude_reply_failure(_tool_response("refusal"), None).startswith("refusal")


def test_a_missing_forced_tool_block_is_a_failed_call():
    spec = {"structured_output": {"kind": "tool", "name": "emit", "required": []}}
    assert "invalid structured output" in tool_call.claude_reply_failure(_tool_response(), spec)


def test_a_required_field_left_empty_is_a_failed_call():
    spec = {"structured_output": {"kind": "tool", "name": "emit", "required": ["verdict"]}}
    reply = _tool_response(tool_input={"verdict": ""})
    assert "verdict" in tool_call.claude_reply_failure(reply, spec)


def test_a_valid_reply_is_not_a_failure():
    spec = {"structured_output": {"kind": "tool", "name": "emit", "required": ["verdict"]}}
    assert tool_call.claude_reply_failure(_tool_response(tool_input={"verdict": "ok"}), spec) is None


def test_every_entry_point_declares_its_output_contract():
    code = _code(_APP / "services" / "core" / "claude_tool_call.py")
    assert code.count("request_data=_structured_output_spec(") == 3
    assert "claude_reply_failure(response, request_data)" in _function(_LOGGER, "log_claude_call")


# ── Backfills ────────────────────────────────────────────────────────────────


def test_understanding_backfill_persists_the_analysis_before_the_vector():
    body = _function(_UNDERSTANDING, "backfill_understanding_embeddings")
    assert body.index("_persist_vision_analysis(") < body.index("upsert_understanding_embedding(")
    rollup = _function(_UNDERSTANDING, "_trigger_product_rollup_recompute")
    assert "supabase=supabase," in rollup and "supabase=client" not in rollup
    assert rollup.index("rerolled += 1") > rollup.index('stats.get("error")')


def test_backfills_filter_stale_rows_before_the_limit():
    fetch = _function(_UNDERSTANDING, "_fetch_stale_images")
    assert fetch.index(".or_(") < fetch.index(".limit(")
    aspect = _function(_ASPECT, "_fetch_stale_aspect_images")
    assert aspect.index(".or_(stale_aspect_rows_filter())") < aspect.index(".limit(")
    chunks = _function(_TEXT, "_fetch_chunks_missing_embedding")
    assert chunks.index('.neq("content", "")') < chunks.index(".limit(")


def test_an_empty_aspect_is_recorded_absent_and_not_retried():
    src = _ASPECT.read_text(encoding="utf-8")
    ns: dict = {}
    tree = ast.parse(src)
    wanted = {"_is_aspect_stale", "_aspect_stale_filter"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(_ASPECT), "exec"), {
        "SCHEMA_VERSION": 3, "ABSENT_ASPECT_MODEL": "absent", "Dict": dict, "Any": object,
    }, ns)
    fresh_absent = {"has_color_slig": False, "color_aspect_schema_version": 3,
                    "color_aspect_embedding_model": "absent"}
    assert ns["_is_aspect_stale"](fresh_absent, "color") is False
    assert ns["_is_aspect_stale"]({**fresh_absent, "color_aspect_schema_version": 2}, "color") is True
    assert ns["_is_aspect_stale"]({"has_color_slig": False}, "color") is True
    assert "color_aspect_embedding_model.neq.absent" in ns["_aspect_stale_filter"]("color")


@pytest.mark.parametrize("path,fn", [
    (_TEXT, "backfill_product_text_embeddings"),
    (_UNDERSTANDING, "backfill_understanding_embeddings"),
    (_ASPECT, "_generate_aspects_for_row"),
])
def test_paid_backfill_calls_are_attributed(path, fn):
    assert 'workspace_id=row.get("workspace_id")' in _function(path, fn)
