"""A multi-brand upload: the flag reaches product creation, and the distributor becomes a supplier offer."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _tree(rel: str) -> ast.Module:
    return ast.parse((_ROOT / rel).read_text(encoding="utf-8"))


def _fn(tree: ast.Module, name: str):
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _params(fn) -> dict:
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults) + list(fn.args.kw_defaults)
    return {a.arg: d for a, d in zip(args, defaults)}


def _calls(node, attr=None, name=None):
    out = []
    for c in ast.walk(node):
        if not isinstance(c, ast.Call):
            continue
        if attr and getattr(c.func, "attr", None) == attr:
            out.append(c)
        if name and getattr(c.func, "id", None) == name:
            out.append(c)
    return out


def _kwargs(call) -> dict:
    return {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}


def test_the_upload_declares_multi_brand_off_by_default():
    upload = _fn(_tree("app/api/rag_routes.py"), "upload_document")
    assert ast.unparse(_params(upload)["multi_brand"]).startswith("Form(False")


def test_the_flag_travels_from_the_upload_to_stage_0_and_to_product_creation():
    routes = _tree("app/api/rag_routes.py")
    runner = _fn(routes, "process_document_with_discovery")
    assert "multi_brand" in _params(runner)
    stage_0_call = next(c for c in _calls(runner, name="process_stage_0_discovery"))
    assert _kwargs(stage_0_call).get("multi_brand") == "multi_brand"

    stage_0 = _fn(_tree("app/api/pdf_processing/stage_0_discovery.py"), "process_stage_0_discovery")
    assert "multi_brand" in _params(stage_0)
    create = next(c for c in _calls(stage_0, name="create_single_product"))
    assert _kwargs(create).get("multi_brand") == "multi_brand"


def test_a_resumed_job_reads_the_flag_back_from_its_own_metadata():
    routes = _tree("app/api/rag_routes.py")
    reads = [
        c for c in _calls(routes, attr="get")
        if ast.unparse(c.func) == "job_meta.get" and c.args and getattr(c.args[0], "value", None) == "multi_brand"
    ]
    assert reads


def test_the_distributor_is_recorded_as_a_supplier_only_for_a_multi_brand_upload():
    stage_0 = _fn(_tree("app/api/pdf_processing/stage_0_discovery.py"), "process_stage_0_discovery")
    gated = [
        n for n in ast.walk(stage_0)
        if isinstance(n, ast.If) and "multi_brand" in ast.unparse(n.test)
        and any(
            getattr(c.args[0], "value", None) == "record_catalog_supplier_offer"
            for c in _calls(n, attr="rpc") if c.args
        )
    ]
    assert gated, "the supplier offer must be written only under `if multi_brand ...`"
    assert "catalog.catalog_factory" in ast.unparse(gated[0].test)


def test_a_single_brand_catalog_still_gets_its_maker_as_the_default_brand():
    create = _fn(_tree("app/api/pdf_processing/stage_4_products.py"), "create_single_product")
    assert "multi_brand" in _params(create)
    defaults = [
        n for n in ast.walk(create)
        if isinstance(n, ast.If) and "catalog.catalog_factory" in ast.unparse(n.test)
    ]
    assert defaults and all("not multi_brand" in ast.unparse(n.test) for n in defaults)


def test_maker_facts_are_not_shared_across_a_multi_brand_catalog():
    propagate = _fn(_tree("app/api/pdf_processing/stage_4_products.py"), "propagate_common_fields_to_products")
    loops = [n for n in ast.walk(propagate) if isinstance(n, ast.For) and "_CATALOGUE_LEVEL_FIELDS" in ast.unparse(n.iter)]
    assert loops and "_job_is_multi_brand" in ast.unparse(loops[0].iter)
