"""A credited designer is never a product's maker, so a studio never becomes a brand company."""

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_STAGE_4 = _ROOT / "app/api/pdf_processing/stage_4_products.py"

STUDIO, PERSON, CATALOG_MAKER, REAL_MAKER = "Acme Studio", "Jane Doe", "Maker Co", "Other Brand"


def _tree() -> ast.Module:
    return ast.parse(_STAGE_4.read_text(encoding="utf-8"))


def _load() -> Dict[str, Any]:
    spec = importlib.util.spec_from_file_location("text_fold_under_test", _ROOT / "app/utils/text_fold.py")
    fold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fold)

    wanted = {"_designer_names", "_maker_is_a_designer_credit", "_resolve_product_maker"}
    consts = {"_CREDIT_KEYS", "_CREDIT_SPLIT"}
    body = _tree().body
    funcs = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assigns = [n for n in body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) in consts for t in n.targets)]
    assert {f.name for f in funcs} == wanted and len(assigns) == len(consts)
    scope: Dict[str, Any] = {"fold_identity": fold.fold_identity, "re": re, "Dict": Dict, "Any": Any, "Optional": Optional}
    exec(compile(ast.Module(body=assigns + funcs, type_ignores=[]), str(_STAGE_4), "exec"), scope)
    return scope


CREDIT_SHAPES = [
    f"{STUDIO} / {PERSON}",
    f"{PERSON} / {STUDIO}",
    f"{STUDIO} & {PERSON}",
    f"{STUDIO}, {PERSON}",
    f"{STUDIO} + {PERSON}",
    f"{STUDIO} x {PERSON}",
    f"{STUDIO} and {PERSON}",
    f"{STUDIO.upper()} / {PERSON}",
    STUDIO,
]


@pytest.mark.parametrize("credit", CREDIT_SHAPES)
def test_a_studio_inside_any_credit_shape_is_not_a_maker(credit):
    assert _load()["_maker_is_a_designer_credit"](STUDIO, {"designer": credit})


@pytest.mark.parametrize("key", ["designer", "designers", "studio", "design_studio", "designer_studio", "designer_name", "designed_by"])
def test_every_credit_field_counts_as_a_credit(key):
    is_credit = _load()["_maker_is_a_designer_credit"]
    assert is_credit(STUDIO, {key: STUDIO})
    assert is_credit(STUDIO, {key: [PERSON, STUDIO]})


def test_a_real_maker_is_not_mistaken_for_a_credit():
    is_credit = _load()["_maker_is_a_designer_credit"]
    assert not is_credit(REAL_MAKER, {"designer": f"{STUDIO} / {PERSON}"})
    assert not is_credit(REAL_MAKER, {})
    assert not is_credit(None, {"designer": STUDIO})
    assert not is_credit("", {"designer": ""})


@pytest.mark.parametrize("extracted", [STUDIO, PERSON, REAL_MAKER, None, CATALOG_MAKER.upper()])
def test_when_the_catalog_states_its_maker_it_is_every_products_maker(extracted):
    resolve = _load()["_resolve_product_maker"]
    assert resolve(extracted, CATALOG_MAKER, {"designer": f"{STUDIO} / {PERSON}"}) == CATALOG_MAKER


def test_with_no_catalog_maker_a_credit_is_dropped_and_a_real_maker_kept():
    resolve = _load()["_resolve_product_maker"]
    assert resolve(STUDIO, None, {"designer": f"{STUDIO} / {PERSON}"}) is None
    assert resolve(REAL_MAKER, None, {"designer": f"{STUDIO} / {PERSON}"}) == REAL_MAKER
    assert resolve(None, None, {}) is None


def test_the_resolved_maker_is_set_before_any_company_can_be_minted():
    """The property that matters: the maker is resolved BEFORE resolve_brand_company reads it."""
    creator = next(
        n for n in ast.walk(_tree())
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "rpc"
            and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "resolve_brand_company"
            for c in ast.walk(n)
        )
    )
    resolve_lines = [
        c.lineno for c in ast.walk(creator)
        if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_resolve_product_maker"
    ]
    rpc_lines = [
        c.lineno for c in ast.walk(creator)
        if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "rpc"
        and c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "resolve_brand_company"
    ]
    assert resolve_lines, "the company-minting function no longer resolves the maker first"
    assert max(resolve_lines) < min(rpc_lines)
