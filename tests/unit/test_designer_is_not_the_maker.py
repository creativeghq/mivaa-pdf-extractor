"""A credited designer is never the maker, so no studio becomes a brand company."""

import ast
import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
_STAGE_4 = _ROOT / "app/api/pdf_processing/stage_4_products.py"


def _load_helpers():
    spec = importlib.util.spec_from_file_location("text_fold_under_test", _ROOT / "app/utils/text_fold.py")
    fold = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fold)

    tree = ast.parse(_STAGE_4.read_text(encoding="utf-8"))
    wanted = {"_designer_names", "_maker_is_a_designer_credit"}
    consts = {"_CREDIT_KEYS", "_CREDIT_SPLIT"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assigns = [
        n for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) in consts for t in n.targets)
    ]
    assert {f.name for f in funcs} == wanted and len(assigns) == len(consts)
    scope: Dict[str, Any] = {
        "fold_identity": fold.fold_identity, "re": re, "Dict": Dict, "Any": Any, "Optional": Optional,
    }
    exec(compile(ast.Module(body=assigns + funcs, type_ignores=[]), str(_STAGE_4), "exec"), scope)
    return scope["_maker_is_a_designer_credit"]


def test_the_studio_credited_as_designer_is_not_a_maker():
    is_credit = _load_helpers()
    assert is_credit("MUT", {"designer": "MUT"})
    assert is_credit("Estudi{H}ac", {"designers": ["Estudi{H}ac"]})
    assert is_credit("  dsignio ", {"designer": "Dsignio"})


def test_a_studio_and_person_credit_is_split_and_the_studio_is_still_caught():
    is_credit = _load_helpers()
    assert is_credit("Estudi{H}ac", {"designer": "Estudi{H}ac / José Manuel Ferrero", "designers": "José Manuel Ferrero"})
    assert is_credit("Onset", {"designer": "ONSET / Francisco Segarra"})
    assert is_credit("ALT Design", {"designer": "ALT Design / Alexandre Touguet"})
    assert is_credit("Estudi{H}ac", {"studio": "estudi{H}ac"})
    assert is_credit("Dsignio", {"design_studio": "Dsignio & Partners"})


def test_the_catalog_maker_next_to_a_named_designer_is_not_swallowed():
    is_credit = _load_helpers()
    assert not is_credit("HARMONY", {"designer": "Stacy Garcia (SG NY)", "designers": "Stacy Garcia"})
    assert not is_credit("Harmony", {"designer": "Fran Silvestre", "designers": ["Fran Silvestre"]})


def test_the_real_maker_and_unrelated_names_pass():
    is_credit = _load_helpers()
    assert not is_credit("Harmony", {"designer": "MUT"})
    assert not is_credit("Harmony", {})
    assert not is_credit(None, {"designer": "MUT"})
    assert not is_credit("", {"designer": ""})


def test_a_designer_list_may_hold_a_mix_of_shapes():
    is_credit = _load_helpers()
    assert is_credit("ALT Design", {"designer": None, "designers": ["Someone", "ALT Design"]})
    assert not is_credit("ALT Design", {"designers": "not a list"})
