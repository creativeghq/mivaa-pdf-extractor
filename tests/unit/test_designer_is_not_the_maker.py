"""A credited designer is never the maker, so no studio becomes a brand company."""

import ast
import importlib.util
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
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {f.name for f in funcs} == wanted
    scope: Dict[str, Any] = {"fold_identity": fold.fold_identity, "Dict": Dict, "Any": Any, "Optional": Optional}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(_STAGE_4), "exec"), scope)
    return scope["_maker_is_a_designer_credit"]


def test_the_studio_credited_as_designer_is_not_a_maker():
    is_credit = _load_helpers()
    assert is_credit("MUT", {"designer": "MUT"})
    assert is_credit("Estudi{H}ac", {"designers": ["Estudi{H}ac"]})
    assert is_credit("  dsignio ", {"designer": "Dsignio"})


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
