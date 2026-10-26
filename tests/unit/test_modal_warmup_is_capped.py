"""The PaddleOCR cold-start warm-up must cap its generation; an uncapped one wedged containers."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "modal_app/paddleocr_vl.py"


def test_the_warmup_predict_passes_the_token_cap_and_the_parse_path_does_not_reuse_it():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    load = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load")
    parse = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_parse_image")

    def predict_caps(fn):
        return [
            ast.unparse(k.value)
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "predict"
            for k in c.keywords if k.arg == "max_new_tokens"
        ]

    assert predict_caps(load) == ["WARMUP_MAX_NEW_TOKENS"]
    assert predict_caps(parse) == ["2048"]


def test_the_warmup_budget_is_short_enough_to_recycle_a_hang_and_long_enough_to_dump_it():
    """The first predict either returns in seconds or hangs; the budget leaves room for two stack dumps (25s apart) before recycling."""
    import re
    match = re.search(r'PADDLEOCR_WARMUP_BUDGET", "(\d+)"', _SRC.read_text(encoding="utf-8"))
    assert match and 60 <= int(match.group(1)) <= 120


def test_a_wedged_warmup_dumps_every_thread_before_the_container_is_recycled():
    source = _SRC.read_text(encoding="utf-8")
    assert "faulthandler.dump_traceback_later(25, repeat=True" in source
    assert source.index("dump_traceback_later") < source.index("wt.start()") < source.index("cancel_dump_traceback_later")
