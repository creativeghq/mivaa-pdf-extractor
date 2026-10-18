"""#482 F7: a mention refresh with a failed source must not report clean."""

import ast
from pathlib import Path
from typing import Dict, Optional

MENTIONS = Path(__file__).resolve().parents[2] / "app" / "services" / "integrations" / "tracked_mentions_service.py"
SRC = MENTIONS.read_text(encoding="utf-8")


def _load(name: str):
    node = next(n for n in ast.walk(ast.parse(SRC)) if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"Optional": Optional, "Dict": Dict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MENTIONS), "exec"), scope)
    return scope[name]


refresh_error_summary = _load("refresh_error_summary")


def test_no_failed_source_is_clean():
    assert refresh_error_summary({}) is None
    assert refresh_error_summary(None) is None
    assert refresh_error_summary({"gdelt": ""}) is None


def test_every_failed_source_is_named():
    summary = refresh_error_summary({
        "perplexity_sonar": "401 Unauthorized",
        "dataforseo_news": "402 Payment Required",
        "gdelt": "",
    })
    assert summary == "dataforseo_news: 402 Payment Required; perplexity_sonar: 401 Unauthorized"


def test_summary_is_bounded():
    assert len(refresh_error_summary({f"s{i}": "x" * 500 for i in range(20)})) <= 1000


def test_the_refresh_stamp_writes_it():
    stamp = ast.get_source_segment(SRC, next(
        n for n in ast.walk(ast.parse(SRC)) if isinstance(n, ast.FunctionDef) and n.name == "_stamp_refresh"))
    assert '"last_error": refresh_error_summary(errors)' in stamp
