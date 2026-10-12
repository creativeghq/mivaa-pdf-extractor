"""The grammar-check route: LanguageTool's reply flattened, failures named, and the route gated."""

import importlib.util
import os
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "languagetool_under_test", _ROOT / "app" / "services" / "integrations" / "languagetool.py"
)
lt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lt)

_TEXT = "Ο αρχιτέκτονας 🏛 σχεδιάζει και και το σχέδιο."


def _payload(offset, length):
    return {
        "software": {"version": "6.6"},
        "matches": [{
            "message": "Πιθανό λάθος: επαναλάβατε μία λέξη",
            "offset": offset,
            "length": length,
            "replacements": [{"value": "και"}, {"value": "a"}, {"value": "b"}, {"value": "c"}],
            "sentence": _TEXT,
            "rule": {"id": "WORD_REPEAT_RULE", "issueType": "duplication", "category": {"id": "MISC"}},
        }],
    }


def test_matches_are_flattened_with_utf16_offsets():
    # The emoji is two UTF-16 units, so the JS offset is one past the Python index.
    js_offset = len(_TEXT[: _TEXT.index("και και")].encode("utf-16-le")) // 2
    seen = {}

    def fake_post(url, form, timeout):
        seen.update(url=url, form=form)
        return _payload(js_offset, 7)

    out = lt.check("el", _TEXT, post=fake_post)
    assert seen["url"].endswith("/v2/check")
    assert seen["form"] == {"language": "el-GR", "text": _TEXT}
    assert out["language"] == "el-GR" and out["software"] == "6.6"
    (m,) = out["matches"]
    assert m["text"] == "και και"
    assert m["offset"] == js_offset and m["length"] == 7
    assert m["rule_id"] == "WORD_REPEAT_RULE" and m["category"] == "MISC"
    assert m["replacements"] == ["και", "a", "b"]
    assert m["sentence"] == _TEXT


def test_server_url_is_local_unless_configured(monkeypatch):
    monkeypatch.delenv("LANGUAGETOOL_URL", raising=False)
    assert lt.server_url() == "http://127.0.0.1:8081"
    monkeypatch.setenv("LANGUAGETOOL_URL", "http://127.0.0.1:9000/")
    assert lt.server_url() == "http://127.0.0.1:9000"


@pytest.mark.parametrize("code,expected", [("el", "el-GR"), ("el-GR", "el-GR"), ("en", "en-US"), ("fr", "fr"), ("pt_BR", "pt-BR")])
def test_language_codes_resolve(code, expected):
    assert lt.resolve_language(code) == expected


@pytest.mark.parametrize("code", ["", "greek", "el GR", "../x"])
def test_a_bad_language_code_is_refused(code):
    with pytest.raises(ValueError):
        lt.check(code, "κείμενο", post=lambda *a: pytest.fail("must not call the server"))


def test_text_over_the_limit_is_refused_before_the_call():
    with pytest.raises(ValueError):
        lt.check("el", "α" * (lt.MAX_TEXT_CHARS + 1), post=lambda *a: pytest.fail("must not call the server"))


def test_an_unreachable_server_is_unavailable_not_clean():
    def down(url, form, timeout):
        raise lt.GrammarCheckUnavailable("connection refused")

    with pytest.raises(lt.GrammarCheckUnavailable):
        lt.check("el", "κείμενο", post=down)


def test_a_reply_without_matches_is_unavailable_not_clean():
    with pytest.raises(lt.GrammarCheckUnavailable):
        lt.check("el", "κείμενο", post=lambda *a: {"software": {}})


def test_an_empty_matches_list_is_a_clean_check():
    assert lt.check("el", "κείμενο", post=lambda *a: {"matches": []})["matches"] == []


def test_the_route_is_internal_gated_and_mounted():
    src = (_ROOT / "app" / "api" / "text_routes.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert 'prefix="/api/internal/text"' in code
    assert re.search(r"@router\.post\(\"/grammar-check\"\)[\s\S]*?Depends\(verify_internal_access\)", code)
    main = (_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(text_internal_router)" in main
    assert os.path.basename(lt.__file__) == "languagetool.py"
