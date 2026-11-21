"""Google GoTo passthrough URLs never reach a caller as a google.com result."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "dataforseo_goto",
    Path(__file__).resolve().parents[2] / "app" / "services" / "integrations" / "dataforseo_goto.py",
)
goto = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(goto)

GOTO = "https://google.com/goto?url=CAESkQEBxCJF18Z88VtUEZ0iMhYjprYYiMrpWgJkKs"


def _envelope(items):
    return {"tasks": [{"status_code": 20000, "result": [{"items": items}]}]}


def test_unresolved_item_takes_its_domain_from_the_breadcrumb():
    item = {"type": "organic", "url": GOTO, "domain": "google.com",
            "breadcrumb": "https://www.panelinios.gr › category › attiki"}
    assert goto.normalize_in_place(_envelope([item])) == 1
    assert item["domain"] == "www.panelinios.gr"
    assert item["url"] is None
    assert item["goto_url"] == GOTO
    assert item["url_unresolved"] is True


def test_no_breadcrumb_means_unknown_domain_never_google():
    item = {"type": "organic", "url": GOTO, "domain": "google.com"}
    goto.normalize_in_place(_envelope([item]))
    assert item["domain"] is None


def test_nested_references_are_normalized():
    ref = {"url": GOTO, "domain": "google.com", "breadcrumb": "https://materialshub.gr › blog"}
    data = _envelope([{"type": "ai_overview", "references": [ref]}])
    assert goto.normalize_in_place(data) == 1
    assert ref["domain"] == "materialshub.gr"


def test_a_resolved_item_is_left_alone():
    item = {"type": "organic", "url": "https://materiahub.gr/x", "domain": "materiahub.gr"}
    assert goto.normalize_in_place(_envelope([item])) == 0
    assert item == {"type": "organic", "url": "https://materiahub.gr/x", "domain": "materiahub.gr"}


def test_a_real_google_page_is_not_a_goto():
    assert not goto.is_goto_url("https://support.google.com/goto-guide")
    assert not goto.is_goto_url("https://www.google.com/maps/place/x")
    assert goto.is_goto_url("https://www.google.gr/goto?url=abc")
