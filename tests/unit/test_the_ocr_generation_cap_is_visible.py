import ast
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MODAL = _ROOT / "modal_app/paddleocr_vl.py"


def _modal_namespace():
    source = _MODAL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "truncation_suspected":
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", "") in {"PARSE_MAX_NEW_TOKENS", "CAP_SUSPECT_CHARS_PER_TOKEN", "CAP_SUSPECT_SHARE"} for t in node.targets):
            keep.append(node)
    namespace = {"os": __import__("os")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_MODAL), "exec"), namespace)
    return namespace


def test_the_cap_comes_from_the_environment_and_is_larger_than_the_old_one():
    ns = _modal_namespace()
    assert ns["PARSE_MAX_NEW_TOKENS"] >= 4096
    assert 'PADDLEOCR_MAX_NEW_TOKENS' in _MODAL.read_text(encoding="utf-8")


def test_a_region_near_the_cap_is_marked_and_a_normal_one_is_not():
    ns = _modal_namespace()
    flag, cap = ns["truncation_suspected"], 2048
    assert flag("x" * int(cap * 2.7), cap) is True
    assert flag("x" * 4000, cap) is False
    assert flag("", cap) is False and flag(None, cap) is False


def test_every_parsed_region_and_the_response_carry_the_flag_and_the_cap():
    source = _MODAL.read_text(encoding="utf-8")
    assert '"truncation_suspected": truncation_suspected(content)' in source
    assert '"max_new_tokens": PARSE_MAX_NEW_TOKENS' in source


def test_the_flag_reaches_the_layout_element_and_is_a_hard_reason():
    spec = importlib.util.spec_from_file_location("paddleocr_pipeline_flag_under_test", _ROOT / "app/services/pdf/paddleocr_pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules["paddleocr_pipeline_flag_under_test"] = pipeline
    spec.loader.exec_module(pipeline)
    flagged = pipeline.PaddleRegion("table", (0, 0, 1, 1), "<table></table>", 0, {"truncation_suspected": True})
    plain = pipeline.PaddleRegion("table", (0, 0, 1, 1), "<table></table>", 0)
    assert pipeline.region_to_layout_element(flagged, 100, 100, 1)["metadata"]["truncation_suspected"] is True
    assert pipeline.region_to_layout_element(plain, 100, 100, 1)["metadata"]["truncation_suspected"] is False

    spec = importlib.util.spec_from_file_location("oc_flag_under_test", _ROOT / "app/services/pdf/ocr_completeness.py")
    oc = importlib.util.module_from_spec(spec)
    sys.modules["oc_flag_under_test"] = oc
    spec.loader.exec_module(oc)
    table = "<table>" + "".join(f"<tr><td>a{i}</td><td>{i}</td></tr>" for i in range(6)) + "</table>"
    capped = oc.assess_table(table, generation_capped=True)
    assert capped.status == oc.TRUNCATED and capped.reasons == ["generation_cap_hit"]
    assert "generation_cap_hit" in oc.HARD_REASONS
    assert oc.assess_table(table).status == oc.COMPLETE


def test_the_client_waits_long_enough_for_the_larger_cap():
    import re
    match = re.search(r"paddleocr_inference_timeout: int = Field\(\s*default=(\d+)", (_ROOT / "app/config.py").read_text(encoding="utf-8"))
    assert match and 300 <= int(match.group(1)) <= 540
