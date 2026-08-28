"""
Guard: the vision call is grounded in OCR, and says so when it is not.

WHY THIS EXISTS
---------------
Until issue #393 the material-analysis call was handed exactly two things — the image
and the prompt. Nothing else. So `detected_text` (SKUs, model numbers, socket codes
E27/GU10/G9/G13, IP ratings IP20/IP44/IP65, wattages, dimensions, certifications) was
the vision model reading glyphs off pixels, while PaddleOCR's decoded transcription of
that exact crop sat unread in `document_images.ocr_text`.

That is also a Medallion violation by this platform's own rule: silver already holds
the text, and the vision path was re-deriving it from bronze pixels.

Two things have to hold, and only one of them is about wiring:

1. The OCR text reaches the model, fenced as DATA (security invariant 9). Catalogue
   text is untrusted ingested content — "IGNORE PREVIOUS INSTRUCTIONS" sets in
   Helvetica like anything else.

2. The ABSENCE of OCR text is reported distinctly from OCR finding nothing. An empty
   string renders "OCR crashed", "OCR never ran" and "this crop genuinely has no text"
   identically, and the wrong reading is actively harmful: tell a model there is no
   text on a page covered in SKUs and it will agree with you.

The behavioural half loads `app/models/ocr_context.py` BY PATH — it is stdlib-only
precisely so this is possible in a CI that installs pytest and nothing else.
"""

import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_OCR_CONTEXT = _ROOT / "app" / "models" / "ocr_context.py"
_IMAGE_SERVICE = _ROOT / "app" / "services" / "images" / "image_processing_service.py"
_STAGE_3 = _ROOT / "app" / "api" / "pdf_processing" / "stage_3_images.py"
_OCR_SERVICE = _ROOT / "app" / "services" / "pdf" / "ocr_service.py"


def _load_ocr_context():
    spec = importlib.util.spec_from_file_location("_ocr_context_probe", _OCR_CONTEXT)
    assert spec and spec.loader, f"cannot load {_OCR_CONTEXT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── behaviour ────────────────────────────────────────────────────────────────

def test_ocr_text_is_fenced_as_data() -> None:
    m = _load_ocr_context()
    block = m.build_ocr_context_block("VALENOVA GU10 IP44", ocr_failed=False)
    assert block.startswith(m.OCR_BLOCK_OPEN)
    assert block.rstrip().endswith(m.OCR_BLOCK_CLOSE)
    assert "VALENOVA GU10 IP44" in block


def test_a_crash_is_not_reported_as_an_empty_page() -> None:
    """The failure that matters. Both produce no text; only one licenses none."""
    m = _load_ocr_context()
    failed = m.build_ocr_context_block(None, ocr_failed=True)
    clean = m.build_ocr_context_block("", ocr_failed=False)

    assert failed != clean
    assert "FAILED" in failed
    # A crash must never tell the model the image is textless.
    assert "found NO text" not in failed
    assert "do not assume" in failed
    # A clean empty read is the ONLY case allowed to license an empty answer.
    assert "found NO text" in clean


def test_a_skip_is_not_reported_as_an_empty_page() -> None:
    m = _load_ocr_context()
    skipped = m.build_ocr_context_block(
        None, ocr_failed=False, skipped_reason="photo_not_text_bearing"
    )
    assert "photo_not_text_bearing" in skipped
    assert "found NO text" not in skipped
    assert "do not assume" in skipped


def test_whitespace_only_ocr_counts_as_no_text() -> None:
    m = _load_ocr_context()
    assert "found NO text" in m.build_ocr_context_block("   \n\t ", ocr_failed=False)


def test_the_module_stays_stdlib_only() -> None:
    """It is loadable by path only while it imports nothing that CI lacks."""
    src = _OCR_CONTEXT.read_text(encoding="utf-8")
    for imp in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", src, re.MULTILINE):
        root = imp.split(".")[0]
        assert root in {"typing"}, (
            f"{_OCR_CONTEXT.name} imports {imp!r}. It must stay stdlib-only — MIVAA CI "
            f"installs pytest and nothing else, and a dependency here downgrades this "
            f"guard from behavioural to grep."
        )


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_vision_call_actually_sends_the_ocr_block() -> None:
    """A builder nothing calls is decoration."""
    src = _IMAGE_SERVICE.read_text(encoding="utf-8")
    assert "build_ocr_context_block" in src
    assert "_ocr_context_for_vision" in src
    # The block must be in the content list handed to Claude, not merely computed.
    assert re.search(
        r'\{"type":\s*"text",\s*"text":\s*ocr_block\}', src
    ), (
        "The OCR block is built but no longer appears in the `content` list of the "
        "vision request. Grounding that never reaches the model is worse than none — "
        "it looks done."
    )


def test_ocr_is_memoised_so_grounding_does_not_double_the_ocr_bill() -> None:
    """Stage 3 has fought this exact regression once already.

    An extraction-phase OCR pass duplicating the Phase 3 one cost ~7-15 min/product
    and pushed job 72031fb0 past its 1200s budget; `enable_multimodal=False` is the
    scar. Vision grounding adds a SECOND consumer of the same OCR, so the memo is
    what keeps it from being a second pass.
    """
    ocr_src = _OCR_SERVICE.read_text(encoding="utf-8")
    assert "_result_cache" in ocr_src and "_cache_key" in ocr_src, (
        "The OCR per-file memo is gone. Vision grounding and Phase 3 both OCR the "
        "same crop, so without it every image is OCR'd twice."
    )
    assert re.search(r"cached\s*=\s*self\._result_cache\.get\(", ocr_src), (
        "`extract_text_from_image` no longer consults the memo before running "
        "PaddleOCR."
    )


def test_phase_3_ocrs_the_same_file_the_vision_call_did() -> None:
    """The memo only helps if both consumers ask for the same path."""
    src = _STAGE_3.read_text(encoding="utf-8")
    assert re.search(r"src\.get\(\s*'vision_input_path'\s*\)", src), (
        "Phase 3 OCR no longer prefers `vision_input_path`. It then OCRs a different "
        "file than the vision call did, which misses the memo (two PaddleOCR passes "
        "per image) AND stores the tight-crop transcription, which omits the SKU "
        "printed just outside the tile."
    )
