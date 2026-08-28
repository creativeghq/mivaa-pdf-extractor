"""
How OCR text is presented to the vision model (issue #393 Step 1).

STDLIB ONLY — no pydantic, no app imports, no third-party packages. MIVAA's CI
installs pytest and nothing else, so a rule that lives in a module with real
dependencies can only ever be checked by grepping its source. Keeping this
derivation importable by path is what lets the guard test assert on BEHAVIOUR
instead of on the presence of a string.

Why this exists at all: until #393 the vision call was handed the image and the
prompt, and nothing else. `detected_text` — SKUs, model numbers, socket codes
(E27/GU10/G9/G13), IP ratings (IP20/IP44/IP65), wattages, dimensions,
certifications — was therefore the model reading glyphs off pixels, while
PaddleOCR's decoded transcription of the very same crop sat unused in
`document_images.ocr_text`. A VLM chooses between `GU10`, `GU1O` and `GUIO` on
priors; an OCR engine decodes it.
"""

from typing import Optional

#: Delimiters for the OCR text handed to the vision model (security invariant 9).
#:
#: OCR text is untrusted ingested content — it is whatever a supplier chose to print
#: on a page, and "IGNORE PREVIOUS INSTRUCTIONS" renders in Helvetica like anything
#: else. Fencing it as DATA is what stops a catalogue issuing instructions to the
#: model that is reading it.
OCR_BLOCK_OPEN = "<ocr_text_from_this_image>"
OCR_BLOCK_CLOSE = "</ocr_text_from_this_image>"

#: Said to the model when OCR produced nothing to show it. Deliberately does NOT
#: assert the image is textless — see `build_ocr_context_block`.
_UNAVAILABLE_TAIL = (
    "No transcription is available — read the image directly and do not assume "
    "the image contains no text."
)


def build_ocr_context_block(
    ocr_text: Optional[str],
    *,
    ocr_failed: bool,
    skipped_reason: Optional[str] = None,
) -> str:
    """Render the OCR sidecar the vision model sees, including its ABSENCE.

    The three no-text outcomes are stated distinctly on purpose (pipeline
    convention 1: explicit failure markers, not empty returns).

    "OCR failed", "OCR was never run" and "OCR ran and this crop genuinely has no
    text on it" are different facts that an empty string renders identically. The
    difference is not cosmetic: told *nothing*, a model fills the silence, and told
    "there is no text here" when OCR merely crashed, it will confidently return an
    empty `detected_text` for a page covered in SKUs. Only the third case licenses
    an empty answer, so only the third case says so.
    """
    if ocr_failed:
        body = f"OCR FAILED for this image. {_UNAVAILABLE_TAIL}"
    elif skipped_reason:
        body = f"OCR was not run for this image (reason: {skipped_reason}). {_UNAVAILABLE_TAIL}"
    elif ocr_text and ocr_text.strip():
        body = ocr_text.strip()
    else:
        body = (
            "OCR ran successfully and found NO text on this image. Treat "
            "`detected_text` as empty unless you can clearly read text the OCR "
            "pass missed."
        )
    return f"{OCR_BLOCK_OPEN}\n{body}\n{OCR_BLOCK_CLOSE}"
