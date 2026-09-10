"""How OCR text is presented to the vision model (issue #393 Step 1)."""

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
    """Render the OCR sidecar the vision model sees, including its ABSENCE."""
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
