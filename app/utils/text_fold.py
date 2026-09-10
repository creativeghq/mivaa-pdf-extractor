"""One Greek-aware fold, used everywhere two names are compared for identity."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional

#: Greek letters that are visually identical to a Latin one. Only lookalikes belong
#: here — `σ`, `π`, `λ` and friends have no Latin twin and must never be mapped.
GREEK_TO_LATIN: Dict[str, str] = {
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "a", "β": "b", "ε": "e", "ζ": "z", "η": "h", "ι": "i", "κ": "k",
    "μ": "m", "ν": "n", "ο": "o", "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
}

#: Separators that routinely drift between versions of the same model number:
#: "7012-MT" / "7012 MT" / "7012_MT" / "7012.MT" → "7012MT"
_MODEL_SEP_RE = re.compile(r"[\s\-_./]+")


def strip_accents(text: str) -> str:
    """Accent-insensitive compare: 'Νιπτήρα' ≡ 'Νιπτηρα', 'é' ≡ 'e'."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def fold_for_search(text: Optional[str]) -> str:
    """Case + accent + final-sigma fold. The Python twin of `foldForSearch`.

    The final sigma is the one people forget. `ς` (U+03C2) and `σ` (U+03C3) are
    separate codepoints, so "ΚΩΣΤΑΣ" lowercases to "κωστασ" while the same name typed
    naturally is "κωστάς" → "κωστας". Without this line they never match, and the UI
    just shows an empty result for a name that is plainly in the list.
    """
    if not text:
        return ""
    folded = strip_accents(str(text)).lower().replace("ς", "σ")
    return " ".join(folded.split())


def fold_identity(text: Optional[str]) -> str:
    """`fold_for_search` plus the Greek→Latin lookalike map. For dedupe, not search."""
    if not text:
        return ""
    mapped = "".join(GREEK_TO_LATIN.get(ch, ch) for ch in str(text))
    return fold_for_search(mapped)


def fold_model_token(token: Optional[str]) -> str:
    """Strict SKU/model equality across alphabet lookalikes and separator drift.

      "7012ΜΤ"      → "7012MT"
      "7012 MT"     → "7012MT"
      "preciosa-01" → "PRECIOSA01"
    """
    if not token:
        return ""
    return _MODEL_SEP_RE.sub("", fold_identity(token).upper())
