"""One Greek-aware fold, used everywhere two names are compared for identity.

This platform is Greek. "Is this the same brand / company / SKU / person" cannot be
answered by `.lower()`, and every subsystem that tried grew its own partial answer:

  * `mention_identity_service.normalize_text` — accents + case + Greek→Latin
    confusables, but not the final sigma
  * `product_identity_service.normalize_text` — accents + case, and NOT the
    confusables, even though `normalize_model_token` in the same file does map them
  * the frontend's `foldForSearch` — accents + case + final sigma, no confusables

Three normalizers, three different strengths, and the differences are invisible: each
one returns a plausible string and a missed match looks exactly like "no such record".
That is how the SKU dedupe (#17 M4-6), the CRM company dedupe (platform #366 BU-3) and
mention identity (#18 M5-9) each shipped their own gap.

TWO DISTINCT OPERATIONS, deliberately named apart:

`fold_for_search` — the twin of `foldForSearch` in
`src/components/core/filters/types.ts`, and held to the same behaviour by
tests/unit/test_text_fold.py. Case-insensitive, accent-insensitive, final sigma folded
onto the medial sigma. Greek stays Greek. This is what a search box compares through,
and it must match the frontend's fold or the same query gives different answers on the
two sides.

`fold_identity` — the above PLUS the Greek→Latin lookalike map. `Μ` (U+039C) and `M`
(U+004D) are different codepoints that render identically, so "7012ΜΤ" typed by a
supplier and "7012MT" typed by us are the same product and compare unequal without it.
Use this for dedupe and entity resolution — NOT for search, where collapsing alphabets
would make a Greek query match Latin records and vice versa.

`fold_model_token` — `fold_identity`, uppercased, with separator drift removed.

Transliteration is explicitly NOT in scope: `Trendafil` and `Τρενταφιλ` are different
alphabets spelling the same sound, and guessing that they are one entity is a different
(and much riskier) decision than folding two codepoints that are visually identical.
"""

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
