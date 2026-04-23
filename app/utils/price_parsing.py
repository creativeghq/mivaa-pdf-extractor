"""
Locale-aware price string parser.

Wraps the `price-parser` library and normalizes currency symbols to ISO 4217
codes. Handles:
    "$49.99"        -> (Decimal('49.99'), 'USD')
    "€1.299,00"     -> (Decimal('1299.00'), 'EUR')     # European format
    "£1,299.00"     -> (Decimal('1299.00'), 'GBP')     # US thousands
    "From $49"      -> (Decimal('49'), 'USD')
    "49.99 EUR"     -> (Decimal('49.99'), 'EUR')
"""

from decimal import Decimal
from typing import Optional, Tuple

from price_parser import Price

# Currency symbol → ISO 4217 code. Extend as needed.
_SYMBOL_TO_CODE = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",  # Ambiguous with CNY; caller can override via explicit code
    "₹": "INR",
    "₽": "RUB",
    "₩": "KRW",
    "฿": "THB",
    "₺": "TRY",
    "R$": "BRL",
    "A$": "AUD",
    "C$": "CAD",
    "HK$": "HKD",
    "S$": "SGD",
    "NZ$": "NZD",
    "CHF": "CHF",
}


def parse_price(raw: Optional[str], hint_currency: Optional[str] = None) -> Tuple[Optional[Decimal], Optional[str]]:
    """
    Extract (amount, ISO currency code) from a raw price string.

    Args:
        raw: Price text as scraped (e.g. "$49.99", "€1.299,00 incl. VAT").
        hint_currency: If the page exposes a separate currency field, pass the
            ISO code here so we fall back to it when the string itself has no
            symbol.

    Returns:
        (amount, currency_code). Either element may be None if parsing fails.
    """
    if not raw or not isinstance(raw, str):
        return None, _normalize_currency(hint_currency)

    p = Price.fromstring(raw)
    amount = p.amount  # Decimal | None
    currency = _normalize_currency(p.currency) or _normalize_currency(hint_currency)
    return amount, currency


def _normalize_currency(symbol_or_code: Optional[str]) -> Optional[str]:
    """Map a currency symbol or code to an ISO 4217 code (uppercase)."""
    if not symbol_or_code:
        return None
    s = symbol_or_code.strip()
    if not s:
        return None
    if len(s) == 3 and s.isalpha():
        return s.upper()
    return _SYMBOL_TO_CODE.get(s) or _SYMBOL_TO_CODE.get(s.upper())
