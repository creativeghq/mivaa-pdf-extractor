"""
Job Salary Normalizer — convert salaries from any source to annualized USD.

DataForSEO returns `{min_value, max_value, currency, type}` where `type` is one of
`year` / `month` / `week` / `day` / `hour`. Perplexity returns whatever the page
displayed (often integer year, sometimes monthly EUR for European listings).
Firecrawl extraction is freeform.

The normalized fields (`salary_annual_min_usd`, `salary_annual_max_usd`) on
job_listings let the UI compare apples-to-apples across sources without losing
the source-reported values.

Conversion rates: a static lookup table is fine for sorting / display. We
intentionally do NOT hit a live FX API on every refresh — staleness of a few
percent doesn't matter for "is this listing in the salary band I want?"
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Static rates → 1 unit = N USD. Refresh every quarter or so by hand.
# Sourced 2026-05 from a midpoint of recent ECB / Fed rates.
_FX_TO_USD: dict[str, float] = {
    "USD": 1.00,
    "EUR": 1.08,
    "GBP": 1.27,
    "CHF": 1.13,
    "CAD": 0.74,
    "AUD": 0.66,
    "NZD": 0.61,
    "JPY": 0.0067,
    "CNY": 0.14,
    "INR": 0.012,
    "SEK": 0.094,
    "NOK": 0.094,
    "DKK": 0.144,
    "PLN": 0.25,
    "CZK": 0.043,
    "BRL": 0.20,
    "MXN": 0.058,
    "ZAR": 0.054,
    "TRY": 0.029,
    "ILS": 0.27,
    "AED": 0.272,
    "SGD": 0.74,
    "HKD": 0.128,
    # Crypto / stable / oddballs intentionally omitted.
}

# Hours-per-year baseline for hourly→annual: 40h/wk × 52wk = 2080h.
# Conservative — full-time but not overtime. A US salaried-equivalent baseline.
_HOURS_PER_YEAR = 2080
_DAYS_PER_YEAR = 250
_WEEKS_PER_YEAR = 52
_MONTHS_PER_YEAR = 12

_PERIOD_NORMALIZE = {
    "year": 1,
    "yr": 1,
    "annual": 1,
    "annually": 1,
    "month": _MONTHS_PER_YEAR,
    "mo": _MONTHS_PER_YEAR,
    "monthly": _MONTHS_PER_YEAR,
    "week": _WEEKS_PER_YEAR,
    "wk": _WEEKS_PER_YEAR,
    "weekly": _WEEKS_PER_YEAR,
    "day": _DAYS_PER_YEAR,
    "daily": _DAYS_PER_YEAR,
    "hour": _HOURS_PER_YEAR,
    "hr": _HOURS_PER_YEAR,
    "hourly": _HOURS_PER_YEAR,
}


def _fx_rate(currency: Optional[str]) -> Optional[float]:
    if not currency:
        return None
    code = currency.strip().upper()
    return _FX_TO_USD.get(code)


def _period_multiplier(period: Optional[str]) -> Optional[int]:
    """Returns how many of `period` are in a year. e.g. 'month' → 12."""
    if not period:
        return None
    p = period.strip().lower()
    return _PERIOD_NORMALIZE.get(p)


def _infer_period_from_magnitude(amount: int, currency_code: str) -> str:
    """Last-resort heuristic when the source omits the period. A 3-digit USD figure
    is probably hourly; a 4-5 digit figure is probably monthly; a 6-digit is annual.
    Imperfect, but better than rejecting.
    """
    if amount < 500:
        return "hour"
    if amount < 12000 and currency_code in {"USD", "EUR", "GBP", "CAD", "AUD"}:
        return "month"
    return "year"


def normalize_to_annual_usd(
    *,
    salary_min: Optional[int],
    salary_max: Optional[int],
    salary_currency: Optional[str],
    salary_period: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Returns (annual_min_usd, annual_max_usd, note).
    `note` is None on a clean conversion, otherwise a short string explaining
    what the normalizer assumed (e.g. 'inferred period=year from magnitude').
    """
    # No data at all
    if not salary_min and not salary_max:
        return None, None, None

    currency_code = (salary_currency or "USD").strip().upper()
    fx = _fx_rate(currency_code)
    notes: list[str] = []

    if fx is None:
        notes.append(f"unknown currency '{currency_code}'; treating as USD")
        fx = 1.0

    period = (salary_period or "").strip().lower() or None
    multiplier = _period_multiplier(period)
    if multiplier is None:
        # Use the larger of min/max for the heuristic
        sample = max(int(salary_min or 0), int(salary_max or 0))
        if sample > 0:
            inferred = _infer_period_from_magnitude(sample, currency_code)
            multiplier = _period_multiplier(inferred)
            notes.append(f"inferred period={inferred} from magnitude")
        else:
            multiplier = 1

    def _convert(v: Optional[int]) -> Optional[int]:
        if v is None or v <= 0:
            return None
        return int(round(float(v) * multiplier * fx))

    ann_min = _convert(salary_min)
    ann_max = _convert(salary_max)

    # Sanity floor + ceiling. A normalized value <$5K/yr or >$2M/yr is almost
    # certainly a parse mistake (e.g. someone posted "200" meaning daily contract
    # rate but no period attached, or salary was actually a job-id field).
    def _sane(v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 5000 or v > 2_000_000:
            notes.append(f"rejected out-of-band value {v}")
            return None
        return v

    ann_min_sane = _sane(ann_min)
    ann_max_sane = _sane(ann_max)

    note = "; ".join(notes) if notes else None
    return ann_min_sane, ann_max_sane, note


def normalize_listing_in_place(listing: dict) -> dict:
    """Convenience wrapper: takes a dict matching the job_listings insert shape,
    adds salary_annual_min_usd / salary_annual_max_usd / salary_normalization_note
    if a salary is present. Returns the same dict (mutated)."""
    ann_min, ann_max, note = normalize_to_annual_usd(
        salary_min=listing.get("salary_min"),
        salary_max=listing.get("salary_max"),
        salary_currency=listing.get("salary_currency"),
        salary_period=listing.get("salary_period"),
    )
    if ann_min is not None or ann_max is not None or note is not None:
        listing["salary_annual_min_usd"] = ann_min
        listing["salary_annual_max_usd"] = ann_max
        listing["salary_normalization_note"] = note
    return listing
