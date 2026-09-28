"""Who the SERP is fetched FOR: device and location. Stdlib only, so CI can test it."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

SERP_DEVICES = ("desktop", "mobile")

_COUNTRY_ISO = re.compile(r"^[A-Za-z]{2}$")


def normalize_device(device: Optional[str]) -> str:
    value = (device or "desktop").strip().lower()
    if value not in SERP_DEVICES:
        raise ValueError(f"device must be one of {', '.join(SERP_DEVICES)}, got {device!r}")
    return value


def normalize_location_code(location_code: Any) -> Optional[int]:
    if location_code is None or location_code == "":
        return None
    if isinstance(location_code, bool):
        raise ValueError("location_code must be a positive integer")
    try:
        value = int(location_code)
    except (TypeError, ValueError):
        raise ValueError(f"location_code must be a positive integer, got {location_code!r}") from None
    if value <= 0 or str(value) != str(location_code).strip():
        raise ValueError(f"location_code must be a positive integer, got {location_code!r}")
    return value


def organic_task(
    *, keyword: str, country_location: int, language_code: str, depth: int, paa_depth: int,
    device: Optional[str] = None, location_code: Any = None,
) -> Dict[str, Any]:
    """A city `location_code` replaces the country one; it never combines with it."""
    city = normalize_location_code(location_code)
    return {
        "keyword": keyword,
        "location_code": city if city is not None else country_location,
        "language_code": language_code,
        "device": normalize_device(device),
        "depth": depth,
        "people_also_ask_click_depth": paa_depth,
    }


def locations_path(country_code: Optional[str]) -> str:
    code = (country_code or "").strip()
    if not _COUNTRY_ISO.match(code):
        raise ValueError(f"country_code must be a two-letter ISO code, got {country_code!r}")
    return f"/serp/google/locations/{code.lower()}"
