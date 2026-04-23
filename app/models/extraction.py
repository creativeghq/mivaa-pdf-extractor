"""
Pydantic extraction models for Firecrawl-powered scraping.

The `Field(description=...)` strings are passed to Firecrawl's extraction
engine via `model_json_schema()` and guide the LLM that fills in each field.
Keep descriptions concrete and short.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PriceExtraction(BaseModel):
    """
    Fields Firecrawl should pull from a competitor product page.

    Stored raw (strings) because sites return prices with locale-specific
    formatting and currency symbols. Numeric parsing happens downstream via
    `price_parser`.
    """

    price: Optional[str] = Field(
        default=None,
        description="Current product price with currency symbol or code as shown on the page, e.g. '$49.99', '€1.299,00', 'From £29'. Use the main product price, not related or strike-through prices.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="ISO 4217 currency code (USD, EUR, GBP, JPY, etc.) if clearly determinable from the page.",
    )
    availability: Optional[str] = Field(
        default=None,
        description="Stock status: 'in_stock', 'out_of_stock', 'limited', 'backorder', or 'unknown'.",
    )
    shipping_cost: Optional[str] = Field(
        default=None,
        description="Shipping cost for a standard single-unit order if visible on the page.",
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Canonical product name or title as displayed on the page.",
    )
