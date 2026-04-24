"""
Pydantic extraction models for Firecrawl-powered scraping.

The `Field(description=...)` strings are passed to Firecrawl's extraction
engine via `model_json_schema()` and guide the LLM that fills in each field.
Keep descriptions concrete and short.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PriceExtraction(BaseModel):
    """
    Fields Firecrawl should pull from a competitor product page.

    Stored raw (strings) because sites return prices with locale-specific
    formatting and currency symbols. Numeric parsing happens downstream via
    `price_parser`.

    product_name + product_breadcrumb + visible_attributes are the trio the
    identity classifier uses to decide whether the page actually matches the
    query. Without them a page showing any price on the same domain would
    silently slip in as a "verified" hit for the wrong SKU.
    """

    price: Optional[str] = Field(
        default=None,
        description="Current product price with currency symbol or code as shown on the page, e.g. '$49.99', '€1.299,00', 'From £29'. Use the main product price — if the page shows a was/now promo, this is the NOW price.",
    )
    original_price: Optional[str] = Field(
        default=None,
        description="On-page 'was' / 'original' / 'list' price when the retailer displays a promo or markdown, e.g. '€89.00' when the page shows 'Was €89, Now €79'. Only set if clearly visible on the page; null otherwise.",
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
        description="Canonical product name or title as displayed on the page — pull from the H1 / og:title / primary product heading, NOT the browser tab title.",
    )
    product_breadcrumb: Optional[str] = Field(
        default=None,
        description="Site breadcrumb trail leading to this product, e.g. 'Home > Bath > Faucets > Basin Faucets'. Useful to disambiguate product type when the product name is terse.",
    )
    visible_attributes: Optional[Dict[str, str]] = Field(
        default=None,
        description="Small dict of visible product attributes: color, finish, size, material, etc. — any attribute explicitly shown on the page. Example: {'color':'black','finish':'matt','size':'60cm'}. Keep keys lowercase and concise.",
    )
