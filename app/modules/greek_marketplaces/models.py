"""Pydantic models shared by the Greek Marketplaces adapters."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MarketplaceProduct(BaseModel):
    """Shape Firecrawl extracts from a Bestdeals / Shopflix search result page."""

    found: bool = Field(default=False, description="True if a product match was found on the page.")
    product_url: Optional[str] = Field(default=None, description="Direct URL of the first matching product.")
    retailer_name: Optional[str] = Field(default=None, description="Merchant label shown on the listing.")
    price: Optional[str] = Field(default=None, description="Current price as displayed (e.g. '€79,90').")
    original_price: Optional[str] = Field(default=None, description="On-page 'was' price if the product is on sale.")
    currency: Optional[str] = Field(default="EUR", description="ISO 4217 currency code.")
    availability: Optional[str] = Field(
        default=None,
        description="'in_stock' | 'out_of_stock' | 'limited' | 'unknown'.",
    )
