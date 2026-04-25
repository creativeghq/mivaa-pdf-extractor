"""Source-specific adapters for the Greek Marketplaces module."""

from app.modules.greek_marketplaces.adapters.bestprice import BestpriceAdapter
from app.modules.greek_marketplaces.adapters.shopflix import ShopflixAdapter
from app.modules.greek_marketplaces.adapters.skroutz import SkroutzAdapter

__all__ = ["BestpriceAdapter", "ShopflixAdapter", "SkroutzAdapter"]
