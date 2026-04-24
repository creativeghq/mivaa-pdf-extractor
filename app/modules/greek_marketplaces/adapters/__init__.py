"""Source-specific adapters for the Greek Marketplaces module."""

from app.modules.greek_marketplaces.adapters.bestdeals import BestdealsAdapter
from app.modules.greek_marketplaces.adapters.shopflix import ShopflixAdapter
from app.modules.greek_marketplaces.adapters.skroutz import SkroutzAdapter

__all__ = ["BestdealsAdapter", "ShopflixAdapter", "SkroutzAdapter"]
