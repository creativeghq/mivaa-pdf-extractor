"""
Google Products Integration Service

Integrates with Google Shopping API for price comparison data.
Provides additional price data sources beyond direct competitor scraping.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


class GoogleProductsIntegration:
    """Service for fetching product prices from Google Shopping API"""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_SHOPPING_API_KEY")
        self.cx = os.getenv("GOOGLE_SHOPPING_CX")  # Custom Search Engine ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    async def search_product_prices(
        self,
        product_name: str,
        product_sku: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for product prices using Google Shopping API
        
        Args:
            product_name: Name of the product to search
            product_sku: Optional SKU for more precise matching
            max_results: Maximum number of results to return
            
        Returns:
            List of price data from Google Shopping
        """
        if not self.api_key or not self.cx:
            logger.warning("Google Shopping API credentials not configured")
            return []

        try:
            # Build search query
            query = product_name
            if product_sku:
                query = f"{product_name} {product_sku}"

            # Make API request
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": min(max_results, 10),  # Google API max is 10
                "searchType": "image",  # Use image search for shopping results
            }

            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
            
            # Parse results
            results = []
            for item in data.get("items", []):
                price_info = self._extract_price_info(item)
                if price_info:
                    results.append(price_info)

            logger.info(f"Found {len(results)} price results for '{product_name}'")
            return results

        except httpx.HTTPError as e:
            logger.error(f"Google Shopping API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching Google Shopping: {e}")
            return []

    def _extract_price_info(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract price information from Google Shopping result
        
        Args:
            item: Google Shopping API result item
            
        Returns:
            Formatted price data or None if no price found
        """
        try:
            # Extract price from pagemap metadata
            pagemap = item.get("pagemap", {})
            offers = pagemap.get("offer", [])
            
            if not offers:
                return None

            offer = offers[0]
            price_str = offer.get("price", "")
            
            # Parse price (remove currency symbols and convert to float)
            price = self._parse_price(price_str)
            if price is None:
                return None

            # Extract other details
            source_url = item.get("link", "")
            source_name = item.get("displayLink", "Unknown")
            availability = offer.get("availability", "unknown")
            
            # Map availability to our schema
            availability_map = {
                "in stock": "in_stock",
                "instock": "in_stock",
                "out of stock": "out_of_stock",
                "outofstock": "out_of_stock",
                "limited availability": "limited_stock",
            }
            availability_status = availability_map.get(
                availability.lower(), "unknown"
            )

            return {
                "source_name": source_name,
                "source_url": source_url,
                "price": price,
                "currency": offer.get("priceCurrency", "USD"),
                "availability": availability_status,
                "scraped_at": datetime.utcnow().isoformat(),
                "metadata": {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "image_url": item.get("pagemap", {}).get("cse_image", [{}])[0].get("src"),
                }
            }

        except Exception as e:
            logger.error(f"Error extracting price info: {e}")
            return None

    def _parse_price(self, price_str: str) -> Optional[float]:
        """
        Parse price string to float
        
        Args:
            price_str: Price string (e.g., "$99.99", "€50.00")
            
        Returns:
            Price as float or None if parsing fails
        """
        try:
            # Remove currency symbols and whitespace
            cleaned = price_str.replace("$", "").replace("€", "").replace("£", "")
            cleaned = cleaned.replace(",", "").strip()
            
            return float(cleaned)
        except (ValueError, AttributeError):
            return None


# Singleton instance
_google_products_service: Optional[GoogleProductsIntegration] = None


def get_google_products_service() -> GoogleProductsIntegration:
    """Get or create the Google Products Integration service singleton."""
    global _google_products_service
    if _google_products_service is None:
        _google_products_service = GoogleProductsIntegration()
    return _google_products_service


