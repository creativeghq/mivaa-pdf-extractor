"""
Competitor Scraper Service

Service for scraping competitor prices using Firecrawl API with:
- Retry logic and exponential backoff
- Credit tracking and cost management
- Price extraction with AI-powered schema
- Error handling and logging
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal

from app.config import get_settings
from app.services.core.supabase_client import get_supabase_client
from app.services.core.ai_call_logger import AICallLogger
from app.services.integrations.credits_integration_service import get_credits_service

logger = logging.getLogger(__name__)


class CompetitorScraperError(Exception):
    """Base exception for competitor scraper errors"""
    pass


class CompetitorScraperService:
    """
    Service for scraping competitor prices using Firecrawl API.
    
    Features:
    - Firecrawl v2 API integration
    - Automatic retry with exponential backoff
    - Credit tracking and debit
    - Price extraction with structured schema
    - Error handling and logging
    """
    
    def __init__(self):
        """Initialize the competitor scraper service."""
        self.settings = get_settings()
        self.supabase = get_supabase_client()
        self.ai_logger = AICallLogger()
        self.credits_service = get_credits_service()
        self.logger = logger
        
        # Firecrawl API configuration
        self.firecrawl_api_key = self.settings.firecrawl_api_key
        self.firecrawl_base_url = "https://api.firecrawl.dev/v2"
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        
        if not self.firecrawl_api_key:
            self.logger.warning("⚠️ Firecrawl API key not configured")
    
    async def scrape_competitor_price(
        self,
        url: str,
        product_name: str,
        user_id: str,
        workspace_id: Optional[str] = None,
        scraping_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Scrape competitor price from a URL using Firecrawl.
        
        Args:
            url: Competitor product URL
            product_name: Name of product for context
            user_id: User ID for credit debit
            workspace_id: Workspace ID (optional)
            scraping_config: Additional Firecrawl configuration
            
        Returns:
            Dict with:
                - success: bool
                - price: Decimal or None
                - currency: str or None
                - availability: str or None
                - raw_data: Dict with full extraction
                - credits_used: int
                - error: str (if failed)
        """
        if not self.firecrawl_api_key:
            return {
                "success": False,
                "error": "Firecrawl API key not configured",
                "credits_used": 0
            }
        
        start_time = datetime.utcnow()
        
        try:
            # Build Firecrawl request with price extraction schema
            request_body = {
                "url": url,
                "formats": ["markdown"],
                "timeout": 30000,
                "extract": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "price": {
                                "type": "string",
                                "description": "Product price (numeric value with currency symbol or code)"
                            },
                            "currency": {
                                "type": "string",
                                "description": "Currency code (USD, EUR, GBP, etc.)"
                            },
                            "availability": {
                                "type": "string",
                                "description": "Stock availability status (in stock, out of stock, limited, etc.)"
                            },
                            "shipping_cost": {
                                "type": "string",
                                "description": "Shipping cost if available"
                            },
                            "product_name": {
                                "type": "string",
                                "description": "Product name or title from the page"
                            }
                        }
                    },
                    "prompt": f"Extract the current price, currency, and availability status for the product '{product_name}'. Look for the main product price, not related items."
                },
                **(scraping_config or {})
            }
            
            # Make Firecrawl API call with retry logic
            result = await self._call_firecrawl_with_retry(request_body)
            
            # Calculate latency
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Extract credits used from response
            credits_used = result.get("credits_used", 1)  # Default to 1 if not provided
            
            # Log Firecrawl usage and debit credits
            await self.ai_logger.log_firecrawl_call(
                user_id=user_id,
                workspace_id=workspace_id,
                operation_type="scrape",
                credits_used=credits_used,
                latency_ms=latency_ms,
                url=url,
                pages_scraped=1,
                success=result.get("success", False),
                request_data={"product_name": product_name},
                response_data=result.get("data", {})
            )

            # Parse extracted data
            extracted_data = result.get("data", {}).get("extract", {})

            # Parse price to Decimal
            price = None
            price_str = extracted_data.get("price")
            if price_str:
                try:
                    # Remove currency symbols and parse
                    import re
                    price_numeric = re.sub(r'[^\d.]', '', price_str)
                    if price_numeric:
                        price = Decimal(price_numeric)
                except Exception as e:
                    self.logger.warning(f"Failed to parse price '{price_str}': {e}")

            return {
                "success": True,
                "price": price,
                "currency": extracted_data.get("currency"),
                "availability": extracted_data.get("availability"),
                "shipping_cost": extracted_data.get("shipping_cost"),
                "product_name": extracted_data.get("product_name"),
                "raw_data": extracted_data,
                "credits_used": credits_used,
                "latency_ms": latency_ms,
                "scraped_at": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to scrape competitor price from {url}: {e}")

            # Log failed attempt
            latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self.ai_logger.log_firecrawl_call(
                user_id=user_id,
                workspace_id=workspace_id,
                operation_type="scrape",
                credits_used=0,
                latency_ms=latency_ms,
                url=url,
                pages_scraped=0,
                success=False,
                error_message=str(e)
            )

            return {
                "success": False,
                "error": str(e),
                "credits_used": 0
            }

    async def _call_firecrawl_with_retry(
        self,
        request_body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call Firecrawl API with exponential backoff retry logic.

        Args:
            request_body: Firecrawl API request body

        Returns:
            Firecrawl API response

        Raises:
            CompetitorScraperError: If all retries fail
        """
        last_error = None

        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(self.max_retries):
                try:
                    self.logger.info(f"🔄 Firecrawl API attempt {attempt + 1}/{self.max_retries}")

                    response = await client.post(
                        f"{self.firecrawl_base_url}/scrape",
                        headers={
                            "Authorization": f"Bearer {self.firecrawl_api_key}",
                            "Content-Type": "application/json"
                        },
                        json=request_body
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            self.logger.info(f"✅ Firecrawl API success on attempt {attempt + 1}")
                            return result
                        else:
                            error_msg = result.get("error", "Unknown error")
                            self.logger.warning(f"⚠️ Firecrawl returned success=false: {error_msg}")
                            last_error = CompetitorScraperError(f"Firecrawl error: {error_msg}")
                    else:
                        error_text = response.text
                        self.logger.warning(
                            f"⚠️ Firecrawl API error {response.status_code}: {error_text}"
                        )
                        last_error = CompetitorScraperError(
                            f"HTTP {response.status_code}: {error_text}"
                        )

                except httpx.TimeoutException as e:
                    self.logger.warning(f"⏱️ Firecrawl API timeout on attempt {attempt + 1}")
                    last_error = CompetitorScraperError(f"Timeout: {str(e)}")

                except Exception as e:
                    self.logger.warning(f"❌ Firecrawl API error on attempt {attempt + 1}: {e}")
                    last_error = CompetitorScraperError(str(e))

                # Exponential backoff before retry
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    self.logger.info(f"⏳ Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        # All retries failed
        raise last_error or CompetitorScraperError("All retries failed")


# Singleton instance
_competitor_scraper_service: Optional[CompetitorScraperService] = None


def get_competitor_scraper_service() -> CompetitorScraperService:
    """Get singleton instance of CompetitorScraperService."""
    global _competitor_scraper_service
    if _competitor_scraper_service is None:
        _competitor_scraper_service = CompetitorScraperService()
    return _competitor_scraper_service



