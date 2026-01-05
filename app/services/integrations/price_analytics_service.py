"""
Price Analytics Service

Service for price trend analysis, statistics, and chart data generation.
Provides insights into price movements, competitor comparisons, and trends.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from statistics import mean, median, stdev
from decimal import Decimal

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class PriceAnalyticsService:
    """Service for analyzing price trends and generating statistics"""

    def __init__(self):
        self.supabase = get_supabase_client()

    async def get_price_statistics(
        self,
        product_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate price statistics for a product
        
        Args:
            product_id: Product ID to analyze
            days: Number of days to analyze (default: 30)
            
        Returns:
            Dictionary with price statistics
        """
        try:
            # Fetch price history
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            response = self.supabase.client.table("price_history").select("*").eq(
                "product_id", product_id
            ).gte("scraped_at", start_date).order("scraped_at").execute()

            if not response.data:
                return {
                    "product_id": product_id,
                    "period_days": days,
                    "data_points": 0,
                    "error": "No price data available"
                }

            prices = [float(item["price"]) for item in response.data]
            
            # Calculate statistics
            stats = {
                "product_id": product_id,
                "period_days": days,
                "data_points": len(prices),
                "current_price": prices[-1] if prices else None,
                "lowest_price": min(prices),
                "highest_price": max(prices),
                "average_price": mean(prices),
                "median_price": median(prices),
                "price_range": max(prices) - min(prices),
                "price_volatility": stdev(prices) if len(prices) > 1 else 0,
                "first_price": prices[0] if prices else None,
                "price_change": prices[-1] - prices[0] if len(prices) > 1 else 0,
                "price_change_percent": ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 and prices[0] > 0 else 0,
            }

            # Add trend analysis
            stats["trend"] = self._calculate_trend(prices)
            
            return stats

        except Exception as e:
            logger.error(f"Error calculating price statistics: {e}")
            return {
                "product_id": product_id,
                "period_days": days,
                "error": str(e)
            }

    async def get_price_chart_data(
        self,
        product_id: str,
        days: int = 30,
        group_by: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Get formatted price data for charts
        
        Args:
            product_id: Product ID
            days: Number of days of data
            group_by: Grouping interval ('hour', 'day', 'week')
            
        Returns:
            List of data points for charting
        """
        try:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            response = self.supabase.client.table("price_history").select("*").eq(
                "product_id", product_id
            ).gte("scraped_at", start_date).order("scraped_at").execute()

            if not response.data:
                return []

            # Group data by interval
            grouped_data = self._group_price_data(response.data, group_by)
            
            return grouped_data

        except Exception as e:
            logger.error(f"Error getting chart data: {e}")
            return []

    async def compare_with_competitors(
        self,
        product_id: str,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Compare product price with latest competitor prices
        
        Args:
            product_id: Product ID
            current_price: Current product price
            
        Returns:
            Comparison analysis
        """
        try:
            # Get latest competitor prices
            response = self.supabase.client.table("price_history").select("*").eq(
                "product_id", product_id
            ).order("scraped_at", desc=True).limit(10).execute()

            if not response.data:
                return {
                    "product_id": product_id,
                    "current_price": current_price,
                    "competitors_found": 0,
                    "message": "No competitor data available"
                }

            competitor_prices = [float(item["price"]) for item in response.data]
            
            return {
                "product_id": product_id,
                "current_price": current_price,
                "competitors_found": len(competitor_prices),
                "lowest_competitor": min(competitor_prices),
                "highest_competitor": max(competitor_prices),
                "average_competitor": mean(competitor_prices),
                "price_position": self._calculate_price_position(current_price, competitor_prices),
                "cheaper_than_percent": sum(1 for p in competitor_prices if current_price < p) / len(competitor_prices) * 100,
            }

        except Exception as e:
            logger.error(f"Error comparing prices: {e}")
            return {"error": str(e)}

    def _calculate_trend(self, prices: List[float]) -> str:
        """Calculate price trend direction using simple linear regression"""
        if len(prices) < 2:
            return "stable"

        # Simple linear regression slope
        n = len(prices)
        x = list(range(n))
        slope = (n * sum(i * p for i, p in zip(x, prices)) - sum(x) * sum(prices)) / (n * sum(i**2 for i in x) - sum(x)**2)

        if slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"

    def _group_price_data(self, data: List[Dict], group_by: str) -> List[Dict[str, Any]]:
        """Group price data by time interval"""
        # For now, return raw data formatted for charts
        # TODO: Implement actual grouping by hour/day/week
        return [
            {
                "date": item["scraped_at"],
                "price": float(item["price"]),
                "source": item["source_name"],
                "availability": item["availability"]
            }
            for item in data
        ]

    def _calculate_price_position(self, price: float, competitor_prices: List[float]) -> str:
        """Determine price position relative to competitors"""
        if not competitor_prices:
            return "unknown"

        min_price = min(competitor_prices)
        max_price = max(competitor_prices)

        if price <= min_price:
            return "lowest"
        elif price >= max_price:
            return "highest"
        else:
            return "competitive"


# Singleton instance
_price_analytics_service: Optional['PriceAnalyticsService'] = None


def get_price_analytics_service() -> PriceAnalyticsService:
    """Get or create the Price Analytics service singleton."""
    global _price_analytics_service
    if _price_analytics_service is None:
        _price_analytics_service = PriceAnalyticsService()
    return _price_analytics_service



