"""
Price Monitoring Service

Core service for price monitoring functionality:
- Start/stop monitoring for products
- On-demand price checks
- Price comparison and change detection
- Integration with CompetitorScraperService
- Job management and status tracking
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.competitor_scraper_service import get_competitor_scraper_service
from app.config import get_settings

logger = logging.getLogger(__name__)


class PriceMonitoringError(Exception):
    """Base exception for price monitoring errors"""
    pass


class PriceMonitoringService:
    """
    Core service for price monitoring operations.
    
    Features:
    - Start/stop monitoring for products
    - On-demand price checks from competitor sources
    - Price comparison and change detection
    - Alert triggering based on price changes
    - Job tracking and status management
    """
    
    def __init__(self):
        """Initialize the price monitoring service."""
        self.supabase = get_supabase_client()
        self.scraper = get_competitor_scraper_service()
        self.settings = get_settings()
        self.logger = logger
    
    async def start_monitoring(
        self,
        product_id: str,
        user_id: str,
        workspace_id: str,
        frequency: str = "daily",
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Start price monitoring for a product.
        
        Args:
            product_id: Product UUID to monitor
            user_id: User ID initiating monitoring
            workspace_id: Workspace ID
            frequency: Monitoring frequency ('hourly', 'daily', 'weekly', 'on_demand')
            enabled: Whether monitoring is enabled
            
        Returns:
            Dict with monitoring record
        """
        try:
            # Calculate next check time
            next_check_at = None
            if frequency != "on_demand":
                intervals = {
                    "hourly": timedelta(hours=1),
                    "daily": timedelta(days=1),
                    "weekly": timedelta(weeks=1)
                }
                next_check_at = (datetime.utcnow() + intervals[frequency]).isoformat()
            
            # Upsert monitoring record
            response = self.supabase.client.table("price_monitoring_products").upsert({
                "product_id": product_id,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "monitoring_enabled": enabled,
                "monitoring_frequency": frequency,
                "next_check_at": next_check_at,
                "status": "active",
                "updated_at": datetime.utcnow().isoformat()
            }, on_conflict="product_id,user_id").execute()
            
            if response.data:
                self.logger.info(
                    f"✅ Started {frequency} monitoring for product {product_id}"
                )
                return {
                    "success": True,
                    "monitoring": response.data[0]
                }
            else:
                raise PriceMonitoringError("Failed to create monitoring record")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitoring: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stop_monitoring(
        self,
        product_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Stop price monitoring for a product.
        
        Args:
            product_id: Product UUID
            user_id: User ID
            
        Returns:
            Dict with success status
        """
        try:
            response = self.supabase.client.table("price_monitoring_products").update({
                "monitoring_enabled": False,
                "status": "paused",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("product_id", product_id).eq("user_id", user_id).execute()
            
            self.logger.info(f"✅ Stopped monitoring for product {product_id}")
            return {
                "success": True,
                "monitoring": response.data[0] if response.data else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop monitoring: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def check_prices_now(
        self,
        product_id: str,
        user_id: str,
        workspace_id: str,
        product_name: str
    ) -> Dict[str, Any]:
        """
        Perform on-demand price check for a product.

        Args:
            product_id: Product UUID
            user_id: User ID for credit debit
            workspace_id: Workspace ID
            product_name: Product name for scraping context

        Returns:
            Dict with job results and prices found
        """
        job_id = str(uuid4())

        try:
            # Create monitoring job
            job_response = self.supabase.client.table("price_monitoring_jobs").insert({
                "id": job_id,
                "product_id": product_id,
                "user_id": user_id,
                "job_type": "on_demand",
                "status": "pending",
                "sources_checked": 0,
                "prices_found": 0,
                "credits_consumed": 0,
                "retry_count": 0
            }).execute()

            if not job_response.data:
                raise PriceMonitoringError("Failed to create monitoring job")

            # Update job status to running
            self.supabase.client.table("price_monitoring_jobs").update({
                "status": "running",
                "started_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()

            # Get active competitor sources
            sources_response = self.supabase.client.table("competitor_sources").select("*").eq(
                "product_id", product_id
            ).eq("is_active", True).execute()

            sources = sources_response.data or []

            if not sources:
                # No sources configured
                self.supabase.client.table("price_monitoring_jobs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "error_message": "No active competitor sources configured"
                }).eq("id", job_id).execute()

                return {
                    "success": True,
                    "job_id": job_id,
                    "sources_checked": 0,
                    "prices_found": 0,
                    "message": "No competitor sources configured"
                }

            # Scrape prices from all sources
            sources_checked = 0
            prices_found = 0
            total_credits = 0

            for source in sources:
                try:
                    # Scrape competitor price
                    result = await self.scraper.scrape_competitor_price(
                        url=source["source_url"],
                        product_name=product_name,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        scraping_config=source.get("scraping_config", {})
                    )

                    sources_checked += 1
                    total_credits += result.get("credits_used", 0)

                    if result.get("success") and result.get("price"):
                        # Save to price_history
                        self.supabase.client.table("price_history").insert({
                            "product_id": product_id,
                            "source_name": source["source_name"],
                            "source_url": source["source_url"],
                            "price": float(result["price"]),
                            "currency": result.get("currency", "USD"),
                            "availability": result.get("availability", "unknown"),
                            "scraped_at": result.get("scraped_at"),
                            "metadata": result.get("raw_data", {})
                        }).execute()

                        prices_found += 1

                        # Update source success
                        self.supabase.client.table("competitor_sources").update({
                            "last_successful_scrape": datetime.utcnow().isoformat(),
                            "error_count": 0,
                            "last_error": None
                        }).eq("id", source["id"]).execute()

                        self.logger.info(
                            f"✅ Found price {result['price']} {result.get('currency')} "
                            f"from {source['source_name']}"
                        )
                    else:
                        # Update source error
                        error_count = source.get("error_count", 0) + 1
                        self.supabase.client.table("competitor_sources").update({
                            "error_count": error_count,
                            "last_error": result.get("error", "Unknown error")
                        }).eq("id", source["id"]).execute()

                except Exception as e:
                    self.logger.error(
                        f"❌ Failed to scrape {source['source_name']}: {e}"
                    )
                    sources_checked += 1

            # Update job completion
            self.supabase.client.table("price_monitoring_jobs").update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "sources_checked": sources_checked,
                "prices_found": prices_found,
                "credits_consumed": total_credits
            }).eq("id", job_id).execute()

            # Check for price alerts
            if prices_found > 0:
                await self._check_price_alerts(product_id, user_id)

            # Update monitoring record
            self.supabase.client.table("price_monitoring_products").update({
                "last_check_at": datetime.utcnow().isoformat()
            }).eq("product_id", product_id).eq("user_id", user_id).execute()

            return {
                "success": True,
                "job_id": job_id,
                "sources_checked": sources_checked,
                "prices_found": prices_found,
                "credits_consumed": total_credits
            }

        except Exception as e:
            self.logger.error(f"❌ Price check failed: {e}")

            # Update job as failed
            self.supabase.client.table("price_monitoring_jobs").update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": str(e)
            }).eq("id", job_id).execute()

            return {
                "success": False,
                "job_id": job_id,
                "error": str(e)
            }

    async def _check_price_alerts(
        self,
        product_id: str,
        user_id: str
    ) -> None:
        """
        Check if price changes should trigger alerts.

        Args:
            product_id: Product UUID
            user_id: User ID
        """
        try:
            # Get active alerts for this product
            alerts_response = self.supabase.client.table("price_alerts").select("*").eq(
                "product_id", product_id
            ).eq("user_id", user_id).eq("is_active", True).execute()

            alerts = alerts_response.data or []

            if not alerts:
                return

            # Get latest price
            latest_response = self.supabase.client.table("price_history").select("*").eq(
                "product_id", product_id
            ).order("scraped_at", desc=True).limit(1).execute()

            if not latest_response.data:
                return

            latest_price = latest_response.data[0]

            # Get previous price for comparison
            previous_response = self.supabase.client.table("price_history").select("*").eq(
                "product_id", product_id
            ).eq("source_name", latest_price["source_name"]).order(
                "scraped_at", desc=True
            ).limit(1).offset(1).execute()

            if not previous_response.data:
                return  # Need at least 2 data points

            previous_price = previous_response.data[0]

            # Check each alert
            for alert in alerts:
                should_trigger = await self._should_trigger_alert(
                    alert,
                    previous_price["price"],
                    latest_price["price"]
                )

                if should_trigger:
                    # Calculate price change
                    price_change = latest_price["price"] - previous_price["price"]
                    price_change_pct = (price_change / previous_price["price"]) * 100

                    # Create alert history record
                    self.supabase.client.table("price_alert_history").insert({
                        "alert_id": alert["id"],
                        "user_id": user_id,
                        "product_id": product_id,
                        "alert_type": alert["alert_type"],
                        "old_price": previous_price["price"],
                        "new_price": latest_price["price"],
                        "price_change_percentage": float(price_change_pct),
                        "price_change_amount": float(price_change),
                        "source_name": latest_price["source_name"],
                        "source_url": latest_price["source_url"],
                        "notification_sent": False
                    }).execute()

                    # Update alert trigger count
                    self.supabase.client.table("price_alerts").update({
                        "last_triggered_at": datetime.utcnow().isoformat(),
                        "trigger_count": alert.get("trigger_count", 0) + 1
                    }).eq("id", alert["id"]).execute()

                    self.logger.info(
                        f"🔔 Alert triggered for product {product_id}: "
                        f"{alert['alert_type']} - {price_change_pct:.2f}% change"
                    )

        except Exception as e:
            self.logger.error(f"❌ Failed to check price alerts: {e}")

    async def _should_trigger_alert(
        self,
        alert: Dict[str, Any],
        old_price: float,
        new_price: float
    ) -> bool:
        """
        Determine if an alert should be triggered based on price change.

        Args:
            alert: Alert configuration
            old_price: Previous price
            new_price: Current price

        Returns:
            True if alert should trigger
        """
        alert_type = alert["alert_type"]
        threshold_pct = alert.get("threshold_percentage")
        threshold_amt = alert.get("threshold_amount")

        price_change = new_price - old_price
        price_change_pct = (price_change / old_price) * 100 if old_price > 0 else 0

        if alert_type == "price_drop":
            if threshold_pct and price_change_pct <= -threshold_pct:
                return True
            if threshold_amt and price_change <= -threshold_amt:
                return True

        elif alert_type == "price_increase":
            if threshold_pct and price_change_pct >= threshold_pct:
                return True
            if threshold_amt and price_change >= threshold_amt:
                return True

        elif alert_type == "any_change":
            if threshold_pct and abs(price_change_pct) >= threshold_pct:
                return True
            if threshold_amt and abs(price_change) >= threshold_amt:
                return True

        return False

    async def get_price_statistics(
        self,
        product_id: str
    ) -> Dict[str, Any]:
        """
        Get price statistics for a product.

        Args:
            product_id: Product UUID

        Returns:
            Dict with min, max, avg prices and trend
        """
        try:
            # Get all price history
            response = self.supabase.client.table("price_history").select("*").eq(
                "product_id", product_id
            ).order("scraped_at", desc=True).execute()

            prices = response.data or []

            if not prices:
                return {
                    "min_price": None,
                    "max_price": None,
                    "avg_price": None,
                    "current_price": None,
                    "price_trend": "insufficient_data",
                    "total_sources": 0
                }

            price_values = [p["price"] for p in prices]

            # Calculate statistics
            min_price = min(price_values)
            max_price = max(price_values)
            avg_price = sum(price_values) / len(price_values)
            current_price = prices[0]["price"]

            # Determine trend (simple: compare current to average)
            if len(prices) < 3:
                trend = "insufficient_data"
            elif current_price > avg_price * 1.05:
                trend = "increasing"
            elif current_price < avg_price * 0.95:
                trend = "decreasing"
            else:
                trend = "stable"

            # Count unique sources
            unique_sources = len(set(p["source_name"] for p in prices))

            return {
                "min_price": float(min_price),
                "max_price": float(max_price),
                "avg_price": float(avg_price),
                "current_price": float(current_price),
                "price_trend": trend,
                "total_sources": unique_sources
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get price statistics: {e}")
            return {
                "error": str(e)
            }


# Singleton instance
_price_monitoring_service: Optional[PriceMonitoringService] = None


def get_price_monitoring_service() -> PriceMonitoringService:
    """Get singleton instance of PriceMonitoringService."""
    global _price_monitoring_service
    if _price_monitoring_service is None:
        _price_monitoring_service = PriceMonitoringService()
    return _price_monitoring_service




