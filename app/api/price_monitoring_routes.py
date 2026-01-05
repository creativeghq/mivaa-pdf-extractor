"""
Price Monitoring API Routes

FastAPI endpoints for price monitoring functionality:
- Start/stop monitoring
- On-demand price checks
- Get price history and statistics
- Manage competitor sources
- Configure price alerts
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field

from app.services.integrations.price_monitoring_service import get_price_monitoring_service
from app.services.core.supabase_client import get_supabase_client
from app.dependencies import get_current_user, get_workspace_context
from app.middleware.jwt_auth import User, WorkspaceContext

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/price-monitoring",
    tags=["price-monitoring"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not found"}
    }
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StartMonitoringRequest(BaseModel):
    """Request to start price monitoring for a product"""
    product_id: str = Field(..., description="Product UUID to monitor")
    frequency: str = Field(
        default="daily",
        description="Monitoring frequency: hourly, daily, weekly, on_demand"
    )
    enabled: bool = Field(default=True, description="Enable monitoring")


class CheckPricesRequest(BaseModel):
    """Request to perform on-demand price check"""
    product_id: str = Field(..., description="Product UUID")
    product_name: str = Field(..., description="Product name for scraping context")


class AddCompetitorSourceRequest(BaseModel):
    """Request to add a competitor source"""
    product_id: str = Field(..., description="Product UUID")
    source_name: str = Field(..., description="Competitor name")
    source_url: str = Field(..., description="Competitor product URL")
    scraping_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional Firecrawl configuration"
    )


class CreatePriceAlertRequest(BaseModel):
    """Request to create a price alert"""
    product_id: str = Field(..., description="Product UUID")
    alert_type: str = Field(
        ...,
        description="Alert type: price_drop, price_increase, any_change, availability"
    )
    threshold_percentage: Optional[float] = Field(
        default=None,
        description="Percentage threshold for alert"
    )
    threshold_amount: Optional[float] = Field(
        default=None,
        description="Amount threshold for alert"
    )
    notification_channels: List[str] = Field(
        default=["email"],
        description="Notification channels: email, in_app, sms"
    )


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@router.post("/start")
async def start_monitoring(
    request: StartMonitoringRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context)
):
    """
    Start price monitoring for a product.
    
    Requires Factory or Store role.
    """
    try:
        service = get_price_monitoring_service()
        
        result = await service.start_monitoring(
            product_id=request.product_id,
            user_id=str(user.id),
            workspace_id=str(workspace.workspace_id),
            frequency=request.frequency,
            enabled=request.enabled
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": f"Started {request.frequency} monitoring",
                "monitoring": result.get("monitoring")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to start monitoring")
            )
            
    except Exception as e:
        logger.error(f"❌ Failed to start monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/stop")
async def stop_monitoring(
    product_id: str = Query(..., description="Product UUID"),
    user: User = Depends(get_current_user)
):
    """
    Stop price monitoring for a product.
    """
    try:
        service = get_price_monitoring_service()
        
        result = await service.stop_monitoring(
            product_id=product_id,
            user_id=str(user.id)
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": "Monitoring stopped"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Failed to stop monitoring")
            )
            
    except Exception as e:
        logger.error(f"❌ Failed to stop monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/check-now")
async def check_prices_now(
    request: CheckPricesRequest,
    user: User = Depends(get_current_user),
    workspace: WorkspaceContext = Depends(get_workspace_context)
):
    """
    Perform on-demand price check for a product.

    Scrapes all active competitor sources and saves price history.
    Debits Firecrawl credits from user's account.
    """
    try:
        service = get_price_monitoring_service()

        result = await service.check_prices_now(
            product_id=request.product_id,
            user_id=str(user.id),
            workspace_id=str(workspace.workspace_id),
            product_name=request.product_name
        )

        if result.get("success"):
            return {
                "success": True,
                "message": f"Checked {result.get('sources_checked', 0)} sources",
                "job_id": result.get("job_id"),
                "sources_checked": result.get("sources_checked", 0),
                "prices_found": result.get("prices_found", 0),
                "credits_consumed": result.get("credits_consumed", 0)
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Price check failed")
            )

    except Exception as e:
        logger.error(f"❌ Failed to check prices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/status/{product_id}")
async def get_monitoring_status(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get monitoring status for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_monitoring_products").select("*").eq(
            "product_id", product_id
        ).eq("user_id", str(user.id)).execute()

        if response.data:
            return {
                "success": True,
                "monitoring": response.data[0]
            }
        else:
            return {
                "success": True,
                "monitoring": None,
                "message": "No monitoring configured"
            }

    except Exception as e:
        logger.error(f"❌ Failed to get monitoring status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# PRICE HISTORY ENDPOINTS
# ============================================================================

@router.get("/history/{product_id}")
async def get_price_history(
    product_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    source_name: Optional[str] = Query(default=None),
    user: User = Depends(get_current_user)
):
    """
    Get price history for a product.

    Optionally filter by source_name.
    """
    try:
        supabase = get_supabase_client()

        query = supabase.client.table("price_history").select("*").eq(
            "product_id", product_id
        ).order("scraped_at", desc=True).limit(limit)

        if source_name:
            query = query.eq("source_name", source_name)

        response = query.execute()

        return {
            "success": True,
            "history": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get price history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/statistics/{product_id}")
async def get_price_statistics(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get price statistics for a product.

    Returns min, max, avg prices and trend analysis.
    """
    try:
        service = get_price_monitoring_service()

        stats = await service.get_price_statistics(product_id)

        return {
            "success": True,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"❌ Failed to get price statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# COMPETITOR SOURCES ENDPOINTS
# ============================================================================

@router.post("/sources")
async def add_competitor_source(
    request: AddCompetitorSourceRequest,
    user: User = Depends(get_current_user)
):
    """
    Add a competitor source for price monitoring.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("competitor_sources").insert({
            "product_id": request.product_id,
            "source_name": request.source_name,
            "source_url": request.source_url,
            "scraping_config": request.scraping_config or {},
            "is_active": True,
            "error_count": 0,
            "created_by": str(user.id)
        }).execute()

        if response.data:
            return {
                "success": True,
                "message": "Competitor source added",
                "source": response.data[0]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add competitor source"
            )

    except Exception as e:
        logger.error(f"❌ Failed to add competitor source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/sources/{product_id}")
async def get_competitor_sources(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get all competitor sources for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("competitor_sources").select("*").eq(
            "product_id", product_id
        ).order("created_at", desc=True).execute()

        return {
            "success": True,
            "sources": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get competitor sources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/sources/{source_id}")
async def delete_competitor_source(
    source_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete a competitor source.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("competitor_sources").delete().eq(
            "id", source_id
        ).execute()

        return {
            "success": True,
            "message": "Competitor source deleted"
        }

    except Exception as e:
        logger.error(f"❌ Failed to delete competitor source: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# PRICE ALERTS ENDPOINTS
# ============================================================================

@router.post("/alerts")
async def create_price_alert(
    request: CreatePriceAlertRequest,
    user: User = Depends(get_current_user)
):
    """
    Create a price alert for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_alerts").insert({
            "user_id": str(user.id),
            "product_id": request.product_id,
            "alert_type": request.alert_type,
            "threshold_percentage": request.threshold_percentage,
            "threshold_amount": request.threshold_amount,
            "notification_channels": request.notification_channels,
            "is_active": True,
            "trigger_count": 0
        }).execute()

        if response.data:
            return {
                "success": True,
                "message": "Price alert created",
                "alert": response.data[0]
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create price alert"
            )

    except Exception as e:
        logger.error(f"❌ Failed to create price alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/alerts/{product_id}")
async def get_price_alerts(
    product_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get all price alerts for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_alerts").select("*").eq(
            "product_id", product_id
        ).eq("user_id", str(user.id)).order("created_at", desc=True).execute()

        return {
            "success": True,
            "alerts": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get price alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/alerts/{alert_id}")
async def delete_price_alert(
    alert_id: str,
    user: User = Depends(get_current_user)
):
    """
    Delete a price alert.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_alerts").delete().eq(
            "id", alert_id
        ).eq("user_id", str(user.id)).execute()

        return {
            "success": True,
            "message": "Price alert deleted"
        }

    except Exception as e:
        logger.error(f"❌ Failed to delete price alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# JOB MONITORING ENDPOINTS
# ============================================================================

@router.get("/jobs/{product_id}")
async def get_monitoring_jobs(
    product_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(get_current_user)
):
    """
    Get monitoring job history for a product.
    """
    try:
        supabase = get_supabase_client()

        response = supabase.client.table("price_monitoring_jobs").select("*").eq(
            "product_id", product_id
        ).eq("user_id", str(user.id)).order("created_at", desc=True).limit(limit).execute()

        return {
            "success": True,
            "jobs": response.data or [],
            "count": len(response.data or [])
        }

    except Exception as e:
        logger.error(f"❌ Failed to get monitoring jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )




