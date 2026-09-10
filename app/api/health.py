"""Health Check API Endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
import logging

from app.dependencies import require_admin
from app.services.core.database_health_service import database_health_service
from app.services.tracking.job_monitor_service import job_monitor_service
from app.utils.query_metrics import query_metrics

logger = logging.getLogger(__name__)


# Response Models for OpenAPI Documentation
class BasicHealthResponse(BaseModel):
    """Basic health check response"""
    status: Literal["healthy"] = Field(description="Service status")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")


class DatabasePerformance(BaseModel):
    """Database performance metrics"""
    avg_query_time_ms: float = Field(description="Average query execution time in milliseconds")
    max_query_time_ms: float = Field(description="Maximum query execution time in milliseconds")
    slow_query_count: int = Field(description="Number of slow queries detected")
    slow_query_threshold_ms: float = Field(description="Threshold for slow query detection in milliseconds")


class DatabaseHealthResponse(BaseModel):
    """Database health status"""
    healthy: bool = Field(description="Whether database is healthy")
    connection_test_ms: float = Field(description="Connection test time in milliseconds")
    query_test_ms: float = Field(description="Query test time in milliseconds")
    error_count: int = Field(description="Total error count")
    consecutive_failures: int = Field(description="Consecutive failure count")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    performance: DatabasePerformance


class JobMonitorHealthResponse(BaseModel):
    """Job monitor health status"""
    monitor_running: bool = Field(description="Whether job monitor is running")
    stuck_jobs_count: int = Field(description="Number of stuck jobs detected")
    health: Literal["healthy", "degraded", "unhealthy"] = Field(description="Overall health status")


class QueryMetricsResponse(BaseModel):
    """Query performance metrics"""
    total_queries: int = Field(description="Total number of queries executed")
    slow_queries: int = Field(description="Number of slow queries")
    slow_query_percentage: float = Field(description="Percentage of slow queries")
    avg_query_time_ms: float = Field(description="Average query time in milliseconds")
    max_query_time_ms: float = Field(description="Maximum query time in milliseconds")


class CircuitBreakerState(BaseModel):
    """Circuit breaker state"""
    state: Literal["closed", "open", "half_open"] = Field(description="Circuit breaker state")
    failure_count: int = Field(description="Number of failures")


class DetailedHealthResponse(BaseModel):
    """Detailed system health status"""
    overall_status: Literal["healthy", "degraded", "unhealthy"] = Field(description="Overall system status")
    database: DatabaseHealthResponse
    job_monitor: JobMonitorHealthResponse
    query_metrics: QueryMetricsResponse
    circuit_breaker: CircuitBreakerState
    timestamp: str = Field(description="Timestamp of health check")


router = APIRouter(
    prefix="/health",
    tags=["Health & Monitoring"],
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)


@router.get(
    "/",
    response_model=BasicHealthResponse,
    summary="Basic Health Check",
    description="Quick health check to verify service is running. Returns 200 if service is operational."
)
async def health_check() -> BasicHealthResponse:
    """
    **Basic Health Check**

    Quick endpoint to verify the service is running and responsive.

    **Use Case**: Load balancer health checks, uptime monitoring

    **Response Time**: <10ms
    """
    return BasicHealthResponse(
        status="healthy",
        service="mivaa-pdf-extractor",
        version="2.4.0"
    )


@router.get(
    "/detailed",
    response_model=DetailedHealthResponse,
    summary="Detailed System Health",
    description="Comprehensive health check including database, job monitor, query metrics, and circuit breaker status."
)
async def detailed_health_check() -> DetailedHealthResponse:
    """**Detailed System Health Check**"""
    try:
        # Get database health
        db_health = database_health_service.get_health_status()
        
        # Get job monitor health
        monitor_health = await job_monitor_service.get_health_status()
        
        # Get query metrics
        metrics = query_metrics.get_metrics()

        # Circuit breaker status - JobMonitorService no longer uses circuit breaker
        # Return a simple status based on database health
        circuit_breaker_status = {
            "state": "closed" if db_health["healthy"] else "open",
            "failure_count": db_health.get("consecutive_failures", 0)
        }

        # Determine overall status
        overall_status = "healthy"

        if not db_health["healthy"]:
            overall_status = "unhealthy"
        elif monitor_health["health"] == "degraded":
            overall_status = "degraded"
        elif metrics["slow_query_percentage"] > 20:  # More than 20% slow queries
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "database": db_health,
            "job_monitor": monitor_health,
            "query_metrics": metrics,
            "circuit_breaker": circuit_breaker_status,
            "timestamp": db_health["timestamp"]
        }
    
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get(
    "/database",
    response_model=Dict[str, Any],
    summary="Database Health",
    description="Database connection pool health and performance metrics"
)
async def database_health() -> Dict[str, Any]:
    """**Database Connection Health**"""
    return database_health_service.get_health_status()


@router.get(
    "/job-monitor",
    response_model=Dict[str, Any],
    summary="Job Monitor Health",
    description="Job monitoring service status and stuck job detection"
)
async def job_monitor_health() -> Dict[str, Any]:
    """**Job Monitoring Service Health**"""
    return await job_monitor_service.get_health_status()


@router.get(
    "/metrics",
    response_model=QueryMetricsResponse,
    summary="Query Performance Metrics",
    description="Database query performance statistics and slow query tracking"
)
async def performance_metrics() -> QueryMetricsResponse:
    """**Query Performance Metrics**"""
    return query_metrics.get_metrics()


@router.get(
    "/circuit-breakers",
    response_model=Dict[str, CircuitBreakerState],
    summary="Circuit Breaker Status",
    description="Circuit breaker states for all protected services"
)
async def circuit_breaker_status() -> Dict[str, CircuitBreakerState]:
    """**Circuit Breaker Status**"""
    # Get database health to determine circuit breaker status
    db_health = database_health_service.get_health_status()

    return {
        "job_monitor_db": {
            "state": "closed" if db_health.get("healthy", False) else "open",
            "failure_count": db_health.get("consecutive_failures", 0)
        }
    }


@router.post(
    "/metrics/reset",
    response_model=Dict[str, str],
    summary="Reset Query Metrics",
    description="Reset query performance metrics (admin only)"
)
async def reset_metrics(_admin = Depends(require_admin)) -> Dict[str, str]:
    """
    **Reset Query Performance Metrics**

    Clears all query performance metrics and statistics.

    **Use Case**: Testing, post-maintenance cleanup

    **⚠️ Warning**: This will reset all historical query metrics.
    Requires admin permission.
    """
    query_metrics.reset_metrics()
    return {"status": "metrics_reset", "message": "Query metrics have been reset"}


