"""
AI Metrics API endpoints.

Provides comprehensive AI usage metrics for monitoring dashboard:
- Real-time cost tracking
- Model usage statistics
- Confidence score distribution
- Latency metrics
- Fallback rate tracking
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..services.supabase_client import get_supabase_client, SupabaseClient

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ai-metrics", tags=["AI Metrics"])


# ============================================================================
# Response Models
# ============================================================================

class AIMetricsSummary(BaseModel):
    total_calls: int
    total_cost: float
    total_tokens: int
    average_latency_ms: float
    average_confidence: float
    fallback_rate: float
    time_period: str


class ModelUsageStats(BaseModel):
    model: str
    call_count: int
    total_cost: float
    total_tokens: int
    average_latency_ms: float
    average_confidence: float
    fallback_count: int


class TaskBreakdown(BaseModel):
    task: str
    call_count: int
    total_cost: float
    average_confidence: float
    models_used: List[str]


class ConfidenceDistribution(BaseModel):
    range: str  # "0.0-0.2", "0.2-0.4", etc.
    count: int
    percentage: float


class AIMetricsResponse(BaseModel):
    summary: AIMetricsSummary
    model_usage: List[ModelUsageStats]
    task_breakdown: List[TaskBreakdown]
    confidence_distribution: List[ConfidenceDistribution]
    recent_calls: List[Dict[str, Any]]


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/summary", response_model=AIMetricsResponse)
async def get_ai_metrics_summary(
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d, all"),
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """
    Get comprehensive AI metrics summary.
    
    Returns:
    - Summary statistics (total calls, cost, tokens, latency, confidence, fallback rate)
    - Model usage breakdown
    - Task breakdown
    - Confidence score distribution
    - Recent AI calls
    """
    try:
        # Calculate time range
        now = datetime.utcnow()
        if time_period == "1h":
            start_time = now - timedelta(hours=1)
        elif time_period == "24h":
            start_time = now - timedelta(hours=24)
        elif time_period == "7d":
            start_time = now - timedelta(days=7)
        elif time_period == "30d":
            start_time = now - timedelta(days=30)
        else:  # "all"
            start_time = datetime(2020, 1, 1)  # Far past
        
        # Query ai_call_logs table
        query = supabase.client.table("ai_call_logs").select("*")
        
        if time_period != "all":
            query = query.gte("timestamp", start_time.isoformat())
        
        response = query.execute()
        logs = response.data if response.data else []
        
        if not logs:
            return AIMetricsResponse(
                summary=AIMetricsSummary(
                    total_calls=0,
                    total_cost=0.0,
                    total_tokens=0,
                    average_latency_ms=0.0,
                    average_confidence=0.0,
                    fallback_rate=0.0,
                    time_period=time_period
                ),
                model_usage=[],
                task_breakdown=[],
                confidence_distribution=[],
                recent_calls=[]
            )
        
        # Calculate summary statistics
        total_calls = len(logs)
        total_cost = sum(float(log.get("cost", 0) or 0) for log in logs)
        total_tokens = sum(
            (log.get("input_tokens", 0) or 0) + (log.get("output_tokens", 0) or 0)
            for log in logs
        )
        average_latency = sum(log.get("latency_ms", 0) or 0 for log in logs) / total_calls
        average_confidence = sum(float(log.get("confidence_score", 0) or 0) for log in logs) / total_calls
        fallback_count = sum(1 for log in logs if log.get("action") == "fallback_to_rules")
        fallback_rate = fallback_count / total_calls if total_calls > 0 else 0.0
        
        summary = AIMetricsSummary(
            total_calls=total_calls,
            total_cost=round(total_cost, 4),
            total_tokens=total_tokens,
            average_latency_ms=round(average_latency, 2),
            average_confidence=round(average_confidence, 2),
            fallback_rate=round(fallback_rate, 2),
            time_period=time_period
        )
        
        # Calculate model usage stats
        model_stats = {}
        for log in logs:
            model = log.get("model", "unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "call_count": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "total_latency": 0,
                    "total_confidence": 0.0,
                    "fallback_count": 0
                }
            
            model_stats[model]["call_count"] += 1
            model_stats[model]["total_cost"] += float(log.get("cost", 0) or 0)
            model_stats[model]["total_tokens"] += (
                (log.get("input_tokens", 0) or 0) + (log.get("output_tokens", 0) or 0)
            )
            model_stats[model]["total_latency"] += log.get("latency_ms", 0) or 0
            model_stats[model]["total_confidence"] += float(log.get("confidence_score", 0) or 0)
            if log.get("action") == "fallback_to_rules":
                model_stats[model]["fallback_count"] += 1
        
        model_usage = [
            ModelUsageStats(
                model=model,
                call_count=stats["call_count"],
                total_cost=round(stats["total_cost"], 4),
                total_tokens=stats["total_tokens"],
                average_latency_ms=round(stats["total_latency"] / stats["call_count"], 2),
                average_confidence=round(stats["total_confidence"] / stats["call_count"], 2),
                fallback_count=stats["fallback_count"]
            )
            for model, stats in model_stats.items()
        ]
        model_usage.sort(key=lambda x: x.total_cost, reverse=True)
        
        # Calculate task breakdown
        task_stats = {}
        for log in logs:
            task = log.get("task", "unknown")
            if task not in task_stats:
                task_stats[task] = {
                    "call_count": 0,
                    "total_cost": 0.0,
                    "total_confidence": 0.0,
                    "models": set()
                }
            
            task_stats[task]["call_count"] += 1
            task_stats[task]["total_cost"] += float(log.get("cost", 0) or 0)
            task_stats[task]["total_confidence"] += float(log.get("confidence_score", 0) or 0)
            task_stats[task]["models"].add(log.get("model", "unknown"))
        
        task_breakdown = [
            TaskBreakdown(
                task=task,
                call_count=stats["call_count"],
                total_cost=round(stats["total_cost"], 4),
                average_confidence=round(stats["total_confidence"] / stats["call_count"], 2),
                models_used=list(stats["models"])
            )
            for task, stats in task_stats.items()
        ]
        task_breakdown.sort(key=lambda x: x.total_cost, reverse=True)
        
        # Calculate confidence distribution
        confidence_ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for log in logs:
            confidence = float(log.get("confidence_score", 0) or 0)
            if confidence < 0.2:
                confidence_ranges["0.0-0.2"] += 1
            elif confidence < 0.4:
                confidence_ranges["0.2-0.4"] += 1
            elif confidence < 0.6:
                confidence_ranges["0.4-0.6"] += 1
            elif confidence < 0.8:
                confidence_ranges["0.6-0.8"] += 1
            else:
                confidence_ranges["0.8-1.0"] += 1
        
        confidence_distribution = [
            ConfidenceDistribution(
                range=range_name,
                count=count,
                percentage=round((count / total_calls) * 100, 1) if total_calls > 0 else 0.0
            )
            for range_name, count in confidence_ranges.items()
        ]
        
        # Get recent calls (last 20)
        recent_calls = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
        
        return AIMetricsResponse(
            summary=summary,
            model_usage=model_usage,
            task_breakdown=task_breakdown,
            confidence_distribution=confidence_distribution,
            recent_calls=recent_calls
        )
        
    except Exception as e:
        logger.error(f"Failed to get AI metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get AI metrics: {str(e)}"
        )


@router.get("/job/{job_id}")
async def get_job_ai_metrics(
    job_id: str,
    supabase: SupabaseClient = Depends(get_supabase_client)
):
    """Get AI metrics for a specific job (PDF processing, etc.)."""
    try:
        response = supabase.client.table("ai_call_logs").select("*").eq("job_id", job_id).execute()
        logs = response.data if response.data else []
        
        if not logs:
            return {
                "job_id": job_id,
                "total_calls": 0,
                "total_cost": 0.0,
                "calls": []
            }
        
        total_cost = sum(float(log.get("cost", 0) or 0) for log in logs)
        
        return {
            "job_id": job_id,
            "total_calls": len(logs),
            "total_cost": round(total_cost, 4),
            "calls": logs
        }
        
    except Exception as e:
        logger.error(f"Failed to get job AI metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job AI metrics: {str(e)}"
        )

