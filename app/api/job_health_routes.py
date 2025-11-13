"""
Job Health Monitoring API Routes

Provides real-time monitoring of background job health:
- Live job status and progress
- Heartbeat monitoring
- Stuck job detection
- Performance metrics
- Health alerts
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from app.services.supabase_client import get_supabase_client
from app.services.job_monitor_service import job_monitor_service
from app.services.stuck_job_analyzer import stuck_job_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/job-health", tags=["job-health"])


@router.get("/dashboard")
async def get_job_health_dashboard() -> Dict[str, Any]:
    """
    Get comprehensive job health dashboard data.
    
    Returns:
        - Active jobs with progress
        - Stuck jobs with analysis
        - Performance metrics
        - Health alerts
    """
    try:
        supabase = get_supabase_client()
        
        # Get all background jobs from last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        result = supabase.client.table("background_jobs")\
            .select("*")\
            .gte("created_at", cutoff_time.isoformat())\
            .order("created_at", desc=True)\
            .execute()
        
        all_jobs = result.data or []
        
        # Categorize jobs
        active_jobs = [j for j in all_jobs if j['status'] == 'processing']
        completed_jobs = [j for j in all_jobs if j['status'] == 'completed']
        failed_jobs = [j for j in all_jobs if j['status'] == 'failed']
        pending_jobs = [j for j in all_jobs if j['status'] == 'pending']
        
        # Detect stuck jobs
        stuck_jobs = []
        for job in active_jobs:
            # Check heartbeat timeout (2 minutes)
            if job.get('last_heartbeat'):
                last_heartbeat = datetime.fromisoformat(job['last_heartbeat'].replace('Z', '+00:00'))
                if (datetime.utcnow() - last_heartbeat.replace(tzinfo=None)) > timedelta(minutes=2):
                    stuck_jobs.append({
                        **job,
                        'stuck_reason': 'heartbeat_timeout',
                        'stuck_duration_minutes': (datetime.utcnow() - last_heartbeat.replace(tzinfo=None)).total_seconds() / 60
                    })
            
            # Check updated_at timeout (5 minutes)
            elif job.get('updated_at'):
                updated_at = datetime.fromisoformat(job['updated_at'].replace('Z', '+00:00'))
                if (datetime.utcnow() - updated_at.replace(tzinfo=None)) > timedelta(minutes=5):
                    stuck_jobs.append({
                        **job,
                        'stuck_reason': 'updated_at_timeout',
                        'stuck_duration_minutes': (datetime.utcnow() - updated_at.replace(tzinfo=None)).total_seconds() / 60
                    })
        
        # Calculate metrics
        total_jobs = len(all_jobs)
        success_rate = (len(completed_jobs) / total_jobs * 100) if total_jobs > 0 else 0
        
        # Calculate average processing time for completed jobs
        processing_times = []
        for job in completed_jobs:
            if job.get('started_at') and job.get('completed_at'):
                started = datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
                completed = datetime.fromisoformat(job['completed_at'].replace('Z', '+00:00'))
                duration = (completed - started).total_seconds()
                processing_times.append(duration)
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Health status
        health_status = "healthy"
        health_alerts = []
        
        if len(stuck_jobs) > 0:
            health_status = "warning"
            health_alerts.append(f"{len(stuck_jobs)} stuck job(s) detected")
        
        if len(failed_jobs) > len(completed_jobs) * 0.1:  # >10% failure rate
            health_status = "critical"
            health_alerts.append(f"High failure rate: {len(failed_jobs)}/{total_jobs} jobs failed")
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "health_status": health_status,
            "health_alerts": health_alerts,
            "metrics": {
                "total_jobs_24h": total_jobs,
                "active_jobs": len(active_jobs),
                "completed_jobs": len(completed_jobs),
                "failed_jobs": len(failed_jobs),
                "pending_jobs": len(pending_jobs),
                "stuck_jobs": len(stuck_jobs),
                "success_rate": round(success_rate, 2),
                "avg_processing_time_seconds": round(avg_processing_time, 2)
            },
            "active_jobs": active_jobs[:10],  # Latest 10 active jobs
            "stuck_jobs": stuck_jobs,
            "recent_failures": failed_jobs[:5]  # Latest 5 failures
        }
    
    except Exception as e:
        logger.error(f"Failed to get job health dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stuck-jobs")
async def get_stuck_jobs() -> Dict[str, Any]:
    """Get all currently stuck jobs with analysis."""
    try:
        # Use stuck job analyzer service
        analysis = await stuck_job_analyzer.analyze_stuck_jobs()
        
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "stuck_jobs": analysis
        }
    
    except Exception as e:
        logger.error(f"Failed to get stuck jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}")
async def get_job_details(job_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific job."""
    try:
        supabase = get_supabase_client()
        
        # Get job from background_jobs
        result = supabase.client.table("background_jobs")\
            .select("*")\
            .eq("id", job_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job = result.data[0]
        
        return {
            "status": "success",
            "job": job
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

