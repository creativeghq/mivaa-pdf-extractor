"""
Admin Restart Routes - Authorized service restart with job protection
"""

import os
import subprocess
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Body
from pydantic import BaseModel
import logging

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class RestartRequest(BaseModel):
    """Request model for service restart"""
    force: bool = False
    reason: str
    admin_token: str


class RestartResponse(BaseModel):
    """Response model for restart request"""
    success: bool
    message: str
    active_jobs: int
    interrupted_jobs: list[str] = []
    timestamp: str


@router.post("/restart-service", response_model=RestartResponse)
async def restart_service(
    request: RestartRequest
) -> RestartResponse:
    """
    Restart the MIVAA service with job protection.
    
    **Protection Rules**:
    - If active jobs exist and force=False: BLOCK restart
    - If active jobs exist and force=True: Interrupt jobs and restart
    - If no active jobs: Restart immediately
    
    **Required**:
    - admin_token: Must match ADMIN_RESTART_TOKEN environment variable
    - reason: Explanation for restart (logged to Sentry)
    
    **Example**:
    ```json
    {
        "force": false,
        "reason": "Deploy critical security patch",
        "admin_token": "your-secret-token"
    }
    ```
    """
    
    # Validate admin token
    expected_token = os.getenv("ADMIN_RESTART_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_RESTART_TOKEN not configured on server"
        )
    
    if request.admin_token != expected_token:
        logger.warning(f"🔒 Unauthorized restart attempt with reason: {request.reason}")
        raise HTTPException(
            status_code=403,
            detail="Invalid admin token"
        )
    
    # Check for active jobs
    supabase = get_supabase_client()
    active_jobs_response = supabase.client.table('background_jobs')\
        .select('id, filename, progress, status')\
        .eq('status', 'processing')\
        .execute()
    
    active_jobs = active_jobs_response.data or []
    active_count = len(active_jobs)
    
    # If no active jobs, restart immediately
    if active_count == 0:
        logger.info(f"✅ Safe restart initiated: {request.reason}")
        
        try:
            # Restart service
            subprocess.run(
                ["sudo", "systemctl", "restart", "mivaa-pdf-extractor"],
                check=True,
                capture_output=True,
                text=True
            )
            
            return RestartResponse(
                success=True,
                message=f"Service restarted successfully. Reason: {request.reason}",
                active_jobs=0,
                interrupted_jobs=[],
                timestamp=datetime.utcnow().isoformat()
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Restart failed: {e.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Restart command failed: {e.stderr}"
            )
    
    # Active jobs exist
    if not request.force:
        # BLOCK restart
        job_list = [f"{j['id']}: {j['filename']} ({j['progress']}%)" for j in active_jobs]
        logger.warning(f"🛑 Restart blocked - {active_count} active jobs. Reason: {request.reason}")
        
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Active jobs detected - restart blocked",
                "active_jobs": active_count,
                "jobs": job_list,
                "suggestion": "Use force=true to interrupt jobs, or wait for completion"
            }
        )
    
    # Force restart - interrupt jobs
    logger.warning(f"⚠️ FORCE RESTART - Interrupting {active_count} jobs. Reason: {request.reason}")
    
    interrupted_job_ids = []
    for job in active_jobs:
        job_id = job['id']
        try:
            # Mark as interrupted
            supabase.client.table('background_jobs').update({
                'status': 'interrupted',
                'error': f'Service restart (forced): {request.reason}',
                'interrupted_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', job_id).execute()
            
            interrupted_job_ids.append(job_id)
            logger.info(f"   ✅ Interrupted job {job_id}")
        except Exception as e:
            logger.error(f"   ❌ Failed to interrupt job {job_id}: {e}")
    
    # Restart service
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", "mivaa-pdf-extractor"],
            check=True,
            capture_output=True,
            text=True
        )
        
        return RestartResponse(
            success=True,
            message=f"Service force restarted. {len(interrupted_job_ids)} jobs interrupted. Reason: {request.reason}",
            active_jobs=active_count,
            interrupted_jobs=interrupted_job_ids,
            timestamp=datetime.utcnow().isoformat()
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Force restart failed: {e.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"Restart command failed: {e.stderr}"
        )


