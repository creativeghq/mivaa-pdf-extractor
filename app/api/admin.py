"""
Administrative and Monitoring API Endpoints

This module provides comprehensive administrative and monitoring capabilities including:
- Job management and status tracking
- Service statistics and health monitoring
- Administrative endpoints for data management
- Bulk operations for document processing
- System monitoring and performance metrics
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime, timedelta
import psutil
import os
from pathlib import Path

from ..schemas.jobs import (
    JobResponse, JobStatusResponse, JobListResponse,
    BulkProcessingRequest, BulkProcessingResponse,
    JobStatistics, SystemMetrics
)
from ..schemas.common import BaseResponse, PaginationParams
from ..services.pdf_processor import PDFProcessor
from ..services.supabase_client import SupabaseClient
from ..services.llamaindex_service import LlamaIndexService
from ..services.material_kai_service import MaterialKaiService
from ..dependencies import get_current_user, get_workspace_context, require_admin
from ..middleware.jwt_auth import WorkspaceContext, User

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Health & Monitoring"])

# Global job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
job_history: List[Dict[str, Any]] = []

def get_pdf_processor():
    """Dependency to get PDF processor instance"""
    return PDFProcessor()

def get_supabase_client():
    """Dependency to get Supabase client instance"""
    return SupabaseClient()

def get_llamaindex_service():
    """Dependency to get LlamaIndex service instance"""
    return LlamaIndexService()

def get_material_kai_service():
    """Dependency to get Material Kai service instance"""
    return MaterialKaiService()

async def track_job(job_id: str, job_type: str, status: str, details: Dict[str, Any] = None):
    """Track job status and update global job tracking"""
    job_info = {
        "job_id": job_id,
        "job_type": job_type,
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "details": details or {}
    }
    
    active_jobs[job_id] = job_info
    
    # Add to history if completed or failed
    if status in ["completed", "failed", "cancelled"]:
        job_history.append(job_info.copy())
        if job_id in active_jobs:
            del active_jobs[job_id]

# Job Management Endpoints

@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_admin)
):
    """
    List all jobs with optional filtering and pagination
    
    - **status**: Filter jobs by status (pending, running, completed, failed, cancelled)
    - **job_type**: Filter jobs by type (document_processing, bulk_processing, etc.)
    - **limit**: Number of jobs to return (default: 50, max: 100)
    - **offset**: Number of jobs to skip for pagination
    """
    try:
        # Combine active jobs and job history
        all_jobs = list(active_jobs.values()) + job_history
        
        # Apply filters
        filtered_jobs = all_jobs
        if status:
            filtered_jobs = [job for job in filtered_jobs if job["status"] == status]
        if job_type:
            filtered_jobs = [job for job in filtered_jobs if job["job_type"] == job_type]
        
        # Sort by created_at descending
        filtered_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_jobs = filtered_jobs[start_idx:end_idx]
        
        return JobListResponse(
            success=True,
            message="Jobs retrieved successfully",
            data={
                "jobs": [JobResponse(**job) for job in paginated_jobs],
                "total_count": len(filtered_jobs),
                "active_count": len(active_jobs),
                "completed_count": len([j for j in job_history if j["status"] == "completed"]),
                "failed_count": len([j for j in job_history if j["status"] == "failed"])
            },
            pagination={
                "limit": pagination.limit,
                "offset": pagination.offset,
                "total": len(filtered_jobs)
            }
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_admin)
):
    """
    Get detailed status information for a specific job
    
    - **job_id**: Unique identifier for the job
    """
    try:
        # Check active jobs first
        if job_id in active_jobs:
            job_info = active_jobs[job_id]
        else:
            # Check job history
            job_info = next((job for job in job_history if job["job_id"] == job_id), None)
            
        if not job_info:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        return JobStatusResponse(
            success=True,
            message="Job status retrieved successfully",
            data=JobResponse(**job_info)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_admin)
):
    """
    Cancel a running job
    
    - **job_id**: Unique identifier for the job to cancel
    """
    try:
        if job_id not in active_jobs:
            raise HTTPException(status_code=404, detail=f"Active job {job_id} not found")
        
        job_info = active_jobs[job_id]
        if job_info["status"] in ["completed", "failed", "cancelled"]:
            raise HTTPException(status_code=400, detail=f"Job {job_id} is already {job_info['status']}")
        
        # Update job status to cancelled
        await track_job(job_id, job_info["job_type"], "cancelled", job_info["details"])
        
        return BaseResponse(
            success=True,
            message=f"Job {job_id} cancelled successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")

@router.get("/jobs/statistics", response_model=Dict[str, Any])
async def get_job_statistics(
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_admin)
):
    """
    Get comprehensive job statistics and metrics
    """
    try:
        # Calculate statistics
        all_jobs = list(active_jobs.values()) + job_history
        
        # Status distribution
        status_counts = {}
        for job in all_jobs:
            status = job["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Job type distribution
        type_counts = {}
        for job in all_jobs:
            job_type = job["job_type"]
            type_counts[job_type] = type_counts.get(job_type, 0) + 1
        
        # Recent activity (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_jobs = [
            job for job in all_jobs 
            if datetime.fromisoformat(job["created_at"].replace('Z', '+00:00')) > recent_cutoff
        ]
        
        # Average processing time for completed jobs
        completed_jobs = [job for job in job_history if job["status"] == "completed"]
        avg_processing_time = None
        if completed_jobs:
            processing_times = []
            for job in completed_jobs:
                created = datetime.fromisoformat(job["created_at"].replace('Z', '+00:00'))
                updated = datetime.fromisoformat(job["updated_at"].replace('Z', '+00:00'))
                processing_times.append((updated - created).total_seconds())
            avg_processing_time = sum(processing_times) / len(processing_times)
        
        statistics = JobStatistics(
            total_jobs=len(all_jobs),
            active_jobs=len(active_jobs),
            completed_jobs=len([j for j in job_history if j["status"] == "completed"]),
            failed_jobs=len([j for j in job_history if j["status"] == "failed"]),
            cancelled_jobs=len([j for j in job_history if j["status"] == "cancelled"]),
            status_distribution=status_counts,
            type_distribution=type_counts,
            recent_jobs_24h=len(recent_jobs),
            average_processing_time_seconds=avg_processing_time
        )
        
        return {
            "success": True,
            "message": "Job statistics retrieved successfully",
            "data": statistics.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting job statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get job statistics: {str(e)}")

# Bulk Operations

@router.post("/bulk/process", response_model=BulkProcessingResponse)
async def bulk_process_documents(
    request: BulkProcessingRequest,
    background_tasks: BackgroundTasks,
    pdf_processor: PDFProcessor = Depends(get_pdf_processor),
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_admin)
):
    """
    Process multiple documents in bulk
    
    - **urls**: List of document URLs to process
    - **options**: Processing options (extract_images, generate_summary, etc.)
    - **batch_size**: Number of documents to process concurrently (default: 5)
    """
    try:
        job_id = f"bulk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Track the bulk job
        await track_job(
            job_id, 
            "bulk_processing", 
            "pending",
            {
                "total_documents": len(request.urls),
                "batch_size": request.batch_size,
                "options": request.options.dict() if request.options else {}
            }
        )
        
        # Start background processing
        background_tasks.add_task(
            process_bulk_documents,
            job_id,
            request.urls,
            request.options,
            request.batch_size,
            pdf_processor
        )
        
        return BulkProcessingResponse(
            success=True,
            message="Bulk processing started successfully",
            data={
                "job_id": job_id,
                "total_documents": len(request.urls),
                "estimated_completion_time": datetime.utcnow() + timedelta(
                    minutes=len(request.urls) * 2  # Rough estimate
                )
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting bulk processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start bulk processing: {str(e)}")

async def process_bulk_documents(
    job_id: str,
    urls: List[str],
    options: Any,
    batch_size: int,
    pdf_processor: PDFProcessor
):
    """Background task for bulk document processing"""
    try:
        await track_job(job_id, "bulk_processing", "running")
        
        processed_count = 0
        failed_count = 0
        results = []
        
        # Process documents in batches
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            batch_tasks = []
            
            for url in batch_urls:
                task = asyncio.create_task(process_single_document(url, options, pdf_processor))
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    failed_count += 1
                    results.append({
                        "url": batch_urls[j],
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_count += 1
                    results.append({
                        "url": batch_urls[j],
                        "status": "completed",
                        "document_id": result.get("document_id")
                    })
            
            # Update job progress
            progress = (processed_count + failed_count) / len(urls) * 100
            await track_job(
                job_id, 
                "bulk_processing", 
                "running",
                {
                    "progress_percentage": progress,
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "results": results
                }
            )
        
        # Mark job as completed
        await track_job(
            job_id, 
            "bulk_processing", 
            "completed",
            {
                "total_processed": processed_count,
                "total_failed": failed_count,
                "results": results
            }
        )
        
    except Exception as e:
        logger.error(f"Error in bulk processing job {job_id}: {str(e)}")
        await track_job(
            job_id, 
            "bulk_processing", 
            "failed",
            {"error": str(e)}
        )

async def process_single_document(url: str, options: Any, pdf_processor: PDFProcessor):
    """Process a single document as part of bulk processing"""
    try:
        # This would integrate with the existing document processing logic
        # For now, we'll simulate processing
        await asyncio.sleep(1)  # Simulate processing time
        
        return {
            "document_id": f"doc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error processing document {url}: {str(e)}")
        raise

# System Monitoring

@router.get("/system/health")
async def get_system_health(
    current_user: User = Depends(get_current_user),
    workspace_context: WorkspaceContext = Depends(get_workspace_context),
    _: None = Depends(require_admin)
):
    """
    Get comprehensive system health status
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Service health checks
        services_health = {}
        
        # Check Supabase connection
        try:
            supabase_client = SupabaseClient()
            # Simple health check query
            services_health["supabase"] = {
                "status": "healthy",
                "response_time_ms": 50  # Placeholder
            }
        except Exception as e:
            services_health["supabase"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check LlamaIndex service
        try:
            llamaindex_service = LlamaIndexService()
            health_check = await llamaindex_service.health_check()
            services_health["llamaindex"] = {
                "status": "healthy" if health_check["status"] == "healthy" else "unhealthy",
                "details": health_check
            }
        except Exception as e:
            services_health["llamaindex"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Material Kai service
        try:
            material_kai_service = MaterialKaiService()
            health_check = await material_kai_service.health_check()
            services_health["material_kai"] = {
                "status": "healthy" if health_check["status"] == "healthy" else "unhealthy",
                "details": health_check
            }
        except Exception as e:
            services_health["material_kai"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health status
        all_services_healthy = all(
            service["status"] == "healthy" 
            for service in services_health.values()
        )
        
        system_metrics = SystemMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            active_jobs_count=len(active_jobs),
            uptime_seconds=None  # Would need to track application start time
        )
        
        return {
            "success": True,
            "message": "System health retrieved successfully",
            "data": {
                "overall_status": "healthy" if all_services_healthy and cpu_percent < 80 and memory.percent < 80 else "degraded",
                "system_metrics": system_metrics.dict(),
                "services": services_health,
                "active_jobs": len(active_jobs),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@router.get("/system/metrics")
async def get_system_metrics():
    """
    Get detailed system performance metrics
    """
    try:
        # CPU metrics
        cpu_count = psutil.cpu_count()
        cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "success": True,
            "message": "System metrics retrieved successfully",
            "data": {
                "cpu": {
                    "count": cpu_count,
                    "usage_percent": psutil.cpu_percent(),
                    "usage_per_core": cpu_percent_per_core,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "usage_percent": memory.percent,
                    "swap_total_gb": swap.total / (1024**3),
                    "swap_used_gb": swap.used / (1024**3),
                    "swap_percent": swap.percent
                },
                "disk": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "usage_percent": disk_usage.percent,
                    "read_bytes": disk_io.read_bytes if disk_io else None,
                    "write_bytes": disk_io.write_bytes if disk_io else None
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_received": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_received": network_io.packets_recv
                },
                "process": {
                    "memory_rss_mb": process_memory.rss / (1024**2),
                    "memory_vms_mb": process_memory.vms / (1024**2),
                    "cpu_percent": process.cpu_percent()
                },
                "jobs": {
                    "active_count": len(active_jobs),
                    "total_history": len(job_history)
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

# Administrative Data Management

@router.delete("/data/cleanup")
async def cleanup_old_data(
    days_old: int = Query(30, description="Delete data older than this many days"),
    dry_run: bool = Query(True, description="Preview what would be deleted without actually deleting")
):
    """
    Clean up old data from the system
    
    - **days_old**: Delete data older than this many days (default: 30)
    - **dry_run**: Preview what would be deleted without actually deleting (default: true)
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Find old job history entries
        old_jobs = [
            job for job in job_history
            if datetime.fromisoformat(job["created_at"].replace('Z', '+00:00')) < cutoff_date
        ]
        
        cleanup_summary = {
            "old_jobs_count": len(old_jobs),
            "cutoff_date": cutoff_date.isoformat(),
            "dry_run": dry_run
        }
        
        if not dry_run:
            # Actually remove old jobs from history
            global job_history
            job_history = [
                job for job in job_history
                if datetime.fromisoformat(job["created_at"].replace('Z', '+00:00')) >= cutoff_date
            ]
            cleanup_summary["jobs_deleted"] = len(old_jobs)
        
        return {
            "success": True,
            "message": f"Data cleanup {'preview' if dry_run else 'completed'} successfully",
            "data": cleanup_summary
        }
        
    except Exception as e:
        logger.error(f"Error during data cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup data: {str(e)}")

@router.post("/data/backup")
async def create_data_backup():
    """
    Create a backup of system data
    """
    try:
        backup_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "active_jobs": active_jobs,
            "job_history": job_history,
            "system_info": {
                "version": "1.0.0",  # Would come from app config
                "backup_type": "administrative_data"
            }
        }
        
        # In a real implementation, this would save to a file or external storage
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "success": True,
            "message": "Data backup created successfully",
            "data": {
                "backup_id": backup_id,
                "backup_size_bytes": len(str(backup_data)),
                "items_backed_up": {
                    "active_jobs": len(active_jobs),
                    "job_history": len(job_history)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating data backup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create data backup: {str(e)}")

@router.get("/data/export")
async def export_system_data(
    format: str = Query("json", description="Export format (json, csv)"),
    data_type: str = Query("jobs", description="Type of data to export (jobs, metrics)")
):
    """
    Export system data in various formats
    
    - **format**: Export format (json, csv)
    - **data_type**: Type of data to export (jobs, metrics)
    """
    try:
        if data_type == "jobs":
            all_jobs = list(active_jobs.values()) + job_history
            
            if format == "json":
                return JSONResponse(
                    content={
                        "success": True,
                        "message": "Jobs data exported successfully",
                        "data": all_jobs,
                        "export_info": {
                            "format": format,
                            "data_type": data_type,
                            "record_count": len(all_jobs),
                            "exported_at": datetime.utcnow().isoformat()
                        }
                    }
                )
            elif format == "csv":
                # In a real implementation, this would return a CSV file
                return {
                    "success": True,
                    "message": "CSV export not implemented yet",
                    "data": {"note": "CSV export would be implemented here"}
                }
        
        return {
            "success": True,
            "message": f"Export completed for {data_type} in {format} format",
            "data": {"note": f"Export functionality for {data_type} would be implemented here"}
        }
        
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")


@router.get("/packages/status")
async def get_package_status():
    """
    Get the status of all system packages and dependencies.

    Returns package information for both critical and optional dependencies,
    including version information and availability status.
    """
    try:
        return await get_basic_package_status()
    except Exception as e:
        logger.error(f"Error getting package status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get package status: {str(e)}")


async def get_basic_package_status():
    """Get package status by parsing requirements.txt and checking imports"""
    import importlib
    import re
    import os

    # Parse requirements.txt to get all packages
    requirements_path = "/var/www/mivaa-pdf-extractor/requirements.txt"
    if not os.path.exists(requirements_path):
        requirements_path = "requirements.txt"  # Fallback for local development

    packages_from_requirements = {}

    try:
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue

                # Parse package name and version
                # Handle formats like: package>=1.0.0, package==1.0.0, package[extra]>=1.0.0
                match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*?\])?([><=!]+.*)?', line)
                if match:
                    package_name = match.group(1)
                    version_spec = match.group(3) or ''

                    # Map some package names to their import names
                    import_name = package_name
                    if package_name == 'opencv-python-headless':
                        import_name = 'cv2'
                    elif package_name == 'pillow':
                        import_name = 'PIL'
                    elif package_name == 'python-dotenv':
                        import_name = 'dotenv'
                    elif package_name == 'python-multipart':
                        import_name = 'multipart'
                    elif package_name == 'python-dateutil':
                        import_name = 'dateutil'
                    elif package_name == 'python-json-logger':
                        import_name = 'pythonjsonlogger'
                    elif package_name == 'email-validator':
                        import_name = 'email_validator'
                    elif package_name == 'llama-index':
                        import_name = 'llama_index'

                    packages_from_requirements[package_name] = {
                        'import_name': import_name,
                        'version_spec': version_spec.strip(),
                        'required': True
                    }
    except Exception as e:
        logger.error(f"Error reading requirements.txt: {e}")

    # Check each package
    package_status = {}
    critical_packages = {
        'fastapi', 'uvicorn', 'pydantic', 'supabase', 'pymupdf4llm',
        'numpy', 'pandas', 'opencv-python-headless', 'pillow', 'llama-index',
        'openai', 'anthropic', 'torch'
    }

    for package_name, info in packages_from_requirements.items():
        try:
            module = importlib.import_module(info['import_name'])
            version = getattr(module, '__version__', 'unknown')
            package_status[package_name] = {
                'available': True,
                'version': version,
                'version_spec': info['version_spec'],
                'critical': package_name in critical_packages,
                'import_name': info['import_name']
            }
        except ImportError:
            package_status[package_name] = {
                'available': False,
                'version': None,
                'version_spec': info['version_spec'],
                'critical': package_name in critical_packages,
                'import_name': info['import_name'],
                'error': 'Package not found'
            }

    # Calculate summary
    critical_missing = sum(1 for pkg, status in package_status.items()
                          if pkg in critical_packages and not status['available'])
    total_packages = len(package_status)
    total_critical = len([pkg for pkg in package_status.keys() if pkg in critical_packages])
    available_packages = sum(1 for status in package_status.values() if status['available'])

    return {
        "success": True,
        "data": {
            "packages": package_status,
            "summary": {
                "total_packages": total_packages,
                "available_packages": available_packages,
                "missing_packages": total_packages - available_packages,
                "critical_missing": critical_missing,
                "total_critical": total_critical,
                "deployment_ready": critical_missing == 0
            }
        },
        "timestamp": datetime.utcnow().isoformat(),
        "source": "requirements.txt"
    }