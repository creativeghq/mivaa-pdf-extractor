"""
Monitoring API Routes

Provides real-time monitoring of:
- Supabase storage usage
- Database size and row counts
- Resource limits and warnings
- System health status
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from datetime import datetime

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# Tables to monitor
TABLES_TO_MONITOR = [
    'documents',
    'document_chunks',
    'document_images',
    'embeddings',
    'products',
    'materials_catalog',
    'background_jobs'
]

# Storage buckets to monitor
BUCKETS_TO_MONITOR = [
    'pdf-tiles',
    'pdf-documents',
    'material-images'
]

# Supabase Free Tier Limits
RESOURCE_LIMITS = {
    'storage_gb': 1.0,
    'database_size_gb': 0.5,
    'bandwidth_gb_month': 5.0,
    'api_requests_month': 500000
}


async def get_table_row_count(table_name: str) -> int:
    """Get row count for a table."""
    try:
        supabase = get_supabase_client()
        result = supabase.client.table(table_name).select('*', count='exact').limit(0).execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"Failed to get count for {table_name}: {e}")
        return 0


async def get_bucket_stats(bucket_name: str) -> Dict[str, Any]:
    """Get statistics for a storage bucket."""
    try:
        supabase = get_supabase_client()

        # List all files in bucket
        files = supabase.client.storage.from_(bucket_name).list()

        # Handle None or empty response
        if not files:
            files = []

        total_size = 0
        file_count = 0

        for file in files:
            # Skip if file is None or is a folder
            if not file or not isinstance(file, dict):
                continue
            if file.get('name', '').endswith('/'):
                continue

            # Get file size from metadata
            metadata = file.get('metadata', {})
            if metadata and isinstance(metadata, dict):
                total_size += metadata.get('size', 0)
            file_count += 1

        return {
            'bucket': bucket_name,
            'files': file_count,
            'size_bytes': total_size,
            'size_mb': round(total_size / 1024 / 1024, 2),
            'size_gb': round(total_size / 1024 / 1024 / 1024, 3)
        }
    except Exception as e:
        logger.error(f"Failed to get stats for bucket {bucket_name}: {e}")
        return {
            'bucket': bucket_name,
            'files': 0,
            'size_bytes': 0,
            'size_mb': 0,
            'size_gb': 0,
            'error': str(e)
        }


@router.get(
    "/supabase-status",
    response_model=Dict[str, Any],
    summary="Get comprehensive Supabase resource usage and health status",
    description="""
    Monitor all Supabase resources including database, storage, and usage limits.

    **Returns:**
    - **Database Statistics**: Row counts for all monitored tables
    - **Storage Statistics**: Size and file counts per bucket
    - **Resource Limits**: Free tier limits and current usage percentages
    - **Health Status**: Overall system health (healthy, notice, warning, critical)
    - **Warnings**: Alerts when approaching resource limits

    **Monitored Tables:**
    - documents, document_chunks, document_images
    - embeddings, products, materials_catalog
    - background_jobs

    **Monitored Buckets:**
    - pdf-tiles, pdf-documents, material-images

    **Health Status Levels:**
    - `healthy`: <50% usage
    - `notice`: 50-80% usage
    - `warning`: 80-90% usage
    - `critical`: >90% usage (upgrade needed)

    **Example Response:**
    ```json
    {
      "success": true,
      "timestamp": "2025-11-08T16:30:00Z",
      "health_status": "warning",
      "database": {
        "tables": [
          {"table": "documents", "rows": 150},
          {"table": "document_chunks", "rows": 5000}
        ],
        "total_rows": 5150
      },
      "storage": {
        "buckets": [
          {
            "bucket": "pdf-documents",
            "files": 150,
            "size_gb": 0.45
          }
        ],
        "total_files": 150,
        "total_size_gb": 0.85
      },
      "limits": {
        "storage_gb": 1.0,
        "database_size_gb": 0.5
      },
      "usage": {
        "storage_percent": 85.0,
        "storage_remaining_gb": 0.15
      },
      "warnings": [
        {
          "type": "warning",
          "resource": "storage",
          "message": "Storage usage is at 85.0% - Consider upgrading soon.",
          "usage_percent": 85.0
        }
      ]
    }
    ```

    **Use Cases:**
    - Admin dashboard monitoring
    - Automated alerts for resource limits
    - Capacity planning
    - Cost optimization

    **Performance:**
    - Typical: 1-2 seconds (queries all tables and buckets)
    - Recommended: Cache for 5-10 minutes

    **Rate Limits:**
    - 10 requests/minute

    **Error Codes:**
    - 200: Success
    - 500: Failed to retrieve status (check logs)
    """,
    tags=["monitoring"],
    responses={
        200: {"description": "Comprehensive resource status"},
        500: {"description": "Failed to retrieve status"}
    }
)
async def get_supabase_status() -> Dict[str, Any]:
    """
    Get current Supabase resource usage and status.

    Returns:
        - Database statistics (row counts per table)
        - Storage statistics (size per bucket)
        - Resource limits and usage percentages
        - Warnings if approaching limits
    """
    try:
        logger.info("Fetching Supabase status...")
        
        # Get database statistics
        database_stats = []
        total_rows = 0
        
        for table_name in TABLES_TO_MONITOR:
            row_count = await get_table_row_count(table_name)
            database_stats.append({
                'table': table_name,
                'rows': row_count
            })
            total_rows += row_count
        
        # Get storage statistics
        storage_stats = []
        total_storage_bytes = 0
        total_files = 0
        
        for bucket_name in BUCKETS_TO_MONITOR:
            bucket_stats = await get_bucket_stats(bucket_name)
            storage_stats.append(bucket_stats)
            total_storage_bytes += bucket_stats['size_bytes']
            total_files += bucket_stats['files']
        
        # Calculate usage percentages
        storage_gb = total_storage_bytes / 1024 / 1024 / 1024
        storage_usage_percent = (storage_gb / RESOURCE_LIMITS['storage_gb']) * 100
        
        # Generate warnings
        warnings = []
        if storage_usage_percent > 90:
            warnings.append({
                'type': 'critical',
                'resource': 'storage',
                'message': f'Storage usage is at {storage_usage_percent:.1f}% - CRITICAL! Upgrade needed.',
                'usage_percent': storage_usage_percent
            })
        elif storage_usage_percent > 80:
            warnings.append({
                'type': 'warning',
                'resource': 'storage',
                'message': f'Storage usage is at {storage_usage_percent:.1f}% - Consider upgrading soon.',
                'usage_percent': storage_usage_percent
            })
        elif storage_usage_percent > 50:
            warnings.append({
                'type': 'notice',
                'resource': 'storage',
                'message': f'Storage usage is at {storage_usage_percent:.1f}%',
                'usage_percent': storage_usage_percent
            })
        
        # Determine overall health status
        health_status = 'healthy'
        if storage_usage_percent > 90:
            health_status = 'critical'
        elif storage_usage_percent > 80:
            health_status = 'warning'
        elif storage_usage_percent > 50:
            health_status = 'notice'
        
        return {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'health_status': health_status,
            'database': {
                'tables': database_stats,
                'total_rows': total_rows
            },
            'storage': {
                'buckets': storage_stats,
                'total_files': total_files,
                'total_size_bytes': total_storage_bytes,
                'total_size_mb': round(total_storage_bytes / 1024 / 1024, 2),
                'total_size_gb': round(storage_gb, 3)
            },
            'limits': RESOURCE_LIMITS,
            'usage': {
                'storage_percent': round(storage_usage_percent, 2),
                'storage_remaining_gb': round(RESOURCE_LIMITS['storage_gb'] - storage_gb, 3)
            },
            'warnings': warnings,
            'can_upload': storage_usage_percent < 95  # Block uploads if > 95%
        }
        
    except Exception as e:
        logger.error(f"Failed to get Supabase status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Supabase status: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Quick health check for monitoring systems.
    
    Returns:
        - Overall system health
        - Database connectivity
        - Storage connectivity
    """
    try:
        checks = []
        overall_healthy = True
        
        # Check database connectivity
        try:
            await get_table_row_count('documents')
            checks.append({
                'check': 'database',
                'status': 'ok',
                'message': 'Database is accessible'
            })
        except Exception as e:
            checks.append({
                'check': 'database',
                'status': 'error',
                'message': f'Database error: {str(e)}'
            })
            overall_healthy = False
        
        # Check storage connectivity
        try:
            await get_bucket_stats('pdf-tiles')
            checks.append({
                'check': 'storage',
                'status': 'ok',
                'message': 'Storage is accessible'
            })
        except Exception as e:
            checks.append({
                'check': 'storage',
                'status': 'error',
                'message': f'Storage error: {str(e)}'
            })
            overall_healthy = False
        
        return {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'checks': checks
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/storage-estimate")
async def estimate_storage_for_upload(
    file_size_mb: float,
    estimated_images: int = 50
) -> Dict[str, Any]:
    """
    Estimate if there's enough storage for a PDF upload.
    
    Args:
        file_size_mb: Size of PDF file in MB
        estimated_images: Estimated number of images to extract
        
    Returns:
        - Whether upload is safe
        - Estimated storage needed
        - Current available storage
    """
    try:
        # Get current storage usage
        status = await get_supabase_status()
        current_usage_gb = status['storage']['total_size_gb']
        
        # Estimate storage needed
        # PDF file + extracted images (assume 200KB per image average)
        estimated_images_mb = estimated_images * 0.2
        total_estimated_mb = file_size_mb + estimated_images_mb
        total_estimated_gb = total_estimated_mb / 1024
        
        # Calculate if safe
        projected_usage_gb = current_usage_gb + total_estimated_gb
        projected_usage_percent = (projected_usage_gb / RESOURCE_LIMITS['storage_gb']) * 100
        
        is_safe = projected_usage_percent < 90
        
        return {
            'success': True,
            'is_safe': is_safe,
            'current_usage_gb': current_usage_gb,
            'estimated_additional_gb': round(total_estimated_gb, 3),
            'projected_usage_gb': round(projected_usage_gb, 3),
            'projected_usage_percent': round(projected_usage_percent, 2),
            'storage_limit_gb': RESOURCE_LIMITS['storage_gb'],
            'recommendation': 'safe' if is_safe else 'upgrade_needed',
            'message': 'Upload is safe' if is_safe else 'Storage limit will be exceeded - upgrade required'
        }
        
    except Exception as e:
        logger.error(f"Failed to estimate storage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to estimate storage: {str(e)}")


@router.get("/pdf/health", tags=["health"])
async def pdf_health():
    """
    PDF service health check endpoint.

    Returns:
        Health status of the PDF processing service
    """
    return {
        "status": "healthy",
        "service": "pdf-processor",
        "timestamp": datetime.now().isoformat()
    }

