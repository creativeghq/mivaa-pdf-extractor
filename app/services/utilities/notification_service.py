"""
Notification Service for Python Backend
Sends notifications via Supabase Edge Function
"""

import logging
from typing import Optional, Dict, Any, List
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending notifications through the notification system."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
    
    async def send_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Send a notification through the notification system.
        
        Args:
            user_id: User ID to send notification to
            notification_type: Type of notification (e.g., 'job_completed', 'job_failed')
            title: Notification title
            message: Notification message
            data: Additional data to include
            channels: Optional list of channels to use (email, push, webhook)
        
        Returns:
            Result dictionary with success status
        """
        try:
            # Write to `user_notifications` — the table the bell UI (NotificationsPanel)
            # actually reads. The old code inserted into a dead `notifications` table that
            # nothing reads, so every MIVAA job-lifecycle notification was silently lost
            # (audit #217 H10). Mapping: message → body, data.action_url → action_url,
            # full data payload → metadata.
            payload = data or {}
            notification_data = {
                'user_id': user_id,
                'type': notification_type,
                'title': title,
                'body': message,
                'action_url': payload.get('action_url'),
                'is_read': False,
                'metadata': payload,
            }

            result = self.supabase.client.table('user_notifications').insert(notification_data).execute()

            logger.info(f"✅ Notification saved for user {user_id}: {title}")
            return {'success': True, 'notification_id': result.data[0]['id']}
                
        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")
            return {'success': False, 'error': str(e)}
    
    async def notify_job_started(
        self,
        user_id: str,
        job_id: str,
        job_type: str
    ) -> None:
        """Notify user that a job has started."""
        job_label = self._format_job_type(job_type)
        await self.send_notification(
            user_id=user_id,
            notification_type='job_started',
            title=f"{job_label} Started",
            message=f"Your {job_label.lower()} job has started processing.",
            data={
                'job_id': job_id,
                'job_type': job_type,
                'action_url': f"/admin/async-jobs?job={job_id}"
            },
            channels=['push']
        )
    
    async def notify_job_completed(
        self,
        user_id: str,
        job_id: str,
        job_type: str,
        duration: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify user that a job has completed successfully."""
        job_label = self._format_job_type(job_type)
        
        message = f"Completed successfully"
        if duration:
            message += f" in {duration}"
        if stats:
            stats_str = self._format_stats(stats)
            if stats_str:
                message += f"\n\n{stats_str}"
        
        await self.send_notification(
            user_id=user_id,
            notification_type='job_completed',
            title=f"✅ {job_label} Completed",
            message=message,
            data={
                'job_id': job_id,
                'job_type': job_type,
                'duration': duration,
                'stats': stats,
                'action_url': f"/admin/async-jobs?job={job_id}"
            }
        )
    
    async def notify_job_failed(
        self,
        user_id: str,
        job_id: str,
        job_type: str,
        error: Optional[str] = None
    ) -> None:
        """Notify user that a job has failed."""
        job_label = self._format_job_type(job_type)
        
        await self.send_notification(
            user_id=user_id,
            notification_type='job_failed',
            title=f"❌ {job_label} Failed",
            message=error or "The job encountered an error and could not complete.",
            data={
                'job_id': job_id,
                'job_type': job_type,
                'error': error,
                'action_url': f"/admin/async-jobs?job={job_id}"
            }
        )
    
    def _format_job_type(self, job_type: str) -> str:
        """Format job type for display."""
        labels = {
            'pdf_processing': 'PDF Processing',
            'web_scraping': 'Web Scraping',
            'product_discovery_upload': 'Product Discovery',
            'image_embedding_regeneration': 'Image Embedding Regeneration',
            'xml_import': 'XML Import'
        }
        return labels.get(job_type, job_type.replace('_', ' ').title())
    
    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """Format job statistics for display."""
        parts = []
        if 'images_processed' in stats:
            parts.append(f"{stats['images_processed']} images")
        if 'embeddings_generated' in stats:
            parts.append(f"{stats['embeddings_generated']} embeddings")
        if 'chunks_created' in stats:
            parts.append(f"{stats['chunks_created']} chunks")
        if 'products_created' in stats:
            parts.append(f"{stats['products_created']} products")
        return ', '.join(parts) if parts else ''


# Singleton instance
_notification_service = None

def get_notification_service() -> NotificationService:
    """Get the notification service singleton."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service

