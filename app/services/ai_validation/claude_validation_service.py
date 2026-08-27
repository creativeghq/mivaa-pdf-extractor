"""
Claude Validation Service

Provides async background validation for images with low quality scores, to
enhance image analysis quality.

ARCHITECTURE (vision is Anthropic-only):
- Initial per-image analysis runs on Claude Opus during sync processing
- Claude validation queued for low-quality images (score < 0.7)
- Updates image records with enhanced analysis

TENANCY (audit #23 M10-1). Every method here takes an explicit `workspace_id` and
binds it into every read and write. MIVAA connects as service role, so RLS is not a
backstop: without the predicate a caller holding any valid `image_id` could read
another tenant's `document_images` row in full, spend Claude analysing it, and
overwrite its validation metadata. `document_images` is the SILVER layer — a
corrupted verdict there propagates into embeddings, search ranking and product
association, and no drift check would surface it.

Two ids are never trusted to agree: `queue_image_for_validation` resolves the image
inside the workspace and rejects a `document_id` that is not the one the image
actually belongs to, rather than storing the pair the caller asserted.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from app.services.images.real_image_analysis_service import RealImageAnalysisService
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class ClaudeValidationService:
    """
    Service for async Claude validation of low-quality images.
    
    Workflow:
    1. Images are analyzed with Claude during sync processing
    2. Images with quality score < 0.7 are queued for Claude validation
    3. This service processes the queue before product creation
    4. Updates image records with Claude validation results
    """
    
    def __init__(self, workspace_id: Optional[str] = None):
        self.logger = logger
        self.supabase = get_supabase_client()
        self.workspace_id = workspace_id
        self.image_analysis_service = RealImageAnalysisService(workspace_id=workspace_id)

    def _require_workspace(self, workspace_id: Optional[str]) -> str:
        """Resolve the workspace for this call, fail-closed.

        Falls back to the one the service was constructed with. Missing means
        missing: an unscoped query here reads and writes every tenant's images, so
        there is no safe default to guess.
        """
        ws = workspace_id or self.workspace_id
        if not ws or str(ws) == "None":
            raise ValueError(
                "claude_validation: workspace_id is required — an unscoped queue "
                "read/write spans every tenant (audit #23 M10-1)"
            )
        return str(ws)

    async def queue_image_for_validation(
        self,
        image_id: str,
        document_id: str,
        quality_score: float,
        priority: int = 5,
        workspace_id: Optional[str] = None,
    ) -> str:
        """
        Queue an image for Claude validation.

        Args:
            image_id: Image ID to validate
            document_id: Parent document ID
            quality_score: Quality score from the initial vision analysis
            priority: Job priority (1-10, lower = higher priority)
            workspace_id: Owning workspace. Required — see `_require_workspace`.

        Returns:
            Validation job ID

        Raises:
            ValueError: no workspace, image not in that workspace, or the image
                belongs to a different document than the caller claimed.
        """
        ws = self._require_workspace(workspace_id)
        try:
            # Resolve the image INSIDE the workspace, and make the two ids agree.
            # Storing the caller's asserted pair is the mass-assignment shape:
            # nothing downstream re-checks it.
            owner = self.supabase.client.table('document_images')                 .select('id, document_id, workspace_id')                 .eq('id', image_id)                 .eq('workspace_id', ws)                 .limit(1)                 .execute()
            owner_row = (owner.data or [None])[0]
            if not owner_row:
                raise ValueError(f"Image {image_id} not found in workspace {ws}")
            if str(owner_row.get('document_id')) != str(document_id):
                raise ValueError(
                    f"Image {image_id} belongs to document {owner_row.get('document_id')}, "
                    f"not {document_id}"
                )

            job_id = str(uuid.uuid4())

            job_data = {
                'id': job_id,
                'workspace_id': ws,
                'document_id': document_id,
                'image_id': image_id,
                'job_type': 'claude_validation',
                'status': 'pending',
                'priority': priority,
                'metadata': {
                    'quality_score': quality_score,
                    'queued_at': datetime.utcnow().isoformat()
                },
                'retry_count': 0,
                'max_retries': 3,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Insert into validation queue table
            self.supabase.client.table('claude_validation_queue').insert(job_data).execute()
            
            self.logger.info(
                f"✅ Queued image {image_id} for Claude validation "
                f"(quality score: {quality_score:.2f})"
            )
            
            return job_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to queue image for validation: {e}")
            raise
    
    async def process_validation_queue(
        self,
        document_id: str,
        job_id: Optional[str] = None,
        batch_size: int = 10,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process all pending Claude validation jobs for a document.
        
        This runs BEFORE product creation to ensure all images have
        high-quality analysis.
        
        Args:
            document_id: Document ID to process validations for
            job_id: Optional parent job ID for progress tracking
            batch_size: Number of images to process in parallel
            workspace_id: Owning workspace. Required - see `_require_workspace`.

        Returns:
            Processing statistics
        """
        ws = self._require_workspace(workspace_id)
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🔍 Starting Claude validation queue processing for document {document_id}")
            
            # Get all pending validation jobs for this document
            response = self.supabase.client.table('claude_validation_queue')\
                .select('id, document_id, image_id, status, priority, metadata, retry_count, max_retries')\
                .eq('workspace_id', ws)\
                .eq('document_id', document_id)\
                .eq('status', 'pending')\
                .order('priority')\
                .execute()
            
            pending_jobs = response.data if response.data else []
            total_jobs = len(pending_jobs)
            
            if total_jobs == 0:
                self.logger.info("✅ No images need Claude validation")
                return {
                    'total_images': 0,
                    'validated': 0,
                    'failed': 0,
                    'skipped': 0,
                    'processing_time_ms': 0
                }
            
            self.logger.info(f"📊 Found {total_jobs} images needing Claude validation")
            
            # Process in batches to avoid overwhelming the API
            validated_count = 0
            failed_count = 0
            
            for i in range(0, total_jobs, batch_size):
                batch = pending_jobs[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self._validate_single_image(job, ws) for job in batch],
                    return_exceptions=True
                )
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        failed_count += 1
                    elif result:
                        validated_count += 1
                    else:
                        failed_count += 1
                
                # Update progress if job_id provided
                if job_id:
                    progress = int((i + len(batch)) / total_jobs * 100)
                    self.logger.info(f"📊 Claude validation progress: {progress}%")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            stats = {
                'total_images': total_jobs,
                'validated': validated_count,
                'failed': failed_count,
                'skipped': 0,
                'processing_time_ms': processing_time
            }
            
            self.logger.info(
                f"✅ Claude validation complete: {validated_count}/{total_jobs} validated "
                f"in {processing_time:.0f}ms"
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Claude validation queue processing failed: {e}")
            raise
    
    async def _validate_single_image(self, job: Dict[str, Any], workspace_id: str) -> bool:
        """
        Validate a single image with Claude.

        Args:
            job: Validation job data
            workspace_id: Owning workspace - carried into every read and write.

        Returns:
            True if successful, False otherwise
        """
        image_id = job['image_id']
        job_id = job['id']
        ws = self._require_workspace(workspace_id)

        try:
            # Update job status to processing
            self.supabase.client.table('claude_validation_queue')\
                .update({'status': 'processing', 'updated_at': datetime.utcnow().isoformat()})\
                .eq('id', job_id)\
                .eq('workspace_id', ws)\
                .execute()
            
            # Get image data from database
            image_response = self.supabase.client.table('document_images')\
                .select('id, image_url, metadata, workspace_id')\
                .eq('id', image_id)\
                .eq('workspace_id', ws)\
                .limit(1)\
                .execute()

            image_data = (image_response.data or [None])[0]
            if not image_data:
                raise ValueError(f"Image {image_id} not found in workspace {ws}")
            image_url = image_data.get('image_url')
            
            if not image_url:
                raise ValueError(f"Image {image_id} has no URL")
            
            # Run Claude validation
            claude_result = await self.image_analysis_service._analyze_with_claude(
                image_url=image_url,
                context={'validation_mode': True},
                job_id=job_id
            )
            
            # Update image record with Claude validation
            update_data = {
                'metadata': {
                    **image_data.get('metadata', {}),
                    'claude_validation': claude_result,
                    'claude_validated_at': datetime.utcnow().isoformat()
                },
                'updated_at': datetime.utcnow().isoformat()
            }
            
            self.supabase.client.table('document_images')\
                .update(update_data)\
                .eq('id', image_id)\
                .eq('workspace_id', ws)\
                .execute()
            
            # Mark validation job as complete
            self.supabase.client.table('claude_validation_queue')\
                .update({
                    'status': 'completed',
                    'metadata': {
                        **job.get('metadata', {}),
                        'completed_at': datetime.utcnow().isoformat(),
                        'claude_confidence': claude_result.get('confidence', 0.0)
                    },
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('id', job_id)\
                .eq('workspace_id', ws)\
                .execute()
            
            self.logger.info(f"✅ Claude validation complete for image {image_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Claude validation failed for image {image_id}: {e}")
            
            # Update job status to failed
            retry_count = job.get('retry_count', 0) + 1
            max_retries = job.get('max_retries', 3)
            
            if retry_count >= max_retries:
                status = 'failed'
            else:
                status = 'pending'  # Will retry
            
            self.supabase.client.table('claude_validation_queue')\
                .update({
                    'status': status,
                    'retry_count': retry_count,
                    'metadata': {
                        **job.get('metadata', {}),
                        'last_error': str(e),
                        'last_attempt_at': datetime.utcnow().isoformat()
                    },
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('id', job_id)\
                .eq('workspace_id', ws)\
                .execute()
            
            return False


