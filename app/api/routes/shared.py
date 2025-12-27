"""
Shared utilities and dependencies for RAG routes

This module contains common helpers, dependencies, and storage
used across multiple route modules.
"""

import logging
import asyncio
from typing import Dict, Any
from fastapi import Depends


from app.services.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)

# ============================================================================
# Job Storage (In-Memory Cache)
# ============================================================================
job_storage: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Background Task Helper for Async Functions
# ============================================================================
def run_async_in_background(async_func):
    """
    Wrapper to run async functions in FastAPI BackgroundTasks.

    FastAPI's BackgroundTasks.add_task() expects synchronous functions.
    When an async function is passed, it doesn't execute properly because
    there's no event loop in the background thread.

    This wrapper creates a new event loop specifically for the background task,
    allowing async functions to run correctly in background threads.

    Usage:
        background_tasks.add_task(
            run_async_in_background(process_document_with_discovery),
            job_id=job_id,
            document_id=document_id,
            ...
        )

    Args:
        async_func: The async function to wrap

    Returns:
        A synchronous wrapper function that can be used with BackgroundTasks
    """
    def wrapper(*args, **kwargs):
        logger.info(f"🚀 Background task wrapper started for {async_func.__name__}")
        # Create a new event loop for this background task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(f"▶️  Executing async function {async_func.__name__} in background")
            # Run the async function to completion
            loop.run_until_complete(async_func(*args, **kwargs))
            logger.info(f"✅ Background task {async_func.__name__} completed successfully")
        except Exception as e:
            logger.error(f"❌ Background task {async_func.__name__} failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up the event loop
            loop.close()
            logger.info(f"🔚 Background task wrapper finished for {async_func.__name__}")
    return wrapper


# ============================================================================
# Service Dependencies (Using centralized dependencies)
# ============================================================================
# REMOVED: Local get_rag_service - now using centralized dependency from app.dependencies
# Import centralized dependencies instead
from app.dependencies import get_rag_service, get_supabase_client, get_pdf_processor


async def get_embedding_service() -> RealEmbeddingsService:
    """Get embedding service instance."""
    return RealEmbeddingsService()

