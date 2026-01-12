"""
Orchestration utilities for background task management.
"""
import asyncio
from typing import Callable, Any
from functools import wraps


def run_async_in_background(async_func: Callable) -> Callable:
    """
    Decorator to run an async function in the background without blocking.
    
    This is used for fire-and-forget async operations that should not block
    the current request/response cycle.
    
    Args:
        async_func: The async function to run in background
        
    Returns:
        A wrapper function that schedules the async function as a background task
        
    Example:
        @run_async_in_background
        async def process_document(doc_id: str):
            # Long-running processing
            pass
            
        # Call it like a normal function - it will run in background
        process_document(doc_id="123")
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        """Wrapper that creates and schedules the background task."""
        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Create the coroutine
        coro = async_func(*args, **kwargs)
        
        # Schedule it as a background task
        task = asyncio.create_task(coro)
        
        # Add error handling to prevent unhandled exceptions
        def handle_task_result(task):
            try:
                task.result()
            except Exception as e:
                # Log the error but don't raise it
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Background task failed: {e}", exc_info=True)
        
        task.add_done_callback(handle_task_result)
        
        return task
    
    return wrapper

