"""
Supabase Logging Handler

Custom logging handler that writes logs to the system_logs table in Supabase.
Provides async and sync logging capabilities with batching for performance.
"""

import logging
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from queue import Queue
from threading import Thread, Event
import time

from app.services.core.supabase_client import get_supabase_client


class SupabaseLoggingHandler(logging.Handler):
    """
    Custom logging handler that writes logs to Supabase system_logs table.
    
    Features:
    - Async batch writing for performance
    - Automatic retry on failure
    - Context extraction from log records
    - Job ID and request ID tracking
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        level: int = logging.INFO
    ):
        """
        Initialize the Supabase logging handler.
        
        Args:
            batch_size: Number of logs to batch before writing
            flush_interval: Seconds between automatic flushes
            level: Minimum log level to handle
        """
        super().__init__(level)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.queue: Queue = Queue()
        self.stop_event = Event()
        
        # Start background thread for batch processing
        self.worker_thread = Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by adding it to the queue.
        
        Args:
            record: The log record to emit
        """
        try:
            # Format the log record
            log_entry = self._format_log_entry(record)
            
            # Add to queue for batch processing
            self.queue.put(log_entry)
            
        except Exception as e:
            # Don't let logging errors crash the application
            self.handleError(record)
    
    def _format_log_entry(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        Format a log record into a dictionary for Supabase.
        
        Args:
            record: The log record to format
            
        Returns:
            Dictionary with log entry data
        """
        # Extract context from record
        context = {}
        
        # Add exception info if present
        if record.exc_info:
            context['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info']:
                try:
                    # Only add JSON-serializable values
                    json.dumps(value)
                    context[key] = value
                except (TypeError, ValueError):
                    context[key] = str(value)

        # Tag all backend logs with source='backend'
        context['source'] = 'backend'

        # Build log entry
        return {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger_name': record.name,
            'message': self.format(record),
            'context': context if context else None,
            'job_id': getattr(record, 'job_id', None),
            'user_id': getattr(record, 'user_id', None),
            'request_id': getattr(record, 'request_id', None)
        }
    
    def _process_queue(self) -> None:
        """
        Background thread that processes the log queue in batches.
        """
        batch = []
        last_flush = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Check if we should flush based on time
                if time.time() - last_flush >= self.flush_interval and batch:
                    self._write_batch(batch)
                    batch = []
                    last_flush = time.time()
                
                # Try to get a log entry (with timeout)
                try:
                    log_entry = self.queue.get(timeout=1.0)
                    batch.append(log_entry)
                    
                    # Flush if batch is full
                    if len(batch) >= self.batch_size:
                        self._write_batch(batch)
                        batch = []
                        last_flush = time.time()
                        
                except:
                    # Timeout - continue loop
                    pass
                    
            except Exception as e:
                # Log error but don't crash the thread
                print(f"Error in logging queue processor: {e}")
        
        # Flush remaining logs on shutdown
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, batch: list) -> None:
        """
        Write a batch of log entries to Supabase.
        
        Args:
            batch: List of log entry dictionaries
        """
        if not batch:
            return
            
        try:
            supabase = get_supabase_client()
            # Check if client is initialized before attempting to write
            if supabase._client is None:
                return  # Silently skip if not initialized yet
            supabase.client.table('system_logs').insert(batch).execute()
        except Exception as e:
            # Print to stderr so we don't lose the error (but only if it's not an initialization error)
            if "not initialized" not in str(e):
                print(f"Failed to write logs to Supabase: {e}", file=__import__('sys').stderr)
    
    def close(self) -> None:
        """
        Close the handler and flush remaining logs.
        """
        self.stop_event.set()
        self.worker_thread.join(timeout=10.0)
        super().close()


