"""
Query Performance Metrics and Monitoring

Tracks database query performance and logs slow queries.
Provides metrics for monitoring and alerting.
"""

import logging
import time
from functools import wraps
from typing import Callable, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryMetrics:
    """
    Tracks query performance metrics.
    
    Features:
    - Query execution time tracking
    - Slow query detection and logging
    - Per-table metrics
    - Aggregated statistics
    """
    
    def __init__(self, slow_query_threshold_ms: float = 1000.0):
        """
        Initialize query metrics.
        
        Args:
            slow_query_threshold_ms: Threshold for slow query alerts (milliseconds)
        """
        self.slow_query_threshold = slow_query_threshold_ms
        
        # Metrics storage
        self.query_count = 0
        self.slow_query_count = 0
        self.total_query_time_ms = 0.0
        self.max_query_time_ms = 0.0
        self.min_query_time_ms = float('inf')
        
        # Per-table metrics
        self.table_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time_ms': 0.0,
            'max_time_ms': 0.0,
            'slow_count': 0
        })
        
        # Recent slow queries (keep last 10)
        self.slow_queries: list[Dict[str, Any]] = []
        self.max_slow_queries = 10
        
        logger.info(f"QueryMetrics initialized (slow_query_threshold={slow_query_threshold_ms}ms)")
    
    def track_query(
        self,
        table_name: str,
        operation: str,
        execution_time_ms: float,
        query_details: Optional[str] = None
    ):
        """
        Track a query execution.
        
        Args:
            table_name: Name of the table queried
            operation: Operation type (select, insert, update, delete)
            execution_time_ms: Query execution time in milliseconds
            query_details: Optional query details for logging
        """
        # Update global metrics
        self.query_count += 1
        self.total_query_time_ms += execution_time_ms
        self.max_query_time_ms = max(self.max_query_time_ms, execution_time_ms)
        self.min_query_time_ms = min(self.min_query_time_ms, execution_time_ms)
        
        # Update table metrics
        table_stats = self.table_metrics[table_name]
        table_stats['count'] += 1
        table_stats['total_time_ms'] += execution_time_ms
        table_stats['max_time_ms'] = max(table_stats['max_time_ms'], execution_time_ms)
        
        # Check for slow query
        if execution_time_ms > self.slow_query_threshold:
            self.slow_query_count += 1
            table_stats['slow_count'] += 1
            
            # Log slow query
            slow_query_info = {
                'timestamp': datetime.utcnow().isoformat(),
                'table': table_name,
                'operation': operation,
                'execution_time_ms': execution_time_ms,
                'query_details': query_details
            }
            
            self.slow_queries.append(slow_query_info)
            
            # Keep only recent slow queries
            if len(self.slow_queries) > self.max_slow_queries:
                self.slow_queries.pop(0)
            
            logger.warning(
                f"🐌 Slow query detected: {table_name}.{operation} "
                f"took {execution_time_ms:.2f}ms (threshold: {self.slow_query_threshold}ms)"
                + (f" - {query_details}" if query_details else "")
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_query_time = (
            self.total_query_time_ms / self.query_count
            if self.query_count > 0
            else 0.0
        )
        
        return {
            'total_queries': self.query_count,
            'slow_queries': self.slow_query_count,
            'slow_query_percentage': (
                (self.slow_query_count / self.query_count * 100)
                if self.query_count > 0
                else 0.0
            ),
            'avg_query_time_ms': round(avg_query_time, 2),
            'max_query_time_ms': round(self.max_query_time_ms, 2),
            'min_query_time_ms': round(self.min_query_time_ms, 2) if self.min_query_time_ms != float('inf') else 0.0,
            'slow_query_threshold_ms': self.slow_query_threshold,
            'table_metrics': {
                table: {
                    'count': stats['count'],
                    'avg_time_ms': round(stats['total_time_ms'] / stats['count'], 2),
                    'max_time_ms': round(stats['max_time_ms'], 2),
                    'slow_count': stats['slow_count']
                }
                for table, stats in self.table_metrics.items()
            },
            'recent_slow_queries': self.slow_queries[-5:]  # Last 5 slow queries
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.query_count = 0
        self.slow_query_count = 0
        self.total_query_time_ms = 0.0
        self.max_query_time_ms = 0.0
        self.min_query_time_ms = float('inf')
        self.table_metrics.clear()
        self.slow_queries.clear()
        logger.info("Query metrics reset")


# Global instance
query_metrics = QueryMetrics(slow_query_threshold_ms=1000.0)


def track_query_performance(table_name: str, operation: str = "query"):
    """
    Decorator to track query performance.
    
    Example:
        @track_query_performance("background_jobs", "select")
        async def get_stuck_jobs():
            return await supabase.table("background_jobs").select("*").execute()
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time_ms = (time.time() - start_time) * 1000
                query_metrics.track_query(table_name, operation, execution_time_ms)
                return result
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                query_metrics.track_query(
                    table_name,
                    operation,
                    execution_time_ms,
                    query_details=f"ERROR: {str(e)}"
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.time() - start_time) * 1000
                query_metrics.track_query(table_name, operation, execution_time_ms)
                return result
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                query_metrics.track_query(
                    table_name,
                    operation,
                    execution_time_ms,
                    query_details=f"ERROR: {str(e)}"
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

