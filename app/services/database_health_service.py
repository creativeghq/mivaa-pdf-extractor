"""
Database Connection Pool Health Monitoring Service

Monitors Supabase connection pool health and provides metrics:
- Active connections
- Idle connections
- Connection wait times
- Query performance
- Connection errors

Provides health check endpoint for monitoring systems.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


@dataclass
class DatabaseHealthMetrics:
    """Database health metrics"""
    is_healthy: bool = True
    last_check: Optional[datetime] = None
    connection_test_ms: Optional[float] = None
    query_test_ms: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    uptime_seconds: float = 0.0
    
    # Query performance metrics
    slow_query_count: int = 0
    avg_query_time_ms: float = 0.0
    max_query_time_ms: float = 0.0
    
    # Connection metrics (if available from Supabase)
    active_connections: Optional[int] = None
    idle_connections: Optional[int] = None
    max_connections: Optional[int] = None


class DatabaseHealthService:
    """
    Service for monitoring database connection pool health.
    
    Features:
    - Periodic health checks
    - Connection pool monitoring
    - Query performance tracking
    - Automatic alerting on degradation
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 30,
        slow_query_threshold_ms: float = 1000.0,
        failure_threshold: int = 3
    ):
        """
        Initialize database health service.
        
        Args:
            check_interval_seconds: How often to check health
            slow_query_threshold_ms: Threshold for slow query alerts
            failure_threshold: Number of failures before marking unhealthy
        """
        self.supabase_client = get_supabase_client()
        self.check_interval = check_interval_seconds
        self.slow_query_threshold = slow_query_threshold_ms
        self.failure_threshold = failure_threshold
        
        self.metrics = DatabaseHealthMetrics()
        self.running = False
        self.start_time = time.time()
        
        # Query performance tracking
        self.query_times: list[float] = []
        self.max_query_history = 100
        
        logger.info(
            f"DatabaseHealthService initialized "
            f"(check_interval={check_interval_seconds}s, "
            f"slow_query_threshold={slow_query_threshold_ms}ms)"
        )
    
    async def start(self):
        """Start health monitoring loop"""
        if self.running:
            logger.warning("Health monitor already running")
            return
        
        self.running = True
        logger.info("🏥 Database health monitor started")
        
        try:
            while self.running:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("🛑 Database health monitor cancelled")
            self.running = False
        except Exception as e:
            logger.error(f"❌ Health monitor error: {e}", exc_info=True)
            self.running = False
    
    async def stop(self):
        """Stop health monitoring"""
        logger.info("🛑 Stopping database health monitor...")
        self.running = False
    
    async def _perform_health_check(self):
        """Perform health check"""
        try:
            # Test 1: Connection test
            conn_start = time.time()
            await self._test_connection()
            conn_time = (time.time() - conn_start) * 1000
            
            # Test 2: Simple query test
            query_start = time.time()
            await self._test_query()
            query_time = (time.time() - query_start) * 1000
            
            # Update metrics
            self.metrics.last_check = datetime.utcnow()
            self.metrics.connection_test_ms = conn_time
            self.metrics.query_test_ms = query_time
            self.metrics.consecutive_failures = 0
            self.metrics.is_healthy = True
            self.metrics.uptime_seconds = time.time() - self.start_time
            
            # Track query performance
            self._track_query_time(query_time)
            
            # Check for slow queries
            if query_time > self.slow_query_threshold:
                self.metrics.slow_query_count += 1
                logger.warning(
                    f"⚠️ Slow database query detected: {query_time:.2f}ms "
                    f"(threshold: {self.slow_query_threshold}ms)"
                )
            
            logger.debug(
                f"✅ Database health check passed "
                f"(conn: {conn_time:.2f}ms, query: {query_time:.2f}ms)"
            )
            
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.utcnow()
            
            if self.metrics.consecutive_failures >= self.failure_threshold:
                self.metrics.is_healthy = False
                logger.error(
                    f"🔴 Database unhealthy after {self.metrics.consecutive_failures} "
                    f"consecutive failures: {e}"
                )
            else:
                logger.warning(
                    f"⚠️ Database health check failed "
                    f"({self.metrics.consecutive_failures}/{self.failure_threshold}): {e}"
                )
    
    async def _test_connection(self):
        """Test database connection"""
        # Simple connection test - just check if client is available
        if not self.supabase_client or not self.supabase_client.client:
            raise Exception("Supabase client not initialized")
    
    async def _test_query(self):
        """Test database with simple query"""
        # Simple query to test database responsiveness
        result = self.supabase_client.client.table("background_jobs")\
            .select("id")\
            .limit(1)\
            .execute()
        
        if not hasattr(result, 'data'):
            raise Exception("Invalid query response")
    
    def _track_query_time(self, query_time_ms: float):
        """Track query performance"""
        self.query_times.append(query_time_ms)
        
        # Keep only recent queries
        if len(self.query_times) > self.max_query_history:
            self.query_times.pop(0)
        
        # Update metrics
        if self.query_times:
            self.metrics.avg_query_time_ms = sum(self.query_times) / len(self.query_times)
            self.metrics.max_query_time_ms = max(self.query_times)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status including connection pool stats"""
        # Get connection pool statistics
        pool_stats = self.supabase_client.get_connection_pool_stats()

        return {
            "healthy": self.metrics.is_healthy,
            "last_check": self.metrics.last_check.isoformat() if self.metrics.last_check else None,
            "connection_test_ms": self.metrics.connection_test_ms,
            "query_test_ms": self.metrics.query_test_ms,
            "error_count": self.metrics.error_count,
            "consecutive_failures": self.metrics.consecutive_failures,
            "last_error": self.metrics.last_error,
            "last_error_time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
            "uptime_seconds": self.metrics.uptime_seconds,
            "performance": {
                "slow_query_count": self.metrics.slow_query_count,
                "avg_query_time_ms": round(self.metrics.avg_query_time_ms, 2),
                "max_query_time_ms": round(self.metrics.max_query_time_ms, 2),
                "slow_query_threshold_ms": self.slow_query_threshold
            },
            "connection_pool": pool_stats,  # NEW: Connection pool statistics
            "status": "healthy" if self.metrics.is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
database_health_service = DatabaseHealthService(
    check_interval_seconds=30,
    slow_query_threshold_ms=1000.0,
    failure_threshold=3
)

