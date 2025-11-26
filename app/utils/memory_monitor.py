"""
Memory Pressure Monitoring

Monitors system memory usage and prevents OOM crashes by pausing processing
when memory pressure is high.

Features:
- Real-time memory monitoring
- Automatic pause/resume based on thresholds
- Memory cleanup triggers
- Logging and alerts
"""

import psutil
import asyncio
import logging
import gc
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_mb: float
    
    @property
    def is_high_pressure(self) -> bool:
        """Check if memory pressure is high (>80%)"""
        return self.percent_used > 80
    
    @property
    def is_critical_pressure(self) -> bool:
        """Check if memory pressure is critical (>90%)"""
        return self.percent_used > 90


class MemoryPressureMonitor:
    """
    Monitor memory usage and prevent OOM crashes.
    
    Pauses processing when memory usage exceeds thresholds.
    """
    
    def __init__(
        self,
        high_threshold: float = 80.0,
        critical_threshold: float = 90.0,
        check_interval: float = 5.0,
        enable_auto_cleanup: bool = True
    ):
        """
        Initialize memory pressure monitor.
        
        Args:
            high_threshold: Pause processing at this % (default: 80%)
            critical_threshold: Force cleanup at this % (default: 90%)
            check_interval: How often to check memory (seconds)
            enable_auto_cleanup: Automatically trigger GC on high pressure
        """
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._pause_callback: Optional[Callable] = None
        self._resume_callback: Optional[Callable] = None
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        mem = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        return MemoryStats(
            total_mb=mem.total / 1024 / 1024,
            available_mb=mem.available / 1024 / 1024,
            used_mb=mem.used / 1024 / 1024,
            percent_used=mem.percent,
            process_mb=process_mem
        )
    
    async def check_memory_pressure(self) -> MemoryStats:
        """
        Check current memory pressure and take action if needed.
        
        Returns:
            Current memory statistics
        """
        stats = self.get_memory_stats()
        
        if stats.is_critical_pressure:
            logger.warning(
                f"🚨 CRITICAL memory pressure: {stats.percent_used:.1f}% "
                f"(process: {stats.process_mb:.1f} MB)"
            )
            
            if self.enable_auto_cleanup:
                logger.info("🧹 Triggering emergency garbage collection...")
                gc.collect()
                
                # Check again after cleanup
                stats = self.get_memory_stats()
                logger.info(f"   Memory after cleanup: {stats.percent_used:.1f}%")
        
        elif stats.is_high_pressure:
            logger.warning(
                f"⚠️ HIGH memory pressure: {stats.percent_used:.1f}% "
                f"(process: {stats.process_mb:.1f} MB)"
            )
            
            if self.enable_auto_cleanup:
                logger.info("🧹 Triggering garbage collection...")
                gc.collect()
        
        return stats
    
    async def wait_for_memory_available(
        self,
        required_mb: float = 100,
        max_wait_seconds: float = 60,
        operation_name: str = "operation"
    ):
        """
        Wait until enough memory is available to proceed.
        
        Args:
            required_mb: Minimum MB of free memory needed
            max_wait_seconds: Maximum time to wait
            operation_name: Name of operation for logging
        
        Raises:
            MemoryError: If memory doesn't become available in time
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            stats = await self.check_memory_pressure()
            
            if stats.available_mb >= required_mb:
                return  # Memory available
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait_seconds:
                raise MemoryError(
                    f"Insufficient memory for {operation_name}: "
                    f"need {required_mb} MB, have {stats.available_mb:.1f} MB"
                )
            
            logger.info(
                f"⏳ Waiting for memory: need {required_mb} MB, "
                f"have {stats.available_mb:.1f} MB (waiting {elapsed:.0f}s)"
            )
            
            await asyncio.sleep(self.check_interval)
    
    def log_memory_stats(self, prefix: str = ""):
        """Log current memory statistics"""
        stats = self.get_memory_stats()
        logger.info(
            f"{prefix}Memory: {stats.percent_used:.1f}% used "
            f"({stats.used_mb:.0f}/{stats.total_mb:.0f} MB), "
            f"process: {stats.process_mb:.1f} MB"
        )

    def calculate_optimal_batch_size(
        self,
        default_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 20,
        memory_per_item_mb: float = 10.0
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            default_batch_size: Default batch size to use
            min_batch_size: Minimum batch size (default: 1)
            max_batch_size: Maximum batch size (default: 20)
            memory_per_item_mb: Estimated memory per item in MB (default: 10MB)

        Returns:
            Optimal batch size based on available memory
        """
        stats = self.get_memory_stats()

        # If memory pressure is low (<50%), use default batch size
        if stats.percent_used < 50:
            return min(default_batch_size, max_batch_size)

        # If memory pressure is high (>80%), use minimum batch size
        if stats.is_high_pressure:
            logger.warning(
                f"⚠️ High memory pressure ({stats.percent_used:.1f}%), "
                f"reducing batch size to {min_batch_size}"
            )
            return min_batch_size

        # Calculate optimal batch size based on available memory
        # Reserve 20% of available memory as safety buffer
        usable_memory = stats.available_mb * 0.8
        optimal_batch = int(usable_memory / memory_per_item_mb)

        # Clamp to min/max range
        optimal_batch = max(min_batch_size, min(optimal_batch, max_batch_size))

        if optimal_batch < default_batch_size:
            logger.info(
                f"📊 Adjusting batch size: {default_batch_size} → {optimal_batch} "
                f"(available memory: {stats.available_mb:.0f} MB)"
            )

        return optimal_batch


# Global memory monitor instance
memory_monitor = MemoryPressureMonitor(
    high_threshold=80.0,
    critical_threshold=90.0,
    check_interval=5.0,
    enable_auto_cleanup=True
)

# Create global instance with alias for backward compatibility
global_memory_monitor = memory_monitor