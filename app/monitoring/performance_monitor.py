"""
Performance Monitoring Module for MIVAA PDF Extractor

This module provides comprehensive performance monitoring capabilities including:
- Request/response time tracking
- Memory usage monitoring
- CPU utilization tracking
- Database query performance
- Custom metrics collection
- Performance alerts and thresholds
- Integration with Sentry for performance monitoring
"""

import time
import psutil
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor

import sentry_sdk
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric with timestamp and value."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = "ms"


@dataclass
class PerformanceThreshold:
    """Defines performance thresholds for alerting."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    unit: str = "ms"
    enabled: bool = True


class PerformanceCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to the collection."""
        with self.lock:
            self.metrics.append(metric)
            self.aggregated_metrics[metric.name].append(metric.value)
            
            # Keep only recent metrics for aggregation
            if len(self.aggregated_metrics[metric.name]) > 1000:
                self.aggregated_metrics[metric.name] = self.aggregated_metrics[metric.name][-500:]
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics, optionally filtered by name and time."""
        with self.lock:
            filtered_metrics = list(self.metrics)
            
            if metric_name:
                filtered_metrics = [m for m in filtered_metrics if m.name == metric_name]
            
            if since:
                filtered_metrics = [m for m in filtered_metrics if m.timestamp >= since]
            
            return filtered_metrics
    
    def get_aggregated_stats(self, metric_name: str) -> Dict[str, float]:
        """Get aggregated statistics for a metric."""
        with self.lock:
            values = self.aggregated_metrics.get(metric_name, [])
            
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]


class SystemMonitor:
    """Monitors system-level performance metrics."""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start system monitoring with specified interval."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started system monitoring with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped system monitoring")
    
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.collector.add_metric(PerformanceMetric(
            name="system.cpu.usage",
            value=cpu_percent,
            timestamp=timestamp,
            unit="percent"
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.collector.add_metric(PerformanceMetric(
            name="system.memory.usage",
            value=memory.percent,
            timestamp=timestamp,
            unit="percent"
        ))
        
        self.collector.add_metric(PerformanceMetric(
            name="system.memory.available",
            value=memory.available / (1024 * 1024),  # MB
            timestamp=timestamp,
            unit="MB"
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.collector.add_metric(PerformanceMetric(
            name="system.disk.usage",
            value=(disk.used / disk.total) * 100,
            timestamp=timestamp,
            unit="percent"
        ))
        
        # Process-specific metrics
        process = psutil.Process()
        self.collector.add_metric(PerformanceMetric(
            name="process.memory.rss",
            value=process.memory_info().rss / (1024 * 1024),  # MB
            timestamp=timestamp,
            unit="MB"
        ))
        
        self.collector.add_metric(PerformanceMetric(
            name="process.cpu.usage",
            value=process.cpu_percent(),
            timestamp=timestamp,
            unit="percent"
        ))


class PerformanceTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(self, collector: PerformanceCollector, operation_name: str, 
                 tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = (time.perf_counter() - self.start_time) * 1000  # ms
            
            metric = PerformanceMetric(
                name=f"operation.{self.operation_name}",
                value=duration,
                timestamp=datetime.utcnow(),
                tags=self.tags
            )
            
            self.collector.add_metric(metric)
            
            # Send to Sentry if available
            try:
                sentry_sdk.set_measurement(f"operation.{self.operation_name}", duration, "millisecond")
            except Exception:
                pass  # Sentry not available or configured


@asynccontextmanager
async def async_performance_tracker(collector: PerformanceCollector, 
                                  operation_name: str,
                                  tags: Optional[Dict[str, str]] = None):
    """Async context manager for tracking operation performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = (time.perf_counter() - start_time) * 1000  # ms
        
        metric = PerformanceMetric(
            name=f"operation.{operation_name}",
            value=duration,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        
        collector.add_metric(metric)
        
        # Send to Sentry if available
        try:
            sentry_sdk.set_measurement(f"operation.{operation_name}", duration, "millisecond")
        except Exception:
            pass


def performance_monitor(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = getattr(func, '_performance_collector', None)
            if not collector:
                return await func(*args, **kwargs)
            
            async with async_performance_tracker(collector, operation_name, tags):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = getattr(func, '_performance_collector', None)
            if not collector:
                return func(*args, **kwargs)
            
            with PerformanceTracker(collector, operation_name, tags):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for tracking request performance."""
    
    def __init__(self, app, collector: PerformanceCollector):
        super().__init__(app)
        self.collector = collector
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        
        # Track request start
        self.collector.add_metric(PerformanceMetric(
            name="http.requests.active",
            value=1,
            timestamp=datetime.utcnow(),
            tags={"method": request.method, "path": request.url.path},
            unit="count"
        ))
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            duration = (time.perf_counter() - start_time) * 1000  # ms
            
            # Track response metrics
            self.collector.add_metric(PerformanceMetric(
                name="http.request.duration",
                value=duration,
                timestamp=datetime.utcnow(),
                tags={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": str(response.status_code)
                }
            ))
            
            # Send to Sentry
            try:
                with sentry_sdk.configure_scope() as scope:
                    scope.set_tag("http.method", request.method)
                    scope.set_tag("http.path", request.url.path)
                    scope.set_tag("http.status_code", response.status_code)
                
                sentry_sdk.set_measurement("http.request.duration", duration, "millisecond")
            except Exception:
                pass
            
            return response
            
        except Exception as e:
            # Track error metrics
            duration = (time.perf_counter() - start_time) * 1000  # ms
            
            self.collector.add_metric(PerformanceMetric(
                name="http.request.errors",
                value=1,
                timestamp=datetime.utcnow(),
                tags={
                    "method": request.method,
                    "path": request.url.path,
                    "error_type": type(e).__name__
                },
                unit="count"
            ))
            
            raise


class PerformanceAlertManager:
    """Manages performance alerts and thresholds."""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alert_callbacks: List[Callable] = []
        self.last_alerts: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)  # Prevent spam
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add a performance threshold."""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added performance threshold for {threshold.metric_name}")
    
    def add_alert_callback(self, callback: Callable[[str, PerformanceMetric, PerformanceThreshold], None]):
        """Add a callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def check_thresholds(self, metric: PerformanceMetric):
        """Check if a metric exceeds thresholds."""
        threshold = self.thresholds.get(metric.name)
        if not threshold or not threshold.enabled:
            return
        
        alert_level = None
        if metric.value >= threshold.critical_threshold:
            alert_level = "critical"
        elif metric.value >= threshold.warning_threshold:
            alert_level = "warning"
        
        if alert_level:
            # Check cooldown
            last_alert = self.last_alerts.get(f"{metric.name}_{alert_level}")
            if last_alert and datetime.utcnow() - last_alert < self.alert_cooldown:
                return
            
            self.last_alerts[f"{metric.name}_{alert_level}"] = datetime.utcnow()
            
            # Trigger alerts
            for callback in self.alert_callbacks:
                try:
                    callback(alert_level, metric, threshold)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")


class PerformanceOptimizer:
    """Provides performance optimization recommendations."""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and provide recommendations."""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": [],
            "metrics_summary": {},
            "health_score": 100
        }
        
        # Analyze HTTP request performance
        http_stats = self.collector.get_aggregated_stats("http.request.duration")
        if http_stats:
            analysis["metrics_summary"]["http_requests"] = http_stats
            
            if http_stats.get("p95", 0) > 2000:  # 2 seconds
                analysis["recommendations"].append({
                    "type": "performance",
                    "severity": "high",
                    "message": "HTTP request P95 latency is high (>2s). Consider optimizing slow endpoints.",
                    "metric": "http.request.duration",
                    "value": http_stats["p95"]
                })
                analysis["health_score"] -= 20
        
        # Analyze memory usage
        memory_stats = self.collector.get_aggregated_stats("system.memory.usage")
        if memory_stats:
            analysis["metrics_summary"]["memory_usage"] = memory_stats
            
            if memory_stats.get("avg", 0) > 80:  # 80%
                analysis["recommendations"].append({
                    "type": "resource",
                    "severity": "medium",
                    "message": "High memory usage detected. Consider optimizing memory consumption.",
                    "metric": "system.memory.usage",
                    "value": memory_stats["avg"]
                })
                analysis["health_score"] -= 15
        
        # Analyze CPU usage
        cpu_stats = self.collector.get_aggregated_stats("system.cpu.usage")
        if cpu_stats:
            analysis["metrics_summary"]["cpu_usage"] = cpu_stats
            
            if cpu_stats.get("avg", 0) > 70:  # 70%
                analysis["recommendations"].append({
                    "type": "resource",
                    "severity": "medium",
                    "message": "High CPU usage detected. Consider optimizing CPU-intensive operations.",
                    "metric": "system.cpu.usage",
                    "value": cpu_stats["avg"]
                })
                analysis["health_score"] -= 10
        
        return analysis
    
    def get_slow_operations(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Get operations that are slower than threshold."""
        slow_ops = []
        
        for metric_name in self.collector.aggregated_metrics:
            if metric_name.startswith("operation."):
                stats = self.collector.get_aggregated_stats(metric_name)
                if stats and stats.get("p95", 0) > threshold_ms:
                    slow_ops.append({
                        "operation": metric_name,
                        "p95_duration": stats["p95"],
                        "avg_duration": stats["avg"],
                        "max_duration": stats["max"],
                        "count": stats["count"]
                    })
        
        return sorted(slow_ops, key=lambda x: x["p95_duration"], reverse=True)


class PerformanceMonitor:
    """Main performance monitoring class that coordinates all components."""
    
    def __init__(self, max_metrics: int = 10000):
        self.collector = PerformanceCollector(max_metrics)
        self.system_monitor = SystemMonitor(self.collector)
        self.alert_manager = PerformanceAlertManager(self.collector)
        self.optimizer = PerformanceOptimizer(self.collector)
        
        # Setup default thresholds
        self._setup_default_thresholds()
        
        # Setup default alert callback
        self.alert_manager.add_alert_callback(self._default_alert_callback)
    
    def _setup_default_thresholds(self):
        """Setup default performance thresholds."""
        thresholds = [
            PerformanceThreshold("http.request.duration", 1000, 5000),  # 1s warning, 5s critical
            PerformanceThreshold("system.memory.usage", 80, 95, "percent"),
            PerformanceThreshold("system.cpu.usage", 70, 90, "percent"),
            PerformanceThreshold("operation.pdf_processing", 5000, 15000),  # 5s warning, 15s critical
        ]
        
        for threshold in thresholds:
            self.alert_manager.add_threshold(threshold)
    
    def _default_alert_callback(self, level: str, metric: PerformanceMetric, threshold: PerformanceThreshold):
        """Default alert callback that logs alerts."""
        logger.warning(
            f"Performance alert [{level.upper()}]: {metric.name} = {metric.value}{metric.unit} "
            f"(threshold: {threshold.warning_threshold if level == 'warning' else threshold.critical_threshold}{threshold.unit})"
        )
        
        # Send to Sentry
        try:
            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("alert.level", level)
                scope.set_tag("alert.metric", metric.name)
                scope.set_extra("metric.value", metric.value)
                scope.set_extra("threshold.warning", threshold.warning_threshold)
                scope.set_extra("threshold.critical", threshold.critical_threshold)
            
            sentry_sdk.capture_message(
                f"Performance alert: {metric.name} exceeded {level} threshold",
                level="warning" if level == "warning" else "error"
            )
        except Exception:
            pass
    
    async def start(self, system_monitoring_interval: float = 30.0):
        """Start performance monitoring."""
        await self.system_monitor.start_monitoring(system_monitoring_interval)
        logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring."""
        await self.system_monitor.stop_monitoring()
        logger.info("Performance monitoring stopped")
    
    def get_middleware(self) -> PerformanceMiddleware:
        """Get FastAPI middleware for request monitoring."""
        return PerformanceMiddleware(None, self.collector)
    
    def track_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None) -> PerformanceTracker:
        """Get a performance tracker for an operation."""
        return PerformanceTracker(self.collector, operation_name, tags)
    
    async def track_async_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Get an async performance tracker for an operation."""
        return async_performance_tracker(self.collector, operation_name, tags)
    
    def add_metric(self, name: str, value: float, unit: str = "ms", tags: Optional[Dict[str, str]] = None):
        """Add a custom metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )
        self.collector.add_metric(metric)
        self.alert_manager.check_thresholds(metric)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report."""
        return {
            "analysis": self.optimizer.analyze_performance(),
            "slow_operations": self.optimizer.get_slow_operations(),
            "recent_metrics": {
                name: self.collector.get_aggregated_stats(name)
                for name in ["http.request.duration", "system.memory.usage", "system.cpu.usage"]
                if name in self.collector.aggregated_metrics
            }
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for API endpoint."""
        try:
            # Get recent metrics
            recent_metrics = {}
            for metric_name in ["http.request.duration", "system.memory.usage", "system.cpu.usage", "system.disk.usage"]:
                stats = self.collector.get_aggregated_stats(metric_name)
                if stats:
                    recent_metrics[metric_name] = stats

            # Get system info
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": recent_metrics,
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_usage_percent": (disk.used / disk.total) * 100,
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024)
                },
                "performance": {
                    "total_requests": len(self.collector.metrics),
                    "slow_operations_count": len(self.optimizer.get_slow_operations()),
                    "health_score": self.optimizer.analyze_performance().get("health_score", 100)
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get metrics: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for API endpoint."""
        try:
            analysis = self.optimizer.analyze_performance()
            slow_ops = self.optimizer.get_slow_operations()

            # Calculate uptime (mock for now)
            uptime_hours = 24.0  # Would be calculated from service start time

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "health_score": analysis.get("health_score", 100),
                    "total_requests": len(self.collector.metrics),
                    "slow_operations": len(slow_ops),
                    "uptime_hours": uptime_hours,
                    "recommendations_count": len(analysis.get("recommendations", []))
                },
                "top_slow_operations": slow_ops[:5],  # Top 5 slowest
                "recommendations": analysis.get("recommendations", [])[:3],  # Top 3 recommendations
                "recent_performance": {
                    name: self.collector.get_aggregated_stats(name)
                    for name in ["http.request.duration", "system.cpu.usage", "system.memory.usage"]
                    if name in self.collector.aggregated_metrics
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get performance summary: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
