"""
HuggingFace Endpoint Auto-Scaler

Automatically scales HuggingFace Inference Endpoint replicas based on job queue depth.
Monitors pending/processing jobs and adjusts replicas accordingly.

Strategy:
- Queue > 3 jobs: Scale to 2 replicas
- Queue > 6 jobs: Scale to max replicas (2 or 3 depending on endpoint config)
- Queue = 0 for 5 min: Scale to min replicas (0 or 1)
"""

import os
import time
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta

try:
    from huggingface_hub import get_inference_endpoint
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class EndpointAutoScaler:
    """
    Auto-scales HuggingFace endpoints based on job queue depth.
    
    Manages all 4 HuggingFace endpoints:
    - mh-qwen332binstruct (Qwen VLM - primary, high priority)
    - mh-siglip2 (SLIG visual embeddings)
    - mh-chandra (Chandra OCR fallback)
    - mh-yolo (YOLO layout detection)
    """
    
    def __init__(
        self,
        hf_token: str,
        namespace: str = "basiliskan",
        check_interval_seconds: int = 30,
        scale_up_threshold: int = 3,      # Jobs to trigger scale up
        scale_down_idle_minutes: int = 5,  # Minutes idle before scale down
        enabled: bool = True
    ):
        self.hf_token = hf_token
        self.namespace = namespace
        self.check_interval = check_interval_seconds
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_idle_minutes = scale_down_idle_minutes
        self.enabled = enabled
        
        # Track scaling state
        self._current_replicas: Dict[str, int] = {}
        self._endpoint_configs: Dict[str, Dict] = {}  # Stores min/max replicas per endpoint
        self._last_queue_empty: Dict[str, datetime] = {}
        self._running = False
        self._endpoints_initialized = False
        
        # All HuggingFace endpoints to manage
        self.managed_endpoints = [
            "mh-qwen332binstruct",  # Qwen VLM - primary workload
            "mh-siglip2",           # SLIG visual embeddings
            "mh-chandra",           # Chandra OCR
            "mh-yolo",              # YOLO layout detection
        ]
        
        if not HF_HUB_AVAILABLE:
            logger.warning("huggingface_hub not available - auto-scaling disabled")
            self.enabled = False
        
        logger.info(f"✅ Endpoint Auto-Scaler initialized: enabled={enabled}, endpoints={len(self.managed_endpoints)}")
    
    async def initialize_endpoint_configs(self):
        """Fetch and store the min/max replica configuration for each endpoint."""
        if self._endpoints_initialized:
            return
        
        for endpoint_name in self.managed_endpoints:
            try:
                endpoint = self.get_endpoint(endpoint_name)
                if endpoint:
                    endpoint.fetch()
                    
                    # Get scaling configuration from endpoint
                    scaling = getattr(endpoint, 'scaling', None)
                    raw = getattr(endpoint, 'raw', {})
                    
                    # Try to get from scaling object or raw data
                    if scaling:
                        min_rep = getattr(scaling, 'min_replica', 0)
                        max_rep = getattr(scaling, 'max_replica', 2)
                    elif 'compute' in raw and 'scaling' in raw['compute']:
                        min_rep = raw['compute']['scaling'].get('minReplica', 0)
                        max_rep = raw['compute']['scaling'].get('maxReplica', 2)
                    else:
                        # Default conservative values
                        min_rep = 0
                        max_rep = 2
                    
                    self._endpoint_configs[endpoint_name] = {
                        'min_replica': min_rep,
                        'max_replica': max_rep,
                        'status': endpoint.status
                    }
                    
                    logger.info(f"📊 {endpoint_name}: min={min_rep}, max={max_rep}, status={endpoint.status}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch config for {endpoint_name}: {e}")
                # Use defaults
                self._endpoint_configs[endpoint_name] = {
                    'min_replica': 0,
                    'max_replica': 2
                }
        
        self._endpoints_initialized = True
        logger.info(f"✅ Initialized configs for {len(self._endpoint_configs)} endpoints")
    
    async def get_queue_depth(self) -> int:
        """Get number of pending/processing jobs from database."""
        try:
            from app.services.core.supabase_client import get_supabase_client
            
            supabase = get_supabase_client()
            result = supabase.client.table('background_jobs') \
                .select('id', count='exact') \
                .in_('status', ['pending', 'processing']) \
                .execute()
            
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get queue depth: {e}")
            return 0
    
    def get_endpoint(self, endpoint_name: str):
        """Get HuggingFace endpoint instance."""
        if not HF_HUB_AVAILABLE:
            return None
        try:
            return get_inference_endpoint(
                name=endpoint_name,
                namespace=self.namespace,
                token=self.hf_token
            )
        except Exception as e:
            logger.error(f"Failed to get endpoint {endpoint_name}: {e}")
            return None
    
    def scale_endpoint(self, endpoint_name: str, desired_replicas: int) -> bool:
        """
        Scale endpoint to specified number of replicas.
        
        Uses endpoint.update(min_replica=N, max_replica=M) API.
        Respects endpoint's configured max_replica limit.
        """
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            return False
        
        try:
            endpoint.fetch()
            current = self._current_replicas.get(endpoint_name, 1)
            config = self._endpoint_configs.get(endpoint_name, {'min_replica': 0, 'max_replica': 2})
            
            # Clamp to endpoint's configured limits
            max_allowed = config['max_replica']
            min_allowed = config['min_replica']
            target_replicas = max(min_allowed, min(desired_replicas, max_allowed))
            
            if target_replicas == current:
                return True  # Already at desired scale
            
            logger.info(f"📈 Scaling {endpoint_name}: {current} → {target_replicas} replicas (max={max_allowed})")
            
            # Update endpoint scaling configuration
            # Set min and max to same value for fixed replica count
            endpoint.update(min_replica=target_replicas, max_replica=max_allowed)
            
            self._current_replicas[endpoint_name] = target_replicas
            logger.info(f"✅ {endpoint_name} scaled to {target_replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to scale {endpoint_name}: {e}")
            return False
    
    def calculate_desired_replicas(self, queue_depth: int, max_replicas: int) -> int:
        """Calculate desired replicas based on queue depth and endpoint's max."""
        if queue_depth == 0:
            return 0  # Scale to zero when idle (pause)
        elif queue_depth <= self.scale_up_threshold:
            return 1  # Light load, 1 replica
        elif queue_depth <= self.scale_up_threshold * 2:
            return min(2, max_replicas)  # Medium load
        else:
            return max_replicas  # Heavy load, max out
    
    async def check_and_scale(self):
        """Check queue depth and scale endpoints accordingly."""
        if not self.enabled:
            return
        
        # Initialize endpoint configs on first run
        if not self._endpoints_initialized:
            await self.initialize_endpoint_configs()
        
        queue_depth = await self.get_queue_depth()
        
        for endpoint_name in self.managed_endpoints:
            config = self._endpoint_configs.get(endpoint_name, {'max_replica': 2})
            max_rep = config.get('max_replica', 2)
            
            desired_replicas = self.calculate_desired_replicas(queue_depth, max_rep)
            current = self._current_replicas.get(endpoint_name, 1)
            
            # Scale up immediately if needed
            if desired_replicas > current:
                self.scale_endpoint(endpoint_name, desired_replicas)
                self._last_queue_empty.pop(endpoint_name, None)
            
            # Scale down only after idle period
            elif desired_replicas < current:
                if queue_depth == 0:
                    # Track when queue became empty
                    if endpoint_name not in self._last_queue_empty:
                        self._last_queue_empty[endpoint_name] = datetime.now()
                        logger.info(f"⏳ Queue empty, will scale down {endpoint_name} in {self.scale_down_idle_minutes} min")
                    
                    # Check if idle long enough
                    idle_time = datetime.now() - self._last_queue_empty[endpoint_name]
                    if idle_time >= timedelta(minutes=self.scale_down_idle_minutes):
                        self.scale_endpoint(endpoint_name, desired_replicas)
                else:
                    # Queue not empty, reset idle timer
                    self._last_queue_empty.pop(endpoint_name, None)
    
    async def start(self):
        """Start the auto-scaler background task."""
        if not self.enabled:
            logger.info("Auto-scaler disabled, not starting")
            return
        
        self._running = True
        logger.info(f"🚀 Auto-scaler started (checking every {self.check_interval}s, managing {len(self.managed_endpoints)} endpoints)")
        
        while self._running:
            try:
                await self.check_and_scale()
            except Exception as e:
                logger.error(f"Auto-scaler error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop the auto-scaler."""
        self._running = False
        logger.info("Auto-scaler stopped")
    
    def get_status(self) -> Dict:
        """Get current auto-scaler status."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "managed_endpoints": self.managed_endpoints,
            "endpoint_configs": self._endpoint_configs.copy(),
            "current_replicas": self._current_replicas.copy(),
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_idle_minutes": self.scale_down_idle_minutes
        }


# Global instance
_auto_scaler: Optional[EndpointAutoScaler] = None


def get_auto_scaler() -> Optional[EndpointAutoScaler]:
    """Get the global auto-scaler instance."""
    global _auto_scaler
    return _auto_scaler


def initialize_auto_scaler(
    hf_token: str,
    namespace: str = "basiliskan",
    enabled: bool = True
) -> EndpointAutoScaler:
    """Initialize the global auto-scaler."""
    global _auto_scaler
    
    _auto_scaler = EndpointAutoScaler(
        hf_token=hf_token,
        namespace=namespace,
        enabled=enabled
    )
    
    return _auto_scaler
