"""
HuggingFace Endpoint Auto-Scaler

Automatically scales HuggingFace Inference Endpoint replicas based on job queue depth.
Monitors pending/processing jobs and adjusts replicas accordingly.

Strategy:
- Queue > 3 jobs: Scale to 2 replicas
- Queue > 6 jobs: Scale to max replicas (2 or 3 depending on endpoint config)
- Queue = 0 for 10 min: Scale to min replicas (0 or 1)
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
    
    Manages the 3 HuggingFace endpoints we still depend on:
    - mh-slig (SLIG visual embeddings)
    - mh-chandra (Chandra OCR fallback)
    - mh-yolo (YOLO layout detection)
    """
    
    def __init__(
        self,
        hf_token: str,
        namespace: str = "basiliskan",
        check_interval_seconds: int = 30,
        scale_up_threshold: int = 3,      # Jobs to trigger scale up
        scale_down_idle_minutes: int = 10,  # Minutes idle before scale down
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

        # All HuggingFace endpoints to manage. Resolved from Settings (or env
        # var overrides) so the user can rename endpoints on HF without
        # editing code.
        try:
            from app.config import get_settings
            _s = get_settings()
            self.managed_endpoints = [
                _s.slig_endpoint_name,
                _s.chandra_endpoint_name,
                _s.yolo_endpoint_name,
            ]
        except Exception as _settings_err:
            logger.warning(
                f"Auto-scaler couldn't read endpoint names from Settings ({_settings_err}); "
                f"falling back to current defaults"
            )
            self.managed_endpoints = [
                "mh-slig",
                "chandra-ocr-2",
                "mh-yolo",
            ]
        
        if not HF_HUB_AVAILABLE:
            logger.warning("huggingface_hub not available - auto-scaling disabled")
            self.enabled = False
        
        logger.info(f"✅ Endpoint Auto-Scaler initialized: enabled={enabled}, endpoints={len(self.managed_endpoints)}")
    
    async def initialize_endpoint_configs(self):
        """Fetch and store the min/max replica config AND current replica count
        for each endpoint.

        Seeding `_current_replicas` from actual HF state (not 0) is the fix for
        the 'orphan replicas survive a restart and bill forever' leak: if the
        previous MIVAA process died mid-job without running scale_to_zero(),
        endpoints are still live on HF. Without this seeding the in-memory
        view says replicas=0 (matches desired=0 when queue empty) and the
        scaler never issues a scale-down, so we keep paying.
        """
        if self._endpoints_initialized:
            return

        for endpoint_name in self.managed_endpoints:
            try:
                endpoint = self.get_endpoint(endpoint_name)
                if endpoint:
                    endpoint.fetch()

                    # Min/max replica config
                    scaling = getattr(endpoint, 'scaling', None)
                    raw = getattr(endpoint, 'raw', {})

                    if scaling:
                        min_rep = getattr(scaling, 'min_replica', 0)
                        max_rep = getattr(scaling, 'max_replica', 2)
                    elif 'compute' in raw and 'scaling' in raw['compute']:
                        min_rep = raw['compute']['scaling'].get('minReplica', 0)
                        max_rep = raw['compute']['scaling'].get('maxReplica', 2)
                    else:
                        min_rep = 0
                        max_rep = 2

                    # Current actual replica count on HF — covers the case
                    # where endpoints are still alive from before our restart.
                    current_rep = getattr(endpoint, 'replica', None)
                    if current_rep is None and 'compute' in raw:
                        # Some HF SDK versions expose it as raw.compute.scaling.minReplica
                        # while running; fall back to the configured min.
                        current_rep = min_rep
                    if current_rep is None:
                        current_rep = 0

                    self._endpoint_configs[endpoint_name] = {
                        'min_replica': min_rep,
                        'max_replica': max_rep,
                        'status': endpoint.status,
                    }
                    self._current_replicas[endpoint_name] = int(current_rep)

                    logger.info(
                        f"📊 {endpoint_name}: min={min_rep}, max={max_rep}, "
                        f"status={endpoint.status}, current_replicas={current_rep}"
                    )

            except Exception as e:
                logger.warning(f"⚠️ Could not fetch config for {endpoint_name}: {e}")
                self._endpoint_configs[endpoint_name] = {
                    'min_replica': 0,
                    'max_replica': 2,
                }

        self._endpoints_initialized = True
        logger.info(f"✅ Initialized configs for {len(self._endpoint_configs)} endpoints")
    
    async def get_queue_depth(self) -> int:
        """Get number of LIVE work-units pending/processing.

        Count is **product-level**, not job-level. A single PDF upload with 11
        products counts as 11 units of work — that's the right signal for
        deciding how many replicas to keep alive, since each product is
        roughly an independent stream of YOLO/SLIG/Qwen calls.

        For jobs that haven't started Stage 0 yet (no products in DB), each
        such job contributes 1 to the queue depth (placeholder) so we don't
        de-provision endpoints right when a fresh upload lands.

        Stale-heartbeat jobs (>5 min since last heartbeat) are excluded —
        their products aren't actually doing work, the worker is dead.
        """
        try:
            from app.services.core.supabase_client import get_supabase_client

            supabase = get_supabase_client()
            stale_cutoff = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
            recent_create_cutoff = (datetime.utcnow() - timedelta(seconds=60)).isoformat()

            # Live jobs (pending + processing-with-fresh-heartbeat)
            pending = supabase.client.table('background_jobs') \
                .select('id', count='exact') \
                .eq('status', 'pending') \
                .execute()
            pending_count = pending.count or 0

            # Two-query union — the installed supabase-py version (used by
            # the SyncSelectRequestBuilder path) does not expose `.or_()`.
            # Splitting avoids the AttributeError that wedged this poll
            # every cycle and made queue_depth always return 0.
            fresh_hb = supabase.client.table('background_jobs') \
                .select('id') \
                .eq('status', 'processing') \
                .gte('last_heartbeat', stale_cutoff) \
                .execute()
            no_hb_yet = supabase.client.table('background_jobs') \
                .select('id') \
                .eq('status', 'processing') \
                .is_('last_heartbeat', 'null') \
                .gte('created_at', recent_create_cutoff) \
                .execute()
            live_processing_ids = list({
                *[row['id'] for row in (fresh_hb.data or [])],
                *[row['id'] for row in (no_hb_yet.data or [])],
            })
            processing_count = len(live_processing_ids)

            # For each live processing job, count its in-flight products
            # (status != completed/failed). Falls back to 1 if no products yet
            # (Stage 0 still running) so the scaler doesn't under-provision.
            product_units = 0
            if live_processing_ids:
                pps_resp = supabase.client.table('product_processing_status') \
                    .select('job_id', count='exact') \
                    .in_('job_id', live_processing_ids) \
                    .not_.in_('status', ['completed', 'failed']) \
                    .execute()
                product_units = pps_resp.count or 0
                # If we have processing jobs but zero in-flight products
                # (Stage 0 hasn't created the rows yet), give each job a
                # placeholder weight of 1 so endpoints stay warm.
                if product_units == 0:
                    product_units = processing_count

            depth = pending_count + product_units
            return depth
        except Exception as e:
            logger.warning(f"Failed to get queue depth: {e}")
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
            logger.warning(f"Failed to get endpoint {endpoint_name}: {e}")
            return None
    
    def scale_endpoint(self, endpoint_name: str, desired_replicas: int) -> bool:
        """
        Cap an endpoint's MAX replicas based on queue depth — let HF's
        hardware-usage autoscaler handle actually spinning replicas up
        and down between 0 and that cap.

        Always set `min_replica=0` so scale-to-zero is preserved, and set
        `max_replica` to the desired cap so HF's built-in
        `metric=hardwareUsage, threshold=80%` autoscaler provisions only
        what's actually needed. The desired_replicas argument therefore
        acts as a CEILING, not a floor — moving the floor would pin
        replicas at that count 24/7 and defeat scale-to-zero.
        """
        endpoint = self.get_endpoint(endpoint_name)
        if not endpoint:
            return False

        try:
            endpoint.fetch()
            config = self._endpoint_configs.get(endpoint_name, {'min_replica': 0, 'max_replica': 2})

            # Clamp the requested ceiling to the endpoint's configured maxReplica.
            # `desired_replicas` is the suggested ceiling from queue-depth math;
            # never raise above what HF was originally configured for.
            absolute_max = config['max_replica']
            new_max = max(1, min(desired_replicas, absolute_max))

            current_max = config.get('max_replica', new_max)
            if new_max == current_max:
                return True  # Cap already where we want it

            logger.info(
                f"📈 Adjusting {endpoint_name} max_replica cap: "
                f"{current_max} → {new_max} (min stays 0, HF autoscale fills based on hardware usage)"
            )

            # ALWAYS keep min_replica=0 so the endpoint can scale to zero on
            # idle. We only move the ceiling.
            endpoint.update(min_replica=0, max_replica=new_max)

            # Track the effective cap, not the actual replica count — HF
            # decides actual replicas based on load.
            self._endpoint_configs[endpoint_name]['max_replica'] = new_max
            self._current_replicas[endpoint_name] = self._current_replicas.get(endpoint_name, 0)
            logger.info(f"✅ {endpoint_name} cap now {new_max}, min=0 (scale-to-zero preserved)")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to scale {endpoint_name}: {e}")
            return False
    
    def calculate_desired_replicas(self, queue_depth: int, max_replicas: int) -> int:
        """Calculate desired replicas based on product-level queue depth.

        Tuned for max_replicas=4 (current production config). With queue depth
        now counting in-flight products (not just jobs), we want a single PDF
        with N products to provision proportional replica count:

            queue == 0      → 0 (scaled to zero, $0/hr)
            queue 1-2       → 1 replica
            queue 3-5       → 2 replicas
            queue 6-9       → 3 replicas
            queue >= 10     → max_replicas (typically 4)

        Each replica handles roughly 2-3 concurrent products comfortably given
        the per-endpoint AIMD gates (qwen=8 cap, slig=16, yolo=12). 4 replicas
        × 3 products/replica = 12 products in flight at peak — matches the
        typical catalog size (10-15 products) almost perfectly.
        """
        if queue_depth == 0:
            return 0
        elif queue_depth <= 2:
            return 1
        elif queue_depth <= 5:
            return min(2, max_replicas)
        elif queue_depth <= 9:
            return min(3, max_replicas)
        else:
            return max_replicas
    
    async def check_and_scale(self):
        """Check queue depth and scale endpoints accordingly."""
        if not self.enabled:
            return

        # P3-3: cheap idle short-circuit — when the registry says nothing is
        # processing AND we haven't tracked any live replicas, skip the DB
        # round-trip entirely. Saves ~2880 supabase reads/day in the steady-
        # state-idle case (auto-scaler ticks every 30s × 24h).
        try:
            from app.services.embeddings.endpoint_registry import endpoint_registry
            if not endpoint_registry.is_processing() and self._endpoints_initialized and not any(
                self._current_replicas.get(name, 0) > 0 for name in self.managed_endpoints
            ):
                return
        except Exception:
            pass

        queue_depth = await self.get_queue_depth()

        # First-tick guarantee: always run initialize_endpoint_configs at least
        # once so we discover replicas already running from before this process
        # started (e.g. mid-job restart). Without this we'd silently skip the
        # scale-down for orphaned replicas because the in-memory view says 0.
        if not self._endpoints_initialized:
            await self.initialize_endpoint_configs()

        # P0-1 cost watchdog: even when queue=0, if any endpoint is still
        # running per HF state we must drive it down to 0. Don't early-return
        # in that case — the scale-down logic at the bottom of this function
        # will see desired=0 < current>0 and start the idle timer.
        if queue_depth == 0 and not any(
            self._current_replicas.get(name, 0) > 0 for name in self.managed_endpoints
        ):
            # P3-3: skip the supabase round-trip on idle ticks too —
            # the get_queue_depth call above already returned 0, no further
            # work needed.
            return

        for endpoint_name in self.managed_endpoints:
            config = self._endpoint_configs.get(endpoint_name, {'max_replica': 2})
            max_rep = config.get('max_replica', 2)

            desired_replicas = self.calculate_desired_replicas(queue_depth, max_rep)
            current = self._current_replicas.get(endpoint_name, 0)

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
                logger.warning(f"Auto-scaler cycle error: {e}")
            
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
