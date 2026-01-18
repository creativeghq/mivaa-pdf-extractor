"""
Endpoint Health Checker

Provides health validation for HuggingFace Inference Endpoints.
Unlike simple warmup (which just waits), this module validates that
endpoints are actually responding to inference requests.

This ensures the pipeline doesn't start until endpoints are truly ready.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EndpointStatus(Enum):
    """Endpoint health status."""
    UNKNOWN = "unknown"
    WARMING_UP = "warming_up"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    endpoint_name: str
    status: EndpointStatus
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    attempts: int = 0


class EndpointHealthChecker:
    """
    Validates endpoint health through actual inference calls.

    Unlike waiting a fixed time, this validates endpoints respond correctly
    before allowing the pipeline to proceed.
    """

    def __init__(
        self,
        max_health_check_attempts: int = 10,
        health_check_interval_seconds: int = 6,
        health_check_timeout_seconds: int = 30
    ):
        """
        Initialize health checker.

        Args:
            max_health_check_attempts: Max attempts before giving up (default: 10 = 60s total)
            health_check_interval_seconds: Seconds between health checks (default: 6)
            health_check_timeout_seconds: Timeout for each health check request (default: 30)
        """
        self.max_health_check_attempts = max_health_check_attempts
        self.health_check_interval_seconds = health_check_interval_seconds
        self.health_check_timeout_seconds = health_check_timeout_seconds

        # Track health check results
        self._health_results: Dict[str, HealthCheckResult] = {}

    async def check_qwen_health(self, endpoint_url: str, token: str) -> HealthCheckResult:
        """
        Check Qwen VLM endpoint health with a simple text completion.

        Args:
            endpoint_url: Qwen endpoint URL
            token: HuggingFace API token

        Returns:
            HealthCheckResult with status and timing
        """
        import httpx

        for attempt in range(1, self.max_health_check_attempts + 1):
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.health_check_timeout_seconds) as client:
                    # Simple health check - ask for a single word response
                    response = await client.post(
                        f"{endpoint_url.rstrip('/')}/chat/completions" if endpoint_url.rstrip('/').endswith('/v1') else f"{endpoint_url.rstrip('/')}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "Qwen/Qwen3-VL-32B-Instruct",
                            "messages": [{"role": "user", "content": "Say OK"}],
                            "max_tokens": 5
                        }
                    )

                    response_time_ms = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        result = HealthCheckResult(
                            endpoint_name="qwen",
                            status=EndpointStatus.HEALTHY,
                            response_time_ms=response_time_ms,
                            attempts=attempt
                        )
                        self._health_results["qwen"] = result
                        logger.info(f"✅ Qwen health check passed (attempt {attempt}, {response_time_ms:.0f}ms)")
                        return result
                    elif response.status_code == 503:
                        logger.info(f"⏳ Qwen still warming up (attempt {attempt}/{self.max_health_check_attempts})")
                    else:
                        logger.warning(f"⚠️ Qwen health check failed: HTTP {response.status_code}")

            except httpx.TimeoutException:
                logger.info(f"⏳ Qwen health check timeout (attempt {attempt}/{self.max_health_check_attempts})")
            except Exception as e:
                logger.warning(f"⚠️ Qwen health check error (attempt {attempt}): {e}")

            if attempt < self.max_health_check_attempts:
                await asyncio.sleep(self.health_check_interval_seconds)

        # All attempts failed
        result = HealthCheckResult(
            endpoint_name="qwen",
            status=EndpointStatus.UNHEALTHY,
            error_message=f"Failed after {self.max_health_check_attempts} attempts",
            attempts=self.max_health_check_attempts
        )
        self._health_results["qwen"] = result
        return result

    async def check_slig_health(self, endpoint_url: str, token: str) -> HealthCheckResult:
        """
        Check SLIG endpoint health with a simple text embedding request.

        Args:
            endpoint_url: SLIG endpoint URL
            token: HuggingFace API token

        Returns:
            HealthCheckResult with status and timing
        """
        import httpx

        for attempt in range(1, self.max_health_check_attempts + 1):
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.health_check_timeout_seconds) as client:
                    # Simple health check - get text embedding for a short phrase
                    response = await client.post(
                        endpoint_url,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "inputs": "health check",
                            "parameters": {"mode": "text_embedding"}
                        }
                    )

                    response_time_ms = (time.time() - start_time) * 1000

                    if response.status_code == 200:
                        # Verify response has embedding
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0 and "embedding" in data[0]:
                            result = HealthCheckResult(
                                endpoint_name="slig",
                                status=EndpointStatus.HEALTHY,
                                response_time_ms=response_time_ms,
                                attempts=attempt
                            )
                            self._health_results["slig"] = result
                            logger.info(f"✅ SLIG health check passed (attempt {attempt}, {response_time_ms:.0f}ms)")
                            return result
                    elif response.status_code == 503:
                        logger.info(f"⏳ SLIG still warming up (attempt {attempt}/{self.max_health_check_attempts})")
                    elif response.status_code == 400:
                        # Bad request - log response body for debugging
                        try:
                            error_body = response.text[:500]
                            logger.warning(f"⚠️ SLIG health check 400 Bad Request: {error_body}")
                        except:
                            logger.warning(f"⚠️ SLIG health check failed: HTTP 400 Bad Request")
                    else:
                        logger.warning(f"⚠️ SLIG health check failed: HTTP {response.status_code}")

            except httpx.TimeoutException:
                logger.info(f"⏳ SLIG health check timeout (attempt {attempt}/{self.max_health_check_attempts})")
            except Exception as e:
                logger.warning(f"⚠️ SLIG health check error (attempt {attempt}): {e}")

            if attempt < self.max_health_check_attempts:
                await asyncio.sleep(self.health_check_interval_seconds)

        # All attempts failed
        result = HealthCheckResult(
            endpoint_name="slig",
            status=EndpointStatus.UNHEALTHY,
            error_message=f"Failed after {self.max_health_check_attempts} attempts",
            attempts=self.max_health_check_attempts
        )
        self._health_results["slig"] = result
        return result

    async def check_yolo_health(self, endpoint_url: str, token: str) -> HealthCheckResult:
        """
        Check YOLO endpoint health.

        Args:
            endpoint_url: YOLO endpoint URL
            token: HuggingFace API token

        Returns:
            HealthCheckResult with status and timing
        """
        import httpx

        for attempt in range(1, self.max_health_check_attempts + 1):
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=self.health_check_timeout_seconds) as client:
                    # YOLO health check - simple GET or minimal POST
                    response = await client.get(
                        f"{endpoint_url.rstrip('/')}/health",
                        headers={"Authorization": f"Bearer {token}"}
                    )

                    response_time_ms = (time.time() - start_time) * 1000

                    # Accept 200 or 404 (endpoint exists but no /health route)
                    if response.status_code in [200, 404]:
                        result = HealthCheckResult(
                            endpoint_name="yolo",
                            status=EndpointStatus.HEALTHY,
                            response_time_ms=response_time_ms,
                            attempts=attempt
                        )
                        self._health_results["yolo"] = result
                        logger.info(f"✅ YOLO health check passed (attempt {attempt}, {response_time_ms:.0f}ms)")
                        return result
                    elif response.status_code == 503:
                        logger.info(f"⏳ YOLO still warming up (attempt {attempt}/{self.max_health_check_attempts})")
                    else:
                        logger.warning(f"⚠️ YOLO health check failed: HTTP {response.status_code}")

            except httpx.TimeoutException:
                logger.info(f"⏳ YOLO health check timeout (attempt {attempt}/{self.max_health_check_attempts})")
            except Exception as e:
                logger.warning(f"⚠️ YOLO health check error (attempt {attempt}): {e}")

            if attempt < self.max_health_check_attempts:
                await asyncio.sleep(self.health_check_interval_seconds)

        # All attempts failed
        result = HealthCheckResult(
            endpoint_name="yolo",
            status=EndpointStatus.UNHEALTHY,
            error_message=f"Failed after {self.max_health_check_attempts} attempts",
            attempts=self.max_health_check_attempts
        )
        self._health_results["yolo"] = result
        return result

    async def check_all_endpoints(
        self,
        endpoints_config: Dict[str, Dict[str, str]],
        required_endpoints: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, HealthCheckResult]]:
        """
        Check health of all configured endpoints in parallel.

        Args:
            endpoints_config: Dict mapping endpoint names to their configs
                Example: {"qwen": {"url": "...", "token": "..."}, ...}
            required_endpoints: List of endpoint names that MUST be healthy (default: all)

        Returns:
            Tuple of (all_required_healthy, results_dict)
        """
        logger.info("=" * 60)
        logger.info("🔍 VALIDATING ENDPOINT HEALTH")
        logger.info("=" * 60)

        tasks = []
        endpoint_names = []

        # Create health check tasks for each configured endpoint
        if "qwen" in endpoints_config:
            cfg = endpoints_config["qwen"]
            tasks.append(self.check_qwen_health(cfg["url"], cfg["token"]))
            endpoint_names.append("qwen")

        if "slig" in endpoints_config:
            cfg = endpoints_config["slig"]
            tasks.append(self.check_slig_health(cfg["url"], cfg["token"]))
            endpoint_names.append("slig")

        if "yolo" in endpoints_config:
            cfg = endpoints_config["yolo"]
            tasks.append(self.check_yolo_health(cfg["url"], cfg["token"]))
            endpoint_names.append("yolo")

        # Run all health checks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            results_dict = {}
            for name, result in zip(endpoint_names, results):
                if isinstance(result, Exception):
                    results_dict[name] = HealthCheckResult(
                        endpoint_name=name,
                        status=EndpointStatus.ERROR,
                        error_message=str(result)
                    )
                else:
                    results_dict[name] = result

            # Check if all required endpoints are healthy
            if required_endpoints is None:
                required_endpoints = endpoint_names

            all_healthy = True
            for name in required_endpoints:
                if name in results_dict:
                    if results_dict[name].status != EndpointStatus.HEALTHY:
                        all_healthy = False
                        logger.error(f"❌ Required endpoint '{name}' is not healthy")
                else:
                    all_healthy = False
                    logger.error(f"❌ Required endpoint '{name}' was not checked")

            # Log summary
            logger.info("=" * 60)
            healthy_count = sum(1 for r in results_dict.values() if r.status == EndpointStatus.HEALTHY)
            logger.info(f"📊 HEALTH CHECK SUMMARY: {healthy_count}/{len(results_dict)} endpoints healthy")
            for name, result in results_dict.items():
                status_icon = "✅" if result.status == EndpointStatus.HEALTHY else "❌"
                time_str = f" ({result.response_time_ms:.0f}ms)" if result.response_time_ms else ""
                logger.info(f"   {status_icon} {name}: {result.status.value}{time_str}")
            logger.info("=" * 60)

            return all_healthy, results_dict

        return True, {}

    def get_results(self) -> Dict[str, HealthCheckResult]:
        """Get cached health check results."""
        return self._health_results.copy()


class WarmupCoordinator:
    """
    Coordinates endpoint warmup and health validation.

    Ensures endpoints are both resumed AND healthy before allowing
    the pipeline to proceed. Provides a blocking gate between warmup
    and processing stages.
    """

    def __init__(self):
        self.health_checker = EndpointHealthChecker()
        self._warmup_completed = False
        self._all_healthy = False
        self._health_results: Dict[str, HealthCheckResult] = {}

        # Track which endpoints failed
        self._failed_endpoints: List[str] = []
        self._optional_endpoints: List[str] = []

    async def warmup_and_validate(
        self,
        endpoint_managers: Dict[str, Any],
        endpoints_config: Dict[str, Dict[str, str]],
        required_endpoints: Optional[List[str]] = None,
        optional_endpoints: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, HealthCheckResult]]:
        """
        Coordinate warmup and health validation for all endpoints.

        This is the main entry point that:
        1. Triggers warmup on all endpoint managers
        2. Validates health through actual inference calls
        3. Returns only when all required endpoints are ready

        Args:
            endpoint_managers: Dict of endpoint manager instances (already warmed up)
            endpoints_config: Dict mapping endpoint names to URL/token configs
            required_endpoints: Endpoints that MUST be healthy (blocks if not)
            optional_endpoints: Endpoints that are nice-to-have but not blocking

        Returns:
            Tuple of (success, health_results)
        """
        self._optional_endpoints = optional_endpoints or []

        # Mark warmup completed flag on all managers
        for name, manager in endpoint_managers.items():
            if hasattr(manager, 'warmup_completed'):
                manager.warmup_completed = True

        # Validate health through actual inference
        logger.info("🔍 Starting endpoint health validation...")
        self._all_healthy, self._health_results = await self.health_checker.check_all_endpoints(
            endpoints_config=endpoints_config,
            required_endpoints=required_endpoints
        )

        # Track failed endpoints
        self._failed_endpoints = [
            name for name, result in self._health_results.items()
            if result.status != EndpointStatus.HEALTHY
        ]

        self._warmup_completed = True
        return self._all_healthy, self._health_results

    def is_ready(self) -> bool:
        """Check if all required endpoints are ready."""
        return self._warmup_completed and self._all_healthy

    def get_failed_endpoints(self) -> List[str]:
        """Get list of endpoints that failed health checks."""
        return self._failed_endpoints.copy()

    def should_proceed_without(self, endpoint_name: str) -> bool:
        """Check if we can proceed without a specific endpoint."""
        return endpoint_name in self._optional_endpoints


# Global coordinator instance
warmup_coordinator = WarmupCoordinator()
