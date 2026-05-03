"""
Vision / HuggingFace Endpoint Rate Limit Configuration

This module defines rate limits for HuggingFace dedicated inference endpoints
(SLIG, YOLO DocParser, Chandra OCR) and Anthropic vision tiers. Tiers are
based on usage capacity and determine concurrency limits.

Tiers:
- Tier 1: Default (HuggingFace dedicated endpoints + entry-level Anthropic)
- Tier 2: Medium capacity
- Tier 3: High capacity
- Tier 4: Very high capacity
- Tier 5: Maximum capacity
"""

from dataclasses import dataclass
from typing import Dict
import os


@dataclass
class RateLimitTier:
    """Rate limit configuration for a specific tier."""
    tier: int
    total_spend: float
    llm_rpm: int  # Requests Per Minute for LLMs (including vision models)
    embeddings_rpm: int  # Requests Per Minute for embeddings
    rerank_rpm: int  # Requests Per Minute for rerank

    @property
    def llm_rps(self) -> float:
        """Requests Per Second for LLMs."""
        return self.llm_rpm / 60.0

    @property
    def embeddings_rps(self) -> float:
        """Requests Per Second for embeddings."""
        return self.embeddings_rpm / 60.0

    @property
    def rerank_rps(self) -> float:
        """Requests Per Second for rerank."""
        return self.rerank_rpm / 60.0


# Vision Model Rate Tiers
VISION_RATE_TIERS: Dict[int, RateLimitTier] = {
    1: RateLimitTier(
        tier=1,
        total_spend=5.00,
        llm_rpm=600,
        embeddings_rpm=3000,
        rerank_rpm=500_000
    ),
    2: RateLimitTier(
        tier=2,
        total_spend=50.00,
        llm_rpm=1800,
        embeddings_rpm=5000,
        rerank_rpm=1_500_000
    ),
    3: RateLimitTier(
        tier=3,
        total_spend=100.00,
        llm_rpm=3000,
        embeddings_rpm=5000,
        rerank_rpm=2_000_000
    ),
    4: RateLimitTier(
        tier=4,
        total_spend=250.00,
        llm_rpm=4500,
        embeddings_rpm=10_000,
        rerank_rpm=3_000_000
    ),
    5: RateLimitTier(
        tier=5,
        total_spend=1000.00,
        llm_rpm=6000,
        embeddings_rpm=10_000,
        rerank_rpm=10_000_000
    ),
}


def get_current_tier() -> RateLimitTier:
    """
    Get the current vision model tier based on environment configuration.

    Returns:
        RateLimitTier: The current tier configuration
    """
    # Read tier from environment variable
    tier_number = int(os.getenv('VISION_MODEL_TIER', '1'))

    # Validate tier number
    if tier_number not in VISION_RATE_TIERS:
        tier_number = 1

    return VISION_RATE_TIERS[tier_number]


def get_vision_concurrency_limit() -> int:
    """
    Calculate safe concurrency limit for vision model requests.

    Vision models are part of LLM rate limits. We want to stay well below
    the limit to account for:
    - Other concurrent API usage
    - Retry attempts
    - Burst traffic
    - Model-specific capacity constraints (especially for large vision models on a single GPU replica)

    Returns:
        int: Safe number of concurrent vision requests
    """
    tier = get_current_tier()

    # For HuggingFace dedicated endpoints (SLIG / YOLO / Chandra) we can use
    # higher concurrency — they handle 10 concurrent requests well.
    if tier.tier == 1:
        return 10

    # Use 60% of available RPM to leave headroom for higher tiers
    # Convert to concurrent requests assuming ~2s average response time
    safe_rpm = tier.llm_rpm * 0.6
    avg_response_time_seconds = 2.0

    # Concurrent requests = (RPM / 60) * avg_response_time
    concurrent_limit = int((safe_rpm / 60.0) * avg_response_time_seconds)

    # Ensure minimum of 2 and maximum of 20
    return max(2, min(20, concurrent_limit))


def get_claude_concurrency_limit() -> int:
    """
    Get safe concurrency limit for Claude API requests.

    Post-Qwen-removal (2026-05-01) Stage 3 image classification AND
    vision_analysis both run through Anthropic Claude Opus 4.7. The
    previous hardcoded `2` was set when Claude was a rare fallback —
    leaving it at 2 now serializes ~80 images/product behind a 2-wide
    gate at ~10s per Opus call, eating the full 600s per-product budget
    on classification alone (incident: VALENOVA, 2026-05-03, job
    acff9ebb-8daf-48f0-acd3-4f77308faf8b).

    Anthropic Tier 1 = 600 RPM = 10 RPS. With ~10s avg vision-call
    latency, Little's Law allows ~100 in-flight before saturation;
    we cap at 10 by default to leave headroom for product_discovery,
    icon extraction, and retries running in parallel.

    Override via `CLAUDE_VISION_CONCURRENCY` env var to tune per-tier
    without redeploy.
    """
    return int(os.getenv('CLAUDE_VISION_CONCURRENCY', '10'))


# Export current tier for easy access
CURRENT_TIER = get_current_tier()
VISION_CONCURRENCY = get_vision_concurrency_limit()
CLAUDE_CONCURRENCY = get_claude_concurrency_limit()
