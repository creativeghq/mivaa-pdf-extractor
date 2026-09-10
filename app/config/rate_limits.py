"""Vision / Modal GPU Endpoint Rate Limit Configuration"""

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
    """Calculate safe concurrency limit for vision model requests.

    Returns:
        int: Safe number of concurrent vision requests
    """
    tier = get_current_tier()

    # For the SLIG endpoint and the PaddleOCR structural pass we can use
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
    """Get safe concurrency limit for Claude API requests."""
    return int(os.getenv('CLAUDE_VISION_CONCURRENCY', '10'))


# Export current tier for easy access
CURRENT_TIER = get_current_tier()
VISION_CONCURRENCY = get_vision_concurrency_limit()
CLAUDE_CONCURRENCY = get_claude_concurrency_limit()
