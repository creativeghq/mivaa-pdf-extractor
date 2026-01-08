"""
TogetherAI Rate Limit Configuration

This module defines rate limits for different TogetherAI build tiers.
Based on: https://docs.together.ai/docs/rate-limits

Build Tiers are determined by total spend:
- Tier 1: $5.00 spent
- Tier 2: $50.00 spent
- Tier 3: $100.00 spent
- Tier 4: $250.00 spent
- Tier 5: $1000.00 spent
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


# TogetherAI Build Tiers
TOGETHER_AI_TIERS: Dict[int, RateLimitTier] = {
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
        rerank_rpm=10_000_000  # Assuming unlimited or very high
    ),
}


def get_current_tier() -> RateLimitTier:
    """
    Get the current TogetherAI tier based on environment configuration.
    
    Returns:
        RateLimitTier: The current tier configuration
    """
    # Read tier from environment variable, default to Tier 1
    tier_number = int(os.getenv('TOGETHER_AI_TIER', '1'))
    
    # Validate tier number
    if tier_number not in TOGETHER_AI_TIERS:
        tier_number = 1
    
    return TOGETHER_AI_TIERS[tier_number]


def get_vision_concurrency_limit() -> int:
    """
    Calculate safe concurrency limit for vision model requests.

    Vision models are part of LLM rate limits. We want to stay well below
    the limit to account for:
    - Other concurrent API usage
    - Retry attempts
    - Burst traffic
    - Model-specific capacity constraints (especially for large models like Qwen3-VL-32B)

    Returns:
        int: Safe number of concurrent vision requests
    """
    tier = get_current_tier()

    # CONSERVATIVE: Use lower concurrency for Tier 1 to avoid 503 errors
    # Qwen3-VL-32B has capacity constraints that cause 503s at higher concurrency
    if tier.tier == 1:
        return 5  # Conservative limit for Tier 1 to prevent TogetherAI 503 errors

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
    
    Claude has separate rate limits, but we still want to be conservative.
    
    Returns:
        int: Safe number of concurrent Claude requests
    """
    # Claude typically has lower rate limits, keep conservative
    return 2


# Export current tier for easy access
CURRENT_TIER = get_current_tier()
VISION_CONCURRENCY = get_vision_concurrency_limit()
CLAUDE_CONCURRENCY = get_claude_concurrency_limit()


# Usage Examples and Documentation
"""
USAGE EXAMPLES:
===============

1. Setting the tier via environment variable:

   export TOGETHER_AI_TIER=1  # For Tier 1 ($5 spent)
   export TOGETHER_AI_TIER=2  # For Tier 2 ($50 spent)
   export TOGETHER_AI_TIER=3  # For Tier 3 ($100 spent)
   export TOGETHER_AI_TIER=4  # For Tier 4 ($250 spent)
   export TOGETHER_AI_TIER=5  # For Tier 5 ($1000 spent)

2. Using in code:

   from app.config.rate_limits import CURRENT_TIER, VISION_CONCURRENCY

   print(f"Current tier: {CURRENT_TIER.tier}")
   print(f"LLM RPM: {CURRENT_TIER.llm_rpm}")
   print(f"Safe concurrency: {VISION_CONCURRENCY}")

3. Systemd service configuration:

   Add to /etc/systemd/system/mivaa-pdf-extractor.service:

   [Service]
   Environment="TOGETHER_AI_TIER=1"

   Then reload:
   sudo systemctl daemon-reload
   sudo systemctl restart mivaa-pdf-extractor.service

RATE LIMIT CALCULATIONS:
========================

Vision Concurrency Formula:
- Take 60% of tier's LLM RPM (leave 40% headroom for retries/bursts)
- Assume 2 second average response time
- concurrent_limit = (RPM * 0.6 / 60) * 2
- Clamped between 2 and 20

Examples by tier:
- Tier 1 (600 RPM):  (600 * 0.6 / 60) * 2 = 12 concurrent
- Tier 2 (1800 RPM): (1800 * 0.6 / 60) * 2 = 20 concurrent (capped)
- Tier 3 (3000 RPM): (3000 * 0.6 / 60) * 2 = 20 concurrent (capped)
- Tier 4 (4500 RPM): (4500 * 0.6 / 60) * 2 = 20 concurrent (capped)
- Tier 5 (6000 RPM): (6000 * 0.6 / 60) * 2 = 20 concurrent (capped)

MONITORING:
===========

Check current configuration in logs:
   journalctl -u mivaa-pdf-extractor.service | grep "Rate Limiting Configuration"

Expected output:
   🎯 Rate Limiting Configuration:
      TogetherAI Tier: 1 ($5.0 spent)
      LLM Rate Limit: 600 RPM (10.0 RPS)
      Vision Concurrency: 12 concurrent requests
      Claude Concurrency: 2 concurrent requests
"""

