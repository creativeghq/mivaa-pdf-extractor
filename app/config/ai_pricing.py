"""
AI Model Pricing Configuration

Centralized pricing for all AI models used in the platform.
Prices are per million tokens (input/output) unless otherwise specified.

Last Updated: 2025-10-27
Sources:
- Anthropic: https://www.anthropic.com/pricing
- OpenAI: https://openai.com/api/pricing/
- TogetherAI: https://www.together.ai/pricing

IMPORTANT: Verify prices monthly and update this file.
"""

from typing import Dict, Optional
from datetime import datetime
from decimal import Decimal


class AIPricingConfig:
    """
    Centralized AI model pricing configuration.
    
    All prices are per million tokens (input/output) in USD.
    """
    
    # Last price verification date
    LAST_UPDATED = "2025-12-26"

    # Anthropic Claude Pricing (per 1M tokens)
    CLAUDE_PRICING = {
        "claude-haiku-4-5": {
            "input": Decimal("0.80"),
            "output": Decimal("4.00"),
            "last_verified": "2025-12-26",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-sonnet-4-5": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-12-26",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-opus-4-5": {
            "input": Decimal("15.00"),
            "output": Decimal("75.00"),
            "last_verified": "2025-12-26",
            "source": "https://www.anthropic.com/pricing"
        },
        # Previous generation models
        "claude-3-5-sonnet-20241022": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
        },
        "claude-sonnet-4-5-20250929": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-12-26",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-4-5-haiku-20250514": {
            "input": Decimal("0.80"),
            "output": Decimal("4.00"),
            "last_verified": "2025-12-26",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-4-5-sonnet-20250514": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-12-26",
            "source": "https://www.anthropic.com/pricing"
        }
    }
    
    # OpenAI GPT Pricing (per 1M tokens)
    GPT_PRICING = {
        "gpt-5": {
            "input": Decimal("5.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/",
            "note": "Estimated pricing - verify when GPT-5 is released"
        },
        "gpt-4o": {
            "input": Decimal("2.50"),
            "output": Decimal("10.00"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/"
        },
        "gpt-4": {
            "input": Decimal("30.00"),
            "output": Decimal("60.00"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/"
        },
        "gpt-4-turbo": {
            "input": Decimal("10.00"),
            "output": Decimal("30.00"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/"
        },
        "gpt-3.5-turbo": {
            "input": Decimal("0.50"),
            "output": Decimal("1.50"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/"
        }
    }
    
    # OpenAI Embedding Pricing (per 1M tokens)
    EMBEDDING_PRICING = {
        "text-embedding-3-small": {
            "input": Decimal("0.02"),
            "output": Decimal("0.00"),  # Embeddings don't have output tokens
            "last_verified": "2025-12-26",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 1536
        },
        "text-embedding-3-large": {
            "input": Decimal("0.13"),
            "output": Decimal("0.00"),
            "last_verified": "2025-12-26",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 3072
        },
        "text-embedding-ada-002": {
            "input": Decimal("0.10"),
            "output": Decimal("0.00"),
            "last_verified": "2025-12-26",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 1536
        }
    }

    # Voyage AI Embedding Pricing (per 1M tokens)
    VOYAGE_PRICING = {
        "voyage-3": {
            "input": Decimal("0.06"),
            "output": Decimal("0.00"),
            "last_verified": "2025-12-26",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 1024
        },
        "voyage-3-lite": {
            "input": Decimal("0.02"),
            "output": Decimal("0.00"),
            "last_verified": "2025-12-26",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 512
        },
        "voyage-large-2-instruct": {
            "input": Decimal("0.12"),
            "output": Decimal("0.00"),
            "last_verified": "2025-12-26",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 1024
        }
    }
    
    # OpenAI Vision/Image Pricing
    VISION_PRICING = {
        "clip-vit-large-patch14": {
            "per_image": Decimal("0.00"),  # Free via OpenAI CLIP
            "last_verified": "2025-10-27",
            "source": "OpenAI CLIP (open source)",
            "note": "Free when using OpenAI's CLIP model"
        },
        "gpt-4-vision": {
            "input": Decimal("10.00"),  # Same as GPT-4 Turbo
            "output": Decimal("30.00"),
            "per_image": Decimal("0.00765"),  # ~$0.00765 per image (1024x1024)
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/"
        }
    }
    


    # Qwen Vision Models (HuggingFace Endpoint) - per 1M tokens
    QWEN_PRICING = {
        "qwen3-vl-32b": {
            "input": Decimal("0.40"),
            "output": Decimal("0.40"),
            "last_verified": "2026-01-09",
            "source": "HuggingFace Inference Endpoint",
            "full_name": "Qwen/Qwen3-VL-32B-Instruct",
            "note": "Primary vision model via HuggingFace Endpoint (32B only, 8B removed)"
        }
    }

    # Visual Embedding Models (SLIG Cloud Endpoint)
    VISUAL_EMBEDDING_PRICING = {
        "slig-768d": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-09",
            "source": "SLIG Cloud Endpoint (HuggingFace)",
            "full_name": "SLIG (SigLIP2-based) 768D",
            "dimensions": 768,
            "note": "Primary visual embedding model via SLIG cloud endpoint (768D)"
        }
    }

    # Firecrawl Web Scraping Pricing
    # Note: Firecrawl uses tokens (1 credit = 15 tokens)
    # Pricing is credit-based, exact USD cost depends on plan
    FIRECRAWL_PRICING = {
        "firecrawl-scrape": {
            "cost_per_credit": Decimal("0.001"),  # Estimated $0.001 per credit
            "tokens_per_credit": 15,  # 1 Firecrawl credit = 15 tokens
            "last_verified": "2025-12-25",
            "source": "https://firecrawl.dev/pricing",
            "note": "Firecrawl API - pricing varies by plan. This is an estimate for tracking."
        }
    }
    
    @classmethod
    def get_model_pricing(cls, model: str, provider: Optional[str] = None) -> Dict[str, Decimal]:
        """
        Get pricing for a specific model.

        Args:
            model: Model name (e.g., 'claude-haiku-4-5', 'gpt-4o', 'qwen3-vl-32b')
            provider: Optional provider hint ('anthropic', 'openai', 'huggingface', 'firecrawl')

        Returns:
            Dict with 'input' and 'output' pricing per million tokens

        Raises:
            ValueError: If model pricing not found
        """
        # Try to find in all pricing dictionaries
        all_pricing = {
            **cls.CLAUDE_PRICING,
            **cls.GPT_PRICING,
            **cls.EMBEDDING_PRICING,
            **cls.VOYAGE_PRICING,
            **cls.VISION_PRICING,
            **cls.QWEN_PRICING,
            **cls.VISUAL_EMBEDDING_PRICING,
            **cls.FIRECRAWL_PRICING
        }

        if model in all_pricing:
            pricing = all_pricing[model]
            return {
                "input": pricing.get("input", Decimal("0.00")),
                "output": pricing.get("output", Decimal("0.00"))
            }

        # Try fuzzy matching for model variants
        model_lower = model.lower()
        for key, pricing in all_pricing.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return {
                    "input": pricing.get("input", Decimal("0.00")),
                    "output": pricing.get("output", Decimal("0.00"))
                }

        # Default fallback pricing (conservative estimate)
        return {
            "input": Decimal("3.00"),
            "output": Decimal("15.00")
        }
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider: Optional[str] = None
    ) -> Decimal:
        """
        Calculate cost for an AI call.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Optional provider hint
            
        Returns:
            Total cost in USD as Decimal
        """
        pricing = cls.get_model_pricing(model, provider)
        
        input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing["input"]
        output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing["output"]
        
        return input_cost + output_cost
    
    @classmethod
    def calculate_firecrawl_cost(
        cls,
        credits_used: int = 1,
        operation: str = "firecrawl-scrape"
    ) -> Decimal:
        """
        Calculate cost for Firecrawl operations.

        Args:
            credits_used: Number of Firecrawl credits consumed
            operation: Operation type (default: 'firecrawl-scrape')

        Returns:
            Total cost in USD as Decimal
        """
        pricing = cls.FIRECRAWL_PRICING.get(operation, cls.FIRECRAWL_PRICING["firecrawl-scrape"])
        cost_per_credit = pricing.get("cost_per_credit", Decimal("0.001"))

        return Decimal(credits_used) * cost_per_credit

    @classmethod
    def get_pricing_info(cls, model: str) -> Optional[Dict]:
        """
        Get full pricing information including metadata.

        Args:
            model: Model name

        Returns:
            Full pricing dict with metadata or None if not found
        """
        all_pricing = {
            **cls.CLAUDE_PRICING,
            **cls.GPT_PRICING,
            **cls.EMBEDDING_PRICING,
            **cls.VOYAGE_PRICING,
            **cls.VISION_PRICING,
            **cls.QWEN_PRICING,
            **cls.FIRECRAWL_PRICING
        }

        return all_pricing.get(model)
    
    @classmethod
    def verify_pricing_freshness(cls) -> Dict[str, any]:
        """
        Check if pricing data is fresh (< 30 days old).
        
        Returns:
            Dict with verification status and warnings
        """
        from datetime import datetime, timedelta
        
        last_updated = datetime.strptime(cls.LAST_UPDATED, "%Y-%m-%d")
        days_old = (datetime.now() - last_updated).days
        
        return {
            "last_updated": cls.LAST_UPDATED,
            "days_old": days_old,
            "is_fresh": days_old < 30,
            "warning": f"Pricing data is {days_old} days old. Consider updating." if days_old >= 30 else None
        }


# Singleton instance
ai_pricing = AIPricingConfig()


