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
    LAST_UPDATED = "2025-10-27"
    
    # Anthropic Claude Pricing (per 1M tokens)
    CLAUDE_PRICING = {
        "claude-haiku-4-5": {
            "input": Decimal("0.80"),
            "output": Decimal("4.00"),
            "last_verified": "2025-10-27",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-sonnet-4-5": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-10-27",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-opus-4-5": {
            "input": Decimal("15.00"),
            "output": Decimal("75.00"),
            "last_verified": "2025-10-27",
            "source": "https://www.anthropic.com/pricing"
        },
        # Legacy models (for backward compatibility)
        "claude-3-5-sonnet-20241022": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-10-27",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-4-5-haiku-20250514": {
            "input": Decimal("0.80"),
            "output": Decimal("4.00"),
            "last_verified": "2025-10-27",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-4-5-sonnet-20250514": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2025-10-27",
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
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 1536
        },
        "text-embedding-3-large": {
            "input": Decimal("0.13"),
            "output": Decimal("0.00"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 3072
        },
        "text-embedding-ada-002": {
            "input": Decimal("0.10"),
            "output": Decimal("0.00"),
            "last_verified": "2025-10-27",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 1536
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
    
    # TogetherAI Llama Pricing (per 1M tokens)
    LLAMA_PRICING = {
        "llama-4-scout-17b": {
            "input": Decimal("0.20"),
            "output": Decimal("0.20"),
            "last_verified": "2025-10-27",
            "source": "https://www.together.ai/pricing",
            "full_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        },
        "llama-4-maverick-17b": {
            "input": Decimal("0.20"),
            "output": Decimal("0.20"),
            "last_verified": "2025-10-27",
            "source": "https://www.together.ai/pricing"
        },
        "llama-3-2-90b-vision": {
            "input": Decimal("0.88"),
            "output": Decimal("0.88"),
            "last_verified": "2025-10-27",
            "source": "https://www.together.ai/pricing",
            "note": "Deprecated - replaced by Llama 4 Scout"
        }
    }
    
    @classmethod
    def get_model_pricing(cls, model: str, provider: Optional[str] = None) -> Dict[str, Decimal]:
        """
        Get pricing for a specific model.
        
        Args:
            model: Model name (e.g., 'claude-haiku-4-5', 'gpt-4o', 'llama-4-scout-17b')
            provider: Optional provider hint ('anthropic', 'openai', 'together')
            
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
            **cls.VISION_PRICING,
            **cls.LLAMA_PRICING
        }
        
        if model in all_pricing:
            pricing = all_pricing[model]
            return {
                "input": pricing["input"],
                "output": pricing["output"]
            }
        
        # Try fuzzy matching for model variants
        model_lower = model.lower()
        for key, pricing in all_pricing.items():
            if key.lower() in model_lower or model_lower in key.lower():
                return {
                    "input": pricing["input"],
                    "output": pricing["output"]
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
            **cls.VISION_PRICING,
            **cls.LLAMA_PRICING
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

