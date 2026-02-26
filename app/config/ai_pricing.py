"""
AI Model Pricing Configuration

Centralized pricing for all AI models used in the platform.
Prices are per million tokens (input/output) unless otherwise specified.

Last Updated: 2026-01-24
Sources:
- Anthropic: https://www.anthropic.com/pricing
- OpenAI: https://openai.com/api/pricing/
- Voyage AI: https://docs.voyageai.com/docs/pricing
- HuggingFace: Time-based billing for inference endpoints

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
    LAST_UPDATED = "2026-01-24"

    # Platform markup multiplier (50% markup = 1.50)
    # Users are billed at raw_cost * MARKUP_MULTIPLIER
    MARKUP_MULTIPLIER = Decimal("1.50")

    # Anthropic Claude Pricing (per 1M tokens)
    CLAUDE_PRICING = {
        "claude-haiku-4-5": {
            "input": Decimal("1.00"),
            "output": Decimal("5.00"),
            "last_verified": "2026-01-24",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-sonnet-4-5": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2026-01-24",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-opus-4-5": {
            "input": Decimal("5.00"),
            "output": Decimal("25.00"),
            "last_verified": "2026-01-24",
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
            "last_verified": "2026-01-24",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-4-5-haiku-20250514": {
            "input": Decimal("1.00"),
            "output": Decimal("5.00"),
            "last_verified": "2026-01-24",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-4-5-sonnet-20250514": {
            "input": Decimal("3.00"),
            "output": Decimal("15.00"),
            "last_verified": "2026-01-24",
            "source": "https://www.anthropic.com/pricing"
        },
        "claude-haiku-4-5-20251001": {
            "input": Decimal("1.00"),
            "output": Decimal("5.00"),
            "last_verified": "2026-01-24",
            "source": "https://www.anthropic.com/pricing",
            "note": "Used in agent-chat edge function"
        }
    }
    
    # OpenAI GPT Pricing (per 1M tokens)
    # Only models actually used in the platform
    GPT_PRICING = {
        "gpt-5": {
            "input": Decimal("5.00"),
            "output": Decimal("15.00"),
            "last_verified": "2026-01-24",
            "source": "https://openai.com/api/pricing/",
            "note": "Used in GPT5Service for high-accuracy tasks"
        },
        "gpt-4o": {
            "input": Decimal("2.50"),
            "output": Decimal("10.00"),
            "last_verified": "2026-01-24",
            "source": "https://openai.com/api/pricing/",
            "note": "Main production model for discovery & metadata"
        },
        "gpt-4o-mini": {
            "input": Decimal("0.15"),
            "output": Decimal("0.60"),
            "last_verified": "2026-01-24",
            "source": "https://openai.com/api/pricing/",
            "note": "Cost-effective model for reranking & query expansion"
        }
    }
    
    # OpenAI Embedding Pricing (per 1M tokens)
    EMBEDDING_PRICING = {
        "text-embedding-3-small": {
            "input": Decimal("0.02"),
            "output": Decimal("0.00"),  # Embeddings don't have output tokens
            "last_verified": "2026-01-24",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 1536
        },
        "text-embedding-3-large": {
            "input": Decimal("0.13"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 3072
        },
        "text-embedding-ada-002": {
            "input": Decimal("0.10"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://openai.com/api/pricing/",
            "dimensions": 1536
        }
    }

    # Voyage AI Embedding Pricing (per 1M tokens)
    VOYAGE_PRICING = {
        "voyage-3.5": {
            "input": Decimal("0.06"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 1024,
            "note": "PRIMARY text embedding model"
        },
        "voyage-3.5-lite": {
            "input": Decimal("0.02"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 512,
            "note": "Cost-effective embedding model"
        },
        "voyage-3": {
            "input": Decimal("0.06"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 1024
        },
        "voyage-3-lite": {
            "input": Decimal("0.02"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 512
        },
        "voyage-large-2-instruct": {
            "input": Decimal("0.12"),
            "output": Decimal("0.00"),
            "last_verified": "2026-01-24",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 1024
        }
    }
    
    # OpenAI Vision/Image Pricing
    # Note: gpt-4o now handles vision, gpt-4-vision is deprecated
    VISION_PRICING = {
        "clip-vit-large-patch14": {
            "per_image": Decimal("0.00"),  # Free via OpenAI CLIP
            "last_verified": "2026-01-24",
            "source": "OpenAI CLIP (open source)",
            "note": "Free when using OpenAI's CLIP model"
        }
    }
    


    # ==========================================================================
    # HUGGINGFACE INFERENCE ENDPOINTS - TIME-BASED PRICING
    # All HuggingFace endpoints are billed by GPU compute time, not tokens
    # Cost = inference_seconds × (hourly_rate / 3600)
    # ==========================================================================

    # HuggingFace GPU Instance Pricing Reference
    HUGGINGFACE_GPU_RATES = {
        "nvidia-a10g": Decimal("1.30"),      # ~$1.30/hour (24GB VRAM)
        "nvidia-a100": Decimal("4.50"),      # ~$4.50/hour (40/80GB VRAM)
        "nvidia-l4": Decimal("0.80"),        # ~$0.80/hour (24GB VRAM)
        "nvidia-t4": Decimal("0.60"),        # ~$0.60/hour (16GB VRAM)
    }

    # Qwen Vision Model (A10G GPU)
    QWEN_PRICING = {
        "qwen3-vl-32b": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("1.30"),  # A10G GPU
            "gpu_type": "nvidia-a10g",
            "last_verified": "2026-01-23",
            "source": "HuggingFace Inference Endpoint",
            "full_name": "Qwen/Qwen3-VL-32B-Instruct",
            "namespace": "basiliskan",
            "service": "mh-qwen332binstruct",
            "note": "Primary vision model for material analysis"
        }
    }

    # SLIG Visual Embedding Model (L4 GPU)
    VISUAL_EMBEDDING_PRICING = {
        "slig-768d": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("0.80"),  # L4 GPU
            "gpu_type": "nvidia-l4",
            "last_verified": "2026-01-23",
            "source": "HuggingFace Inference Endpoint",
            "full_name": "SigLIP2 ViT-SO400M (SLIG)",
            "namespace": "basiliskan",
            "service": "mh-slig",
            "dimensions": 768,
            "note": "Visual embeddings: general, color, texture, style, material"
        },
        "siglip2-vit-so400m": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("0.80"),
            "gpu_type": "nvidia-l4",
            "last_verified": "2026-01-23",
            "source": "HuggingFace Inference Endpoint",
            "full_name": "SigLIP2 ViT-SO400M",
            "dimensions": 768,
            "note": "Alias for slig-768d"
        }
    }

    # YOLO Document Layout Detection Model (L4 GPU)
    YOLO_PRICING = {
        "yolo-docparser": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("0.80"),  # L4 GPU
            "gpu_type": "nvidia-l4",
            "last_verified": "2026-01-23",
            "source": "HuggingFace Inference Endpoint",
            "full_name": "YOLO DocParser Layout Detection",
            "note": "Document layout detection, structure analysis"
        }
    }

    # Chandra OCR Model (L4 GPU)
    OCR_PRICING = {
        "chandra-ocr": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("0.80"),  # L4 GPU
            "gpu_type": "nvidia-l4",
            "last_verified": "2026-01-23",
            "source": "HuggingFace Inference Endpoint",
            "full_name": "Chandra OCR Model",
            "note": "Optical character recognition for documents"
        }
    }

    # ==========================================================================
    # REPLICATE IMAGE GENERATION MODELS - PER-GENERATION PRICING
    # ==========================================================================

    REPLICATE_PRICING = {
        # Text-to-Image Models
        "flux-dev": {
            "cost_per_generation": Decimal("0.025"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "FLUX.1-dev high quality text-to-image"
        },
        "flux-schnell": {
            "cost_per_generation": Decimal("0.003"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "FLUX.1-schnell fast generation"
        },
        "sdxl": {
            "cost_per_generation": Decimal("0.01"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "Stable Diffusion XL"
        },
        "playground-v2.5": {
            "cost_per_generation": Decimal("0.01"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing"
        },
        "stable-diffusion-3": {
            "cost_per_generation": Decimal("0.055"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing"
        },
        # Image-to-Image Models (Interior Design)
        "comfyui-interior-remodel": {
            "cost_per_generation": Decimal("0.02"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "Interior remodel - WORKING"
        },
        "interiorly-gen1-dev": {
            "cost_per_generation": Decimal("0.015"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "Interiorly interior design - WORKING"
        },
        "designer-architecture": {
            "cost_per_generation": Decimal("0.018"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "Architecture design - WORKING"
        },
        "interior-v2": {
            "cost_per_generation": Decimal("0.02"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-24",
            "source": "https://replicate.com/pricing",
            "note": "Interior design v2"
        },
        "adirik-interior-design": {
            "cost_per_generation": Decimal("0.015"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-24",
            "source": "https://replicate.com/pricing",
            "note": "Adirik interior design model"
        },
        "interior-design-sdxl": {
            "cost_per_generation": Decimal("0.015"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-24",
            "source": "https://replicate.com/pricing",
            "note": "Interior design SDXL"
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

    # ==========================================================================
    # EXTERNAL SERVICE PRICING - PER-UNIT (non-AI third-party APIs)
    # Cost per single operation (message, query, enrichment, etc.)
    # Kept in sync with supabase/functions/_shared/credit-utils.ts
    # ==========================================================================

    EXTERNAL_SERVICE_PRICING = {
        "twilio-sms": {
            "cost_per_unit": Decimal("0.0079"),
            "unit": "message",
            "last_verified": "2026-02-14",
            "source": "https://www.twilio.com/en-us/sms/pricing/us",
            "note": "US domestic SMS, varies by country"
        },
        "twilio-whatsapp": {
            "cost_per_unit": Decimal("0.005"),
            "unit": "message",
            "last_verified": "2026-02-14",
            "source": "https://www.twilio.com/en-us/whatsapp/pricing",
            "note": "WhatsApp utility conversation, varies by template type"
        },
        "apollo-enrich": {
            "cost_per_unit": Decimal("0.05"),
            "unit": "enrichment",
            "last_verified": "2026-02-14",
            "source": "https://www.apollo.io/pricing",
            "note": "Company enrichment API call"
        },
        "apollo-people-match": {
            "cost_per_unit": Decimal("0.03"),
            "unit": "lookup",
            "last_verified": "2026-02-14",
            "source": "https://www.apollo.io/pricing",
            "note": "People match/email finder fallback"
        },
        "hunter-email-finder": {
            "cost_per_unit": Decimal("0.01"),
            "unit": "search",
            "last_verified": "2026-02-14",
            "source": "https://hunter.io/pricing",
            "note": "Single email finder"
        },
        "hunter-domain-search": {
            "cost_per_unit": Decimal("0.01"),
            "unit": "search",
            "last_verified": "2026-02-14",
            "source": "https://hunter.io/pricing",
            "note": "Domain-wide contact discovery"
        },
        "zerobounce-validate": {
            "cost_per_unit": Decimal("0.008"),
            "unit": "validation",
            "last_verified": "2026-02-14",
            "source": "https://www.zerobounce.net/email-validation-pricing",
            "note": "Single email validation"
        },
    }
    
    @classmethod
    def get_all_pricing(cls) -> Dict[str, Dict]:
        """Get all pricing dictionaries merged together."""
        return {
            **cls.CLAUDE_PRICING,
            **cls.GPT_PRICING,
            **cls.EMBEDDING_PRICING,
            **cls.VOYAGE_PRICING,
            **cls.VISION_PRICING,
            **cls.QWEN_PRICING,
            **cls.VISUAL_EMBEDDING_PRICING,
            **cls.YOLO_PRICING,
            **cls.OCR_PRICING,
            **cls.REPLICATE_PRICING,
            **cls.FIRECRAWL_PRICING,
            **cls.EXTERNAL_SERVICE_PRICING,
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
        all_pricing = cls.get_all_pricing()

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
        provider: Optional[str] = None,
        include_markup: bool = True
    ) -> Dict[str, Decimal]:
        """
        Calculate cost for an AI call with optional markup.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Optional provider hint
            include_markup: If True, returns billed cost with markup (default: True)

        Returns:
            Dict with 'raw_cost_usd', 'billed_cost_usd', 'markup_multiplier',
            'input_cost_usd', 'output_cost_usd', and 'credits_to_debit'
        """
        pricing = cls.get_model_pricing(model, provider)

        input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing["input"]
        output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing["output"]
        raw_cost = input_cost + output_cost

        # Apply markup for billing
        billed_cost = raw_cost * cls.MARKUP_MULTIPLIER if include_markup else raw_cost

        # Convert to credits (1 credit = $0.01)
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "raw_cost_usd": raw_cost,
            "markup_multiplier": cls.MARKUP_MULTIPLIER,
            "billed_cost_usd": billed_cost,
            "credits_to_debit": credits_to_debit
        }
    
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
    def calculate_external_service_cost(
        cls,
        service_name: str,
        units: int = 1,
        include_markup: bool = True
    ) -> Dict[str, Decimal]:
        """
        Calculate cost for external (non-AI) service operations.

        Args:
            service_name: Service identifier (e.g., 'twilio-sms', 'apollo-enrich')
            units: Number of operations performed
            include_markup: If True, applies 50% markup (default: True)

        Returns:
            Dict with raw_cost_usd, billed_cost_usd, credits_to_debit, units, unit_type

        Raises:
            ValueError: If service_name not found in EXTERNAL_SERVICE_PRICING
        """
        pricing = cls.EXTERNAL_SERVICE_PRICING.get(service_name)
        if not pricing:
            raise ValueError(
                f"Service {service_name} not found in EXTERNAL_SERVICE_PRICING. "
                f"Available: {list(cls.EXTERNAL_SERVICE_PRICING.keys())}"
            )

        cost_per_unit = pricing["cost_per_unit"]
        raw_cost = cost_per_unit * Decimal(units)

        billed_cost = raw_cost * cls.MARKUP_MULTIPLIER if include_markup else raw_cost
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "raw_cost_usd": raw_cost,
            "billed_cost_usd": billed_cost,
            "markup_multiplier": cls.MARKUP_MULTIPLIER,
            "credits_to_debit": credits_to_debit,
            "units": units,
            "unit_type": pricing["unit"],
            "cost_per_unit": cost_per_unit,
        }

    @classmethod
    def calculate_time_based_cost(
        cls,
        model: str,
        inference_seconds: float,
        include_markup: bool = True
    ) -> Dict[str, Decimal]:
        """
        Calculate cost for time-based models (HuggingFace Inference Endpoints).

        HuggingFace Inference Endpoints are billed by GPU compute time, not tokens.
        Cost = inference_seconds × (hourly_rate / 3600)

        Args:
            model: Model name (e.g., 'qwen3-vl-32b', 'slig-768d', 'yolo-docparser')
            inference_seconds: Time taken for inference in seconds
            include_markup: If True, applies 50% markup (default: True)

        Returns:
            Dict with raw_cost_usd, billed_cost_usd, credits_to_debit
        """
        # Find the model in any time-based pricing dictionary
        pricing = None
        for pricing_dict in [cls.QWEN_PRICING, cls.VISUAL_EMBEDDING_PRICING, cls.YOLO_PRICING, cls.OCR_PRICING]:
            if model in pricing_dict and pricing_dict[model].get("billing_type") == "time_based":
                pricing = pricing_dict[model]
                break

        if not pricing:
            raise ValueError(f"Model {model} is not configured for time-based billing")

        hourly_rate = pricing.get("hourly_rate_usd", Decimal("1.30"))
        gpu_type = pricing.get("gpu_type", "unknown")

        # Cost = seconds × (hourly_rate / 3600)
        raw_cost = Decimal(str(inference_seconds)) * (hourly_rate / Decimal("3600"))

        # Apply markup
        billed_cost = raw_cost * cls.MARKUP_MULTIPLIER if include_markup else raw_cost

        # Convert to credits (1 credit = $0.01)
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "raw_cost_usd": raw_cost,
            "billed_cost_usd": billed_cost,
            "markup_multiplier": cls.MARKUP_MULTIPLIER,
            "credits_to_debit": credits_to_debit,
            "inference_seconds": Decimal(str(inference_seconds)),
            "hourly_rate_usd": hourly_rate,
            "gpu_type": gpu_type
        }

    @classmethod
    def calculate_replicate_cost(
        cls,
        model: str,
        num_generations: int = 1,
        include_markup: bool = True
    ) -> Dict[str, Decimal]:
        """
        Calculate cost for Replicate image generation models.

        Args:
            model: Model name (e.g., 'flux-dev', 'comfyui-interior-remodel')
            num_generations: Number of images generated
            include_markup: If True, applies 50% markup (default: True)

        Returns:
            Dict with raw_cost_usd, billed_cost_usd, credits_to_debit
        """
        pricing = cls.REPLICATE_PRICING.get(model)

        if not pricing or pricing.get("billing_type") != "per_generation":
            raise ValueError(f"Model {model} is not configured for per-generation billing")

        cost_per_gen = pricing.get("cost_per_generation", Decimal("0.01"))
        raw_cost = cost_per_gen * Decimal(num_generations)

        # Apply markup
        billed_cost = raw_cost * cls.MARKUP_MULTIPLIER if include_markup else raw_cost

        # Convert to credits (1 credit = $0.01)
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "raw_cost_usd": raw_cost,
            "billed_cost_usd": billed_cost,
            "markup_multiplier": cls.MARKUP_MULTIPLIER,
            "credits_to_debit": credits_to_debit,
            "num_generations": num_generations,
            "cost_per_generation": cost_per_gen
        }

    @classmethod
    def is_time_based_model(cls, model: str) -> bool:
        """Check if a model uses time-based billing (HuggingFace endpoints)."""
        # Check all time-based pricing dictionaries
        time_based_dicts = [
            cls.QWEN_PRICING,
            cls.VISUAL_EMBEDDING_PRICING,
            cls.YOLO_PRICING,
            cls.OCR_PRICING
        ]
        for pricing_dict in time_based_dicts:
            if model in pricing_dict:
                return pricing_dict[model].get("billing_type") == "time_based"
        return False

    @classmethod
    def is_per_generation_model(cls, model: str) -> bool:
        """Check if a model uses per-generation billing (Replicate)."""
        return model in cls.REPLICATE_PRICING and cls.REPLICATE_PRICING[model].get("billing_type") == "per_generation"

    @classmethod
    def get_pricing_info(cls, model: str) -> Optional[Dict]:
        """
        Get full pricing information including metadata.

        Args:
            model: Model name

        Returns:
            Full pricing dict with metadata or None if not found
        """
        return cls.get_all_pricing().get(model)
    
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


