"""
AI Model Pricing Configuration

Centralized pricing for all AI models used in the platform.
Prices are per million tokens (input/output) unless otherwise specified.

Last Updated: 2026-04-24
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
    LAST_UPDATED = "2026-04-24"

    # Platform markup multiplier (50% markup = 1.50)
    # Users are billed at raw_cost * MARKUP_MULTIPLIER
    MARKUP_MULTIPLIER = Decimal("1.50")

    # DB-driven pricing overlay (audit #217 H5). The `ai_model_pricing` admin table is
    # authoritative: when a model_key row exists it overrides the hardcoded dicts below
    # (which remain as a fallback so billing never breaks on a missing/unreachable row).
    # Per-model `markup_multiplier` is honored too. Cached per-process with a short TTL
    # so per-call billing stays cheap.
    _DB_CACHE_TTL_SECONDS = 300
    _db_pricing_cache: Optional[Dict[str, Dict]] = None
    _db_pricing_cache_ts: float = 0.0

    @classmethod
    def _get_db_pricing(cls) -> Dict[str, Dict]:
        """Load active rows from ai_model_pricing keyed by lowercased model_key.
        Returns {} on any error so callers fall back to the hardcoded dicts."""
        import time as _time
        now = _time.time()
        if cls._db_pricing_cache is not None and (now - cls._db_pricing_cache_ts) < cls._DB_CACHE_TTL_SECONDS:
            return cls._db_pricing_cache
        try:
            from app.services.core.supabase_client import get_supabase_client
            client = get_supabase_client().client
            rows = client.table('ai_model_pricing').select(
                'model_key, billing_type, input_price_per_million, output_price_per_million, '
                'cost_per_generation, cost_per_unit, markup_multiplier, is_active'
            ).eq('is_active', True).execute()
            cache: Dict[str, Dict] = {}
            for r in (rows.data or []):
                key = (r.get('model_key') or '').strip().lower()
                if not key:
                    continue
                cache[key] = {
                    'input': Decimal(str(r.get('input_price_per_million') or '0')),
                    'output': Decimal(str(r.get('output_price_per_million') or '0')),
                    'cost_per_generation': Decimal(str(r.get('cost_per_generation') or '0')),
                    'cost_per_unit': Decimal(str(r.get('cost_per_unit') or '0')),
                    'markup': Decimal(str(r.get('markup_multiplier') or cls.MARKUP_MULTIPLIER)),
                    'billing_type': r.get('billing_type'),
                }
            cls._db_pricing_cache = cache
            cls._db_pricing_cache_ts = now
            return cache
        except Exception:
            # Network/DB error — keep any stale cache, else empty (→ hardcoded fallback).
            return cls._db_pricing_cache or {}

    @classmethod
    def _db_lookup(cls, model: str) -> Optional[Dict]:
        """Exact-then-fuzzy match a model against the DB pricing cache."""
        db = cls._get_db_pricing()
        if not db:
            return None
        ml = (model or '').lower()
        if ml in db:
            return db[ml]
        for key, val in db.items():
            if key in ml or ml in key:
                return val
        return None

    @classmethod
    def get_model_markup(cls, model: str) -> Decimal:
        """Per-model markup from the DB row if present, else the platform default."""
        row = cls._db_lookup(model)
        if row and row.get('markup'):
            return row['markup']
        return cls.MARKUP_MULTIPLIER

    # Anthropic Claude Pricing (per 1M tokens) — canonical 3 latest-tier models
    CLAUDE_PRICING = {
        "claude-opus-4-8": {
            "input": Decimal("15.00"),
            "output": Decimal("75.00"),
            "last_verified": "2026-04-18",
            "source": "https://www.anthropic.com/pricing",
            "note": "Default primary model — agent chat, discovery, metadata extraction, consensus validation"
        },
        "claude-haiku-4-5": {
            "input": Decimal("1.00"),
            "output": Decimal("5.00"),
            "last_verified": "2026-04-18",
            "source": "https://www.anthropic.com/pricing",
            "note": "Default Haiku — background agents, query parsing, reranking"
        }
    }

    # OpenAI Embedding Pricing (per 1M tokens) — embeddings only, chat models removed
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
        }
    }

    # Voyage AI Embedding Pricing (per 1M tokens) — voyage-4 is sole production embedder.
    VOYAGE_PRICING = {
        "voyage-4": {
            "input": Decimal("0.06"),
            "output": Decimal("0.00"),
            "last_verified": "2026-04-19",
            "source": "https://docs.voyageai.com/docs/pricing",
            "dimensions": 1024,
            "note": "Sole text embedding model — 1024D"
        }
    }
    
    # Vision/Image Pricing — Claude Opus/Haiku now handle vision; OpenAI vision removed.
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

    # Qwen Vision Model — REMOVED 2026-05-01.
    # All vision tasks moved to Anthropic Claude Opus 4.7. The QWEN_PRICING
    # dict is kept as an empty placeholder so any downstream code that
    # iterates over `[QWEN_PRICING, VISUAL_EMBEDDING_PRICING, ...]` still
    # works without raising KeyError on the dict itself.
    QWEN_PRICING: dict = {}

    # SLIG Visual Embedding Model (Modal GPU)
    VISUAL_EMBEDDING_PRICING = {
        "slig-768d": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("1.00"),  # flat $1 / GPU-hour (Modal)
            "gpu_type": "nvidia-a10g",
            "last_verified": "2026-06-14",
            "source": "Modal (app: slig)",
            "full_name": "SigLIP2-base-patch16-512 (SLIG)",
            "service": "slig",
            "dimensions": 768,
            "note": "Visual embeddings: general, color, texture, style, material. Migrated off HuggingFace 2026-06-14 (native 768D, no projection head)."
        },
        "siglip2-base-patch16-512": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("1.00"),  # flat $1 / GPU-hour (Modal)
            "gpu_type": "nvidia-a10g",
            "last_verified": "2026-06-14",
            "source": "Modal (app: slig)",
            "full_name": "SigLIP2-base-patch16-512",
            "dimensions": 768,
            "note": "Alias for slig-768d"
        }
    }

    # PaddleOCR-VL Structural Pass — two-stage (PP-DocLayoutV2 + 0.9B VLM) layout
    # + OCR + figure boxes per page (GPU, Modal).
    PADDLEOCR_PRICING = {
        "paddleocr-vl": {
            "input": Decimal("0.00"),
            "output": Decimal("0.00"),
            "billing_type": "time_based",
            "hourly_rate_usd": Decimal("1.00"),  # flat $1 / GPU-hour / model (self-hosted)  # GPU container (Modal L4)
            "gpu_type": "nvidia-l4",
            "last_verified": "2026-06-13",
            "source": "Modal",
            "full_name": "PaddleOCR-VL Structural Pass (PaddlePaddle/PaddleOCR-VL)",
            "note": "Layout regions + OCR text + figure boxes per page"
        }
    }

    # ==========================================================================
    # REPLICATE IMAGE GENERATION MODELS - PER-GENERATION PRICING
    # ==========================================================================

    REPLICATE_PRICING = {
        # Text-to-Image Models
        "flux-2-pro": {
            "cost_per_generation": Decimal("0.05"),
            "billing_type": "per_generation",
            "last_verified": "2026-03-30",
            "source": "https://replicate.com/pricing",
            "note": "FLUX.2 Pro production-grade text-to-image (4MP, photorealistic)"
        },
        "flux-dev": {
            "cost_per_generation": Decimal("0.025"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "FLUX.1-dev legacy text-to-image"
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
        # Keyed by the model "id" used in interior_design_routes (sd3), not the
        # Replicate model path (stability-ai/stable-diffusion-3) — the AI call
        # logger logs model.get("id"), so the pricing key must match the id.
        "sd3": {
            "cost_per_generation": Decimal("0.055"),
            "billing_type": "per_generation",
            "last_verified": "2026-01-23",
            "source": "https://replicate.com/pricing",
            "note": "Stable Diffusion 3 (stability-ai/stable-diffusion-3)"
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
            "cost_per_credit": Decimal("0.001"),  # Estimate — actual cost depends on plan
            "tokens_per_credit": 15,  # 1 Firecrawl credit = 15 tokens
            "last_verified": "2026-04-24",
            "source": "https://firecrawl.dev/pricing",
            "note": "ESTIMATE — Firecrawl plan-dependent. Re-verify against active plan when usage grows."
        }
    }

    # ==========================================================================
    # EXTERNAL SERVICE PRICING - PER-UNIT (non-AI third-party APIs)
    # Cost per single operation (message, query, enrichment, etc.)
    #
    # CANONICAL SOURCE: ai_model_pricing table
    #   (billing_type='per_unit' AND category='external_service')
    # Editable from the admin UI at /admin/agent-configs → AI Model Pricing.
    # The TypeScript edge-function billing path (credit-utils.ts) reads from
    # the DB at runtime with a 5-minute cache.
    #
    # The dict below is a FALLBACK for offline / standalone Python use only.
    # If you need to bill from Python, query the DB instead — these values
    # may drift from the live table if admins edit prices via the UI.
    # ==========================================================================

    EXTERNAL_SERVICE_PRICING = {
        "zernio-whatsapp": {
            "cost_per_unit": Decimal("0.005"),
            "unit": "message",
            "last_verified": "2026-06-08",
            "source": "https://zernio.com/whatsapp",
            "note": "WhatsApp conversation, Meta pass-through pricing via Zernio (varies by template category/country)"
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
            **cls.EMBEDDING_PRICING,
            **cls.VOYAGE_PRICING,
            **cls.VISION_PRICING,
            **cls.QWEN_PRICING,
            **cls.VISUAL_EMBEDDING_PRICING,
            **cls.PADDLEOCR_PRICING,
            **cls.REPLICATE_PRICING,
            **cls.FIRECRAWL_PRICING,
            **cls.EXTERNAL_SERVICE_PRICING,
        }

    @classmethod
    def get_model_pricing(cls, model: str, provider: Optional[str] = None) -> Dict[str, Decimal]:
        """
        Get pricing for a specific model.

        Args:
            model: Model name (e.g., 'claude-haiku-4-5', 'claude-opus-4-8', 'voyage-4')
            provider: Optional provider hint ('anthropic', 'openai', 'huggingface', 'firecrawl')

        Returns:
            Dict with 'input' and 'output' pricing per million tokens

        Raises:
            ValueError: If model pricing not found
        """
        # DB overlay first (audit #217 H5): admin-edited prices win over the hardcoded
        # defaults. Falls through to the static dicts when no row matches.
        db_row = cls._db_lookup(model)
        if db_row is not None and (db_row["input"] or db_row["output"]):
            return {"input": db_row["input"], "output": db_row["output"]}

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

        # Apply per-model markup (DB row if present, else platform default) — audit #217 H5
        markup = cls.get_model_markup(model)
        billed_cost = raw_cost * markup if include_markup else raw_cost

        # Convert to credits (1 credit = $0.01)
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "raw_cost_usd": raw_cost,
            "markup_multiplier": markup,
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
            model: HuggingFace endpoint model name (e.g., 'slig-768d', 'paddleocr-vl')
            inference_seconds: Time taken for inference in seconds
            include_markup: If True, applies 50% markup (default: True)

        Returns:
            Dict with raw_cost_usd, billed_cost_usd, credits_to_debit
        """
        # Find the model in any time-based pricing dictionary
        pricing = None
        for pricing_dict in [cls.QWEN_PRICING, cls.VISUAL_EMBEDDING_PRICING, cls.PADDLEOCR_PRICING]:
            if model in pricing_dict and pricing_dict[model].get("billing_type") == "time_based":
                pricing = pricing_dict[model]
                break

        if not pricing:
            raise ValueError(f"Model {model} is not configured for time-based billing")

        hourly_rate = pricing.get("hourly_rate_usd", Decimal("1.00"))
        gpu_type = pricing.get("gpu_type", "unknown")

        # Cost = seconds × (hourly_rate / 3600). The caller sums one row per GPU
        # call, so the per-job total is (total GPU-seconds across ALL parallel
        # containers) × hourly_rate — i.e. it already accounts for scaling: more
        # containers doing more work ⇒ more summed seconds ⇒ proportionally more cost.
        raw_cost = Decimal(str(inference_seconds)) * (hourly_rate / Decimal("3600"))

        # GPU endpoints (SLIG, PaddleOCR — self-hosted/Modal at a flat $/GPU-hour)
        # are a DIRECT infra cost, not a resold token API, so the stated hourly
        # rate IS the cost — no platform markup. billed == raw here.
        billed_cost = raw_cost

        # Convert to credits (1 credit = $0.01)
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "raw_cost_usd": raw_cost,
            "billed_cost_usd": billed_cost,
            "markup_multiplier": Decimal("1.0"),
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
        # DB overlay first (audit #217 H5): admin-edited per-generation price + markup win.
        db_row = cls._db_lookup(model)
        if db_row is not None and db_row.get("cost_per_generation"):
            cost_per_gen = db_row["cost_per_generation"]
            markup = db_row.get("markup") or cls.MARKUP_MULTIPLIER
        else:
            pricing = cls.REPLICATE_PRICING.get(model)
            if not pricing or pricing.get("billing_type") != "per_generation":
                raise ValueError(f"Model {model} is not configured for per-generation billing")
            cost_per_gen = pricing.get("cost_per_generation", Decimal("0.01"))
            markup = cls.MARKUP_MULTIPLIER

        raw_cost = cost_per_gen * Decimal(num_generations)

        # Apply markup
        billed_cost = raw_cost * markup if include_markup else raw_cost

        # Convert to credits (1 credit = $0.01)
        credits_to_debit = billed_cost * Decimal("100")

        return {
            "raw_cost_usd": raw_cost,
            "billed_cost_usd": billed_cost,
            "markup_multiplier": markup,
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
            cls.PADDLEOCR_PRICING,
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


