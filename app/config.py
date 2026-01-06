"""
Configuration Management for PDF Processing Service

This module provides centralized configuration management that bridges
existing extractor.py functionality with the production FastAPI structure.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, validator


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    This configuration bridges existing extractor.py functionality
    with the production FastAPI application structure.
    """
    
    # Application Settings
    app_name: str = Field(default="MIVAA - Material Intelligence Vision and Analysis Agent", env="APP_NAME")
    app_version: str = Field(default="2.5.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # API Settings
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    docs_url: str = Field(default="/docs", env="DOCS_URL")
    redoc_url: str = Field(default="/redoc", env="REDOC_URL")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: list = Field(default=["*"], env="CORS_METHODS")
    cors_headers: list = Field(default=["*"], env="CORS_HEADERS")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # PDF Processing Settings (compatible with existing extractor.py)
    output_dir: str = Field(default="output", env="OUTPUT_DIR")
    temp_dir: Optional[str] = Field(default=None, env="TEMP_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_extensions: list = Field(default=[".pdf"], env="ALLOWED_EXTENSIONS")
    
    # PyMuPDF4LLM Settings (existing extractor.py compatibility)
    default_image_format: str = Field(default="png", env="DEFAULT_IMAGE_FORMAT")
    default_image_quality: int = Field(default=95, env="DEFAULT_IMAGE_QUALITY")
    default_table_strategy: str = Field(default="fast", env="DEFAULT_TABLE_STRATEGY")
    write_images: bool = Field(default=True, env="WRITE_IMAGES")
    extract_tables: bool = Field(default=True, env="EXTRACT_TABLES")
    extract_images: bool = Field(default=True, env="EXTRACT_IMAGES")
    
    # Performance Settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes

    # ✅ NEW: Chunking Enhancement Feature Flags (NOW ENABLED BY DEFAULT)
    enable_boundary_detection: bool = Field(default=True, env="ENABLE_BOUNDARY_DETECTION")
    enable_semantic_chunking: bool = Field(default=True, env="ENABLE_SEMANTIC_CHUNKING")
    enable_context_enrichment: bool = Field(default=True, env="ENABLE_CONTEXT_ENRICHMENT")
    enable_metadata_first: bool = Field(default=True, env="ENABLE_METADATA_FIRST")
    enable_chunk_relationships: bool = Field(default=True, env="ENABLE_CHUNK_RELATIONSHIPS")
    fallback_on_error: bool = Field(default=True, env="FALLBACK_ON_ERROR")  # Always fallback to existing pipeline on error

    # Phase 1: Quick Wins
    enable_multi_query: bool = Field(default=False, env="ENABLE_MULTI_QUERY")
    enable_embedding_cache: bool = Field(default=False, env="ENABLE_EMBEDDING_CACHE")
    enable_relevance_truncation: bool = Field(default=False, env="ENABLE_RELEVANCE_TRUNCATION")
    enable_adaptive_chunk_sizing: bool = Field(default=False, env="ENABLE_ADAPTIVE_CHUNK_SIZING")

    # Phase 2: High Impact
    enable_cross_encoder_reranking: bool = Field(default=False, env="ENABLE_CROSS_ENCODER_RERANKING")
    enable_parent_child_chunking: bool = Field(default=False, env="ENABLE_PARENT_CHILD_CHUNKING")
    enable_ensemble_retrieval: bool = Field(default=False, env="ENABLE_ENSEMBLE_RETRIEVAL")
    enable_hyde: bool = Field(default=False, env="ENABLE_HYDE")

    # Phase 3: Advanced Features
    enable_semantic_embedding_chunking: bool = Field(default=False, env="ENABLE_SEMANTIC_EMBEDDING_CHUNKING")
    enable_llm_reranking: bool = Field(default=False, env="ENABLE_LLM_RERANKING")
    enable_sliding_window_retrieval: bool = Field(default=False, env="ENABLE_SLIDING_WINDOW_RETRIEVAL")

    # Phase 4: Research (Future)
    enable_late_interaction: bool = Field(default=False, env="ENABLE_LATE_INTERACTION")

    # RAG Performance Tuning Parameters
    # Multi-Query Expansion
    multi_query_variations: int = Field(default=3, env="MULTI_QUERY_VARIATIONS")

    # Embedding Cache
    embedding_cache_ttl: int = Field(default=86400, env="EMBEDDING_CACHE_TTL")  # 24 hours
    embedding_cache_max_size: int = Field(default=10000, env="EMBEDDING_CACHE_MAX_SIZE")

    # Relevance Truncation
    relevance_min_score: float = Field(default=0.7, env="RELEVANCE_MIN_SCORE")

    # Adaptive Chunk Sizing
    adaptive_chunk_min_size: int = Field(default=256, env="ADAPTIVE_CHUNK_MIN_SIZE")
    adaptive_chunk_max_size: int = Field(default=1500, env="ADAPTIVE_CHUNK_MAX_SIZE")
    adaptive_chunk_technical_threshold: int = Field(default=10, env="ADAPTIVE_CHUNK_TECHNICAL_THRESHOLD")
    adaptive_chunk_complexity_threshold: float = Field(default=0.7, env="ADAPTIVE_CHUNK_COMPLEXITY_THRESHOLD")

    # Cross-Encoder Reranking
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="CROSS_ENCODER_MODEL")
    cross_encoder_top_k: int = Field(default=10, env="CROSS_ENCODER_TOP_K")

    # Parent-Child Chunking
    parent_chunk_size: int = Field(default=2048, env="PARENT_CHUNK_SIZE")
    child_chunk_size: int = Field(default=256, env="CHILD_CHUNK_SIZE")
    child_chunk_overlap: int = Field(default=50, env="CHILD_CHUNK_OVERLAP")

    # Ensemble Retrieval
    ensemble_weights: Dict[str, float] = Field(
        default={"semantic": 0.4, "hybrid": 0.3, "multi_vector": 0.3},
        env="ENSEMBLE_WEIGHTS"
    )

    # HyDE (Hypothetical Document Embeddings)
    hyde_cache_enabled: bool = Field(default=True, env="HYDE_CACHE_ENABLED")
    hyde_cache_ttl: int = Field(default=3600, env="HYDE_CACHE_TTL")  # 1 hour

    # Semantic Embedding Chunking
    semantic_similarity_threshold: float = Field(default=0.8, env="SEMANTIC_SIMILARITY_THRESHOLD")

    # LLM Reranking
    llm_reranking_top_k: int = Field(default=5, env="LLM_RERANKING_TOP_K")
    llm_reranking_model: str = Field(default="gpt-4o-mini", env="LLM_RERANKING_MODEL")

    # Sliding Window Retrieval
    sliding_window_max_tokens: int = Field(default=4000, env="SLIDING_WINDOW_MAX_TOKENS")
    sliding_window_simple_top_k: int = Field(default=3, env="SLIDING_WINDOW_SIMPLE_TOP_K")
    sliding_window_medium_top_k: int = Field(default=5, env="SLIDING_WINDOW_MEDIUM_TOP_K")
    sliding_window_complex_top_k: int = Field(default=10, env="SLIDING_WINDOW_COMPLEX_TOP_K")

    # Security Settings
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    
    # JWT Authentication Settings
    jwt_secret_key: str = Field(default="", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    jwt_issuer: str = Field(default="material-kai-platform", env="JWT_ISSUER")
    jwt_audience: str = Field(default="mivaa-pdf-extractor", env="JWT_AUDIENCE")
    
    # Supabase Settings (aligned with platform standards)
    supabase_url: str = Field(default="", env="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", env="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="", env="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field(default="pdf-documents", env="SUPABASE_STORAGE_BUCKET")
    
    # Database Settings
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    database_timeout: int = Field(default=30, env="DATABASE_TIMEOUT")

    # OpenAI API Settings (Legacy - kept for backward compatibility)
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")
    openai_timeout: int = Field(default=30, env="OPENAI_TIMEOUT")

    # Voyage AI Settings (Primary embedding provider)
    # API key set via GitHub Secrets: VOYAGE_API_KEY
    voyage_api_key: str = Field(default="", env="VOYAGE_API_KEY")
    voyage_model: str = Field(default="voyage-3.5", env="VOYAGE_MODEL")
    voyage_embedding_dimension: int = Field(default=1024, env="VOYAGE_EMBEDDING_DIMENSION")
    voyage_timeout: int = Field(default=30, env="VOYAGE_TIMEOUT")
    voyage_enabled: bool = Field(default=True, env="VOYAGE_ENABLED")
    voyage_fallback_to_openai: bool = Field(default=True, env="VOYAGE_FALLBACK_TO_OPENAI")

    # RAG Settings (model-agnostic - works with Claude 4.5 + Direct Vector DB)

    rag_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="RAG_EMBEDDING_MODEL"
    )
    rag_llm_model: str = Field(
        default="claude-sonnet-4.5",
        env="RAG_LLM_MODEL"
    )
    rag_chunk_size: int = Field(
        default=1024,
        env="RAG_CHUNK_SIZE"
    )
    rag_chunk_overlap: int = Field(
        default=200,
        env="RAG_CHUNK_OVERLAP"
    )
    rag_similarity_top_k: int = Field(
        default=5,
        env="RAG_SIMILARITY_TOP_K"
    )
    rag_storage_dir: str = Field(
        default="./data/rag",
        env="RAG_STORAGE_DIR"
    )
    rag_enable: bool = Field(
        default=True,
        env="RAG_ENABLE"
    )
    
    # Multi-modal Processing Settings
    enable_multimodal: bool = Field(
        default=True,
        env="ENABLE_MULTIMODAL"
    )
    multimodal_llm_model: str = Field(
        default="gpt-4o",
        env="MULTIMODAL_LLM_MODEL"
    )
    multimodal_max_tokens: int = Field(
        default=4096,
        env="MULTIMODAL_MAX_TOKENS"
    )
    multimodal_temperature: float = Field(
        default=0.1,
        env="MULTIMODAL_TEMPERATURE"
    )
    multimodal_image_detail: str = Field(
        default="high",
        env="MULTIMODAL_IMAGE_DETAIL"
    )
    multimodal_batch_size: int = Field(
        default=5,
        env="MULTIMODAL_BATCH_SIZE"
    )
    multimodal_timeout: int = Field(
        default=60,
        env="MULTIMODAL_TIMEOUT"
    )
    
    # OCR Processing Settings
    ocr_enabled: bool = Field(
        default=True,
        env="OCR_ENABLED"
    )
    ocr_language: str = Field(
        default="en",
        env="OCR_LANGUAGE"
    )
    ocr_confidence_threshold: float = Field(
        default=0.6,
        env="OCR_CONFIDENCE_THRESHOLD"
    )
    ocr_engine: str = Field(
        default="easyocr",
        env="OCR_ENGINE"
    )
    ocr_gpu_enabled: bool = Field(
        default=False,
        env="OCR_GPU_ENABLED"
    )
    ocr_preprocessing_enabled: bool = Field(
        default=True,
        env="OCR_PREPROCESSING_ENABLED"
    )
    ocr_deskew_enabled: bool = Field(
        default=True,
        env="OCR_DESKEW_ENABLED"
    )
    ocr_noise_removal_enabled: bool = Field(
        default=True,
        env="OCR_NOISE_REMOVAL_ENABLED"
    )
    
    # Image Processing Settings
    image_processing_enabled: bool = Field(
        default=True,
        env="IMAGE_PROCESSING_ENABLED"
    )
    image_analysis_model: str = Field(
        default="gpt-4o",
        env="IMAGE_ANALYSIS_MODEL"
    )
    image_resize_max_width: int = Field(
        default=2048,
        env="IMAGE_RESIZE_MAX_WIDTH"
    )
    image_resize_max_height: int = Field(
        default=2048,
        env="IMAGE_RESIZE_MAX_HEIGHT"
    )
    image_compression_quality: int = Field(
        default=85,
        env="IMAGE_COMPRESSION_QUALITY"
    )
    image_format_conversion: str = Field(
        default="JPEG",
        env="IMAGE_FORMAT_CONVERSION"
    )
    
    # TogetherAI Settings
    together_api_key: str = Field(
        default="",
        env="TOGETHER_API_KEY"
    )
    together_base_url: str = Field(
        default="https://api.together.xyz/v1/chat/completions",
        env="TOGETHER_BASE_URL"
    )
    together_model: str = Field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        env="TOGETHER_MODEL"
    )
    together_validation_model: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        env="TOGETHER_VALIDATION_MODEL"
    )
    together_max_tokens: int = Field(
        default=4096,
        env="TOGETHER_MAX_TOKENS"
    )
    together_temperature: float = Field(
        default=0.1,
        env="TOGETHER_TEMPERATURE"
    )
    together_timeout: int = Field(
        default=60,
        env="TOGETHER_TIMEOUT"
    )
    together_enabled: bool = Field(
        default=True,
        env="TOGETHER_ENABLED"
    )
    together_rate_limit_rpm: int = Field(
        default=200,
        env="TOGETHER_RATE_LIMIT_RPM"
    )
    together_rate_limit_tpm: int = Field(
        default=20000,
        env="TOGETHER_RATE_LIMIT_TPM"
    )
    together_retry_attempts: int = Field(
        default=3,
        env="TOGETHER_RETRY_ATTEMPTS"
    )
    together_retry_delay: float = Field(
        default=1.0,
        env="TOGETHER_RETRY_DELAY"
    )

    # Anthropic Claude Settings
    anthropic_api_key: str = Field(
        default="",
        env="ANTHROPIC_API_KEY"
    )
    anthropic_model_classification: str = Field(
        default="claude-haiku-4-5-20251001",
        env="ANTHROPIC_MODEL_CLASSIFICATION"
    )
    anthropic_model_validation: str = Field(
        default="claude-sonnet-4-5-20250929",
        env="ANTHROPIC_MODEL_VALIDATION"
    )
    anthropic_model_enrichment: str = Field(
        default="claude-sonnet-4-5-20250929",
        env="ANTHROPIC_MODEL_ENRICHMENT"
    )
    anthropic_max_tokens: int = Field(
        default=4096,
        env="ANTHROPIC_MAX_TOKENS"
    )
    anthropic_temperature: float = Field(
        default=0.1,
        env="ANTHROPIC_TEMPERATURE"
    )
    anthropic_timeout: int = Field(
        default=60,
        env="ANTHROPIC_TIMEOUT"
    )
    anthropic_enabled: bool = Field(
        default=True,
        env="ANTHROPIC_ENABLED"
    )
    # NEW: Claude for RAG Queries
    anthropic_model_rag_query: str = Field(
        default="claude-sonnet-4-5-20250929",
        env="ANTHROPIC_MODEL_RAG_QUERY",
        description="Claude model for RAG question answering"
    )

    # Visual Embedding Models (SigLIP2 primary, CLIP fallback) - NOW CONFIGURABLE
    visual_embedding_primary_model: str = Field(
        default="google/siglip2-so400m-patch14-384",
        env="VISUAL_EMBEDDING_PRIMARY_MODEL",
        description="Primary visual embedding model (SigLIP2)"
    )
    visual_embedding_fallback_model: str = Field(
        default="openai/clip-vit-base-patch32",
        env="VISUAL_EMBEDDING_FALLBACK_MODEL",
        description="Fallback visual embedding model (CLIP)"
    )
    visual_embedding_dimensions: int = Field(
        default=1152,
        env="VISUAL_EMBEDDING_DIMENSIONS",
        description="Dimensions of visual embeddings (1152 for SigLIP, 512 for CLIP)"
    )
    visual_embedding_enabled: bool = Field(
        default=True,
        env="VISUAL_EMBEDDING_ENABLED",
        description="Enable visual embedding generation"
    )

    # Visual Embedding Mode: "local" or "remote"
    visual_embedding_mode: str = Field(
        default="local",
        env="VISUAL_EMBEDDING_MODE",
        description="Visual embedding mode: 'local' (run SigLIP locally) or 'remote' (use Hugging Face API)"
    )

    # Hugging Face API Settings (for remote visual embeddings)
    huggingface_api_key: str = Field(
        default="",
        env="HUGGINGFACE_API_KEY",
        description="Hugging Face API key for remote visual embeddings"
    )
    huggingface_api_url: str = Field(
        default="https://api-inference.huggingface.co",
        env="HUGGINGFACE_API_URL",
        description="Hugging Face Inference API base URL"
    )
    huggingface_siglip_model: str = Field(
        default="google/siglip2-so400m-patch14-384",
        env="HUGGINGFACE_SIGLIP_MODEL",
        description="SigLIP v2 model ID on Hugging Face"
    )
    huggingface_batch_size: int = Field(
        default=10,
        env="HUGGINGFACE_BATCH_SIZE",
        description="Batch size for Hugging Face API calls (max images per request)"
    )
    huggingface_timeout: int = Field(
        default=60,
        env="HUGGINGFACE_TIMEOUT",
        description="Timeout for Hugging Face API calls in seconds"
    )
    huggingface_max_retries: int = Field(
        default=3,
        env="HUGGINGFACE_MAX_RETRIES",
        description="Maximum retry attempts for Hugging Face API calls"
    )

    # Voyage AI Settings (Text Embeddings - Primary Provider)
    voyage_api_key: str = Field(
        default="",
        env="VOYAGE_API_KEY",
        description="Voyage AI API key for text embeddings"
    )
    voyage_model: str = Field(
        default="voyage-3.5",
        env="VOYAGE_MODEL",
        description="Voyage AI model for text embeddings"
    )
    voyage_embedding_dimension: int = Field(
        default=1024,
        env="VOYAGE_EMBEDDING_DIMENSION",
        description="Voyage AI embedding dimensions"
    )
    voyage_timeout: int = Field(
        default=30,
        env="VOYAGE_TIMEOUT",
        description="Voyage AI API timeout in seconds"
    )
    voyage_enabled: bool = Field(
        default=True,
        env="VOYAGE_ENABLED",
        description="Enable Voyage AI embeddings (fallback to OpenAI if disabled)"
    )
    voyage_fallback_to_openai: bool = Field(
        default=True,
        env="VOYAGE_FALLBACK_TO_OPENAI",
        description="Fallback to OpenAI if Voyage AI fails"
    )

    # Document Chunking Models
    chunking_primary_model: str = Field(
        default="Qwen/Qwen3-VL-32B-Instruct",
        env="CHUNKING_PRIMARY_MODEL",
        description="Primary model for chunking"
    )
    chunking_quality_threshold: float = Field(
        default=0.7,
        env="CHUNKING_QUALITY_THRESHOLD",
        ge=0.0,
        le=1.0,
        description="Quality threshold for chunk validation (0.0-1.0)"
    )

    # Material Kai Vision Platform Settings (disabled by default)
    material_kai_platform_url: str = Field(
        default="",  # Disabled by default - set to enable external platform integration
        env="MATERIAL_KAI_PLATFORM_URL"
    )
    material_kai_api_key: str = Field(
        default="",
        env="MATERIAL_KAI_API_KEY"
    )
    material_kai_workspace_id: str = Field(
        default="",
        env="MATERIAL_KAI_WORKSPACE_ID"
    )
    material_kai_service_name: str = Field(
        default="mivaa-pdf-extractor",
        env="MATERIAL_KAI_SERVICE_NAME"
    )
    material_kai_sync_enabled: bool = Field(
        default=False,  # Disabled by default
        env="MATERIAL_KAI_SYNC_ENABLED"
    )
    material_kai_real_time_enabled: bool = Field(
        default=False,  # Disabled by default
        env="MATERIAL_KAI_REAL_TIME_ENABLED"
    )
    material_kai_batch_size: int = Field(
        default=10,
        env="MATERIAL_KAI_BATCH_SIZE"
    )
    material_kai_retry_attempts: int = Field(
        default=3,
        env="MATERIAL_KAI_RETRY_ATTEMPTS"
    )
    material_kai_timeout: int = Field(
        default=30,
        env="MATERIAL_KAI_TIMEOUT"
    )

    # Price Monitoring Settings
    firecrawl_api_key: str = Field(
        default="",
        env="FIRECRAWL_API_KEY"
    )
    google_shopping_api_key: str = Field(
        default="",
        env="GOOGLE_SHOPPING_API_KEY"
    )
    google_shopping_cx: str = Field(
        default="",
        env="GOOGLE_SHOPPING_CX"
    )

    # Development and Testing Settings
    environment: str = Field(
        default="production",
        env="ENVIRONMENT"
    )
    enable_test_authentication: bool = Field(
        default=False,
        env="ENABLE_TEST_AUTHENTICATION"
    )
    test_api_keys: str = Field(
        default="",
        env="TEST_API_KEYS"  # Comma-separated list of test API keys
    )

    # Sentry Error Tracking and Monitoring Settings
    sentry_dsn: str = Field(
        default="",
        env="SENTRY_DSN"
    )
    sentry_environment: str = Field(
        default="development",
        env="SENTRY_ENVIRONMENT"
    )
    sentry_traces_sample_rate: float = Field(
        default=0.1,
        env="SENTRY_TRACES_SAMPLE_RATE"
    )
    sentry_profiles_sample_rate: float = Field(
        default=0.1,
        env="SENTRY_PROFILES_SAMPLE_RATE"
    )
    sentry_enabled: bool = Field(
        default=False,
        env="SENTRY_ENABLED"
    )
    sentry_release: Optional[str] = Field(
        default=None,
        env="SENTRY_RELEASE"
    )
    sentry_server_name: Optional[str] = Field(
        default=None,
        env="SENTRY_SERVER_NAME"
    )
    
    @validator("cors_origins", "cors_methods", "cors_headers", "allowed_extensions", pre=True)
    @classmethod
    def parse_list_from_string(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
            """Validate log level."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in valid_levels:
                raise ValueError(f"Log level must be one of: {valid_levels}")
            return v.upper()

    @validator("default_image_format")
    @classmethod
    def validate_image_format(cls, v):
        """Validate image format."""
        valid_formats = ["png", "jpg", "jpeg", "webp"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Image format must be one of: {valid_formats}")
        return v.lower()

    @validator("default_table_strategy")
    @classmethod
    def validate_table_strategy(cls, v):
        """Validate table extraction strategy."""
        valid_strategies = ["fast", "accurate"]
        if v.lower() not in valid_strategies:
            raise ValueError(f"Table strategy must be one of: {valid_strategies}")
        return v.lower()

    @validator("multimodal_image_detail")
    @classmethod
    def validate_multimodal_image_detail(cls, v):
        """Validate multi-modal image detail level."""
        valid_details = ["low", "high", "auto"]
        if v.lower() not in valid_details:
            raise ValueError(f"Multi-modal image detail must be one of: {valid_details}")
        return v.lower()

    @validator("ocr_engine")
    @classmethod
    def validate_ocr_engine(cls, v):
        """Validate OCR engine selection."""
        valid_engines = ["easyocr", "pytesseract", "both"]
        if v.lower() not in valid_engines:
            raise ValueError(f"OCR engine must be one of: {valid_engines}")
        return v.lower()

    @validator("ocr_language")
    @classmethod
    def validate_ocr_language(cls, v):
        """Validate OCR language code."""
        # Common language codes - can be extended as needed
        valid_languages = [
            "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
        if v.lower() not in valid_languages:
            raise ValueError(f"OCR language must be one of: {valid_languages}")
        return v.lower()

    @validator("image_format_conversion")
    @classmethod
    def validate_image_format_conversion(cls, v):
        """Validate image format for conversion."""
        valid_formats = ["JPEG", "PNG", "WEBP", "TIFF"]
        if v.upper() not in valid_formats:
            raise ValueError(f"Image format conversion must be one of: {valid_formats}")
        return v.upper()

    @validator("temp_dir", pre=True)
    @classmethod
    def set_temp_dir(cls, v):
        """Set default temp directory if not provided."""
        if v is None:
            import tempfile
            return tempfile.gettempdir()
        return v
    
    def get_output_path(self) -> Path:
        """Get the output directory path."""
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def get_temp_path(self) -> Path:
        """Get the temporary directory path."""
        temp_path = Path(self.temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)
        return temp_path
    
    def is_file_allowed(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.allowed_extensions
    
    def is_file_size_valid(self, file_size: int) -> bool:
        """Check if file size is within limits."""
        return file_size <= self.max_file_size
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary."""
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": self.log_format,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": self.log_level,
                },
            },
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console"],
                    "level": self.log_level,
                    "propagate": False,
                },
                "uvicorn": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
        
        # Add file handler if log file is specified
        if self.log_file:
            config["handlers"]["file"] = {
                "class": "logging.FileHandler",
                "filename": self.log_file,
                "formatter": "default",
                "level": self.log_level,
            }
            # Add file handler to all loggers
            for logger_config in config["loggers"].values():
                logger_config["handlers"].append("file")
        
        return config
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            "allow_origins": self.cors_origins,
            "allow_methods": self.cors_methods,
            "allow_headers": self.cors_headers,
            "allow_credentials": True,
        }

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG configuration (model-agnostic).

        This provides all necessary configuration for the RAG service
        including embedding models, LLM settings, storage options, and vision model integration.
        Works with any vision/LLM models - just change the model names in config.
        """
        return {
            "embedding_model": self.rag_embedding_model,
            "llm_model": self.rag_llm_model,
            "chunk_size": self.rag_chunk_size,
            "chunk_overlap": self.rag_chunk_overlap,
            "similarity_top_k": self.rag_similarity_top_k,
            "storage_dir": self.rag_storage_dir,
            "enable_rag": self.rag_enable,
            # TogetherAI/Qwen Vision Model Configuration
            "together_model": self.together_model,
            "together_validation_model": self.together_validation_model,
            "together_api_key": self.together_api_key,
            "together_base_url": self.together_base_url,
            "together_max_tokens": self.together_max_tokens,
            "together_temperature": self.together_temperature,
            "together_enabled": self.together_enabled,
        }

    def get_material_kai_config(self) -> Dict[str, Any]:
        """
        Get Material Kai Vision Platform configuration.
        
        This provides all necessary configuration for the Material Kai service
        including platform URL, authentication, and integration settings.
        """
        return {
            "platform_url": self.material_kai_platform_url,
            "api_key": self.material_kai_api_key,
            "workspace_id": self.material_kai_workspace_id,
            "service_name": self.material_kai_service_name,
            "sync_enabled": self.material_kai_sync_enabled,
            "real_time_enabled": self.material_kai_real_time_enabled,
            "batch_size": self.material_kai_batch_size,
            "retry_attempts": self.material_kai_retry_attempts,
            "timeout": self.material_kai_timeout,
        }
    
    def get_sentry_config(self) -> Dict[str, Any]:
        """
        Get Sentry error tracking and monitoring configuration.
        
        This provides all necessary configuration for Sentry integration
        including DSN, environment, sampling rates, and optional metadata.
        """
        return {
            "dsn": self.sentry_dsn,
            "environment": self.sentry_environment,
            "traces_sample_rate": self.sentry_traces_sample_rate,
            "profiles_sample_rate": self.sentry_profiles_sample_rate,
            "enabled": self.sentry_enabled,
            "release": self.sentry_release,
            "server_name": self.sentry_server_name,
        }
    
    def get_multimodal_config(self) -> Dict[str, Any]:
        """
        Get multi-modal processing configuration.

        This provides all necessary configuration for multi-modal document
        processing including LLM settings, image processing, and performance tuning.
        """
        return {
            "enabled": self.enable_multimodal,
            "enable_multimodal": self.enable_multimodal,
            "image_processing_enabled": self.image_processing_enabled,
            "llm_model": self.multimodal_llm_model,
            "max_tokens": self.multimodal_max_tokens,
            "temperature": self.multimodal_temperature,
            "image_detail": self.multimodal_image_detail,
            "batch_size": self.multimodal_batch_size,
            "timeout": self.multimodal_timeout,
        }
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """
        Get OCR processing configuration.
        
        This provides all necessary configuration for OCR text extraction
        including engine selection, language settings, and preprocessing options.
        """
        return {
            "enabled": self.ocr_enabled,
            "language": self.ocr_language,
            "confidence_threshold": self.ocr_confidence_threshold,
            "engine": self.ocr_engine,
            "gpu_enabled": self.ocr_gpu_enabled,
            "preprocessing_enabled": self.ocr_preprocessing_enabled,
            "deskew_enabled": self.ocr_deskew_enabled,
            "noise_removal_enabled": self.ocr_noise_removal_enabled,
        }
    
    def get_image_processing_config(self) -> Dict[str, Any]:
        """
        Get image processing configuration.
        
        This provides all necessary configuration for image processing
        including analysis models, resizing, compression, and format conversion.
        """
        return {
            "enabled": self.image_processing_enabled,
            "analysis_model": self.image_analysis_model,
            "resize_max_width": self.image_resize_max_width,
            "resize_max_height": self.image_resize_max_height,
            "compression_quality": self.image_compression_quality,
            "format_conversion": self.image_format_conversion,
        }
    
    @validator("together_model")
    @classmethod
    def validate_together_model(cls, v):
        """Validate TogetherAI vision model name."""
        valid_models = [
            # Qwen Vision Models
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct"
        ]
        if v not in valid_models:
            raise ValueError(f"TogetherAI model must be one of: {valid_models}")
        return v

    def get_jwt_config(self) -> Dict[str, Any]:
        """
        Get JWT authentication configuration.
        
        This provides all necessary configuration for JWT authentication
        including secret key, algorithm, token expiration, and validation settings.
        """
        return {
            "secret_key": self.jwt_secret_key,
            "algorithm": self.jwt_algorithm,
            "access_token_expire_minutes": self.jwt_access_token_expire_minutes,
            "refresh_token_expire_days": self.jwt_refresh_token_expire_days,
            "issuer": self.jwt_issuer,
            "audience": self.jwt_audience,
        }
    
    def get_together_ai_config(self) -> Dict[str, Any]:
        """
        Get TogetherAI configuration.
        
        This provides all necessary configuration for TogetherAI integration
        including API settings, model configuration, rate limiting, and retry logic.
        """
        return {
            "api_key": self.together_api_key,
            "base_url": self.together_base_url,
            "model": self.together_model,
            "max_tokens": self.together_max_tokens,
            "temperature": self.together_temperature,
            "timeout": self.together_timeout,
            "enabled": self.together_enabled,
            "rate_limit_rpm": self.together_rate_limit_rpm,
            "rate_limit_tpm": self.together_rate_limit_tpm,
            "retry_attempts": self.together_retry_attempts,
            "retry_delay": self.together_retry_delay,
        }
    
    class Config:
        """Pydantic configuration."""
        # Load from environment variables only (GitHub Secrets, not .env files)
        case_sensitive = False


# Global settings instance
settings = Settings()


def configure_logging():
    """Configure application logging using the settings."""
    import logging.config

    logging_config = settings.get_logging_config()
    logging.config.dictConfig(logging_config)

    # Add Supabase logging handler to root logger
    try:
        from app.utils.supabase_logging_handler import SupabaseLoggingHandler

        root_logger = logging.getLogger()
        supabase_handler = SupabaseLoggingHandler(
            batch_size=10,
            flush_interval=5.0,
            level=logging.INFO  # Only log INFO and above to database
        )
        root_logger.addHandler(supabase_handler)

        # Log startup information
        logger = logging.getLogger(__name__)
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Log level set to: {settings.log_level}")
        logger.info(f"Output directory: {settings.get_output_path()}")
        logger.info(f"Temp directory: {settings.get_temp_path()}")
        logger.info("Supabase logging handler initialized")

    except Exception as e:
        # Don't crash if Supabase logging fails
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize Supabase logging handler: {e}")
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Log level set to: {settings.log_level}")
        logger.info(f"Output directory: {settings.get_output_path()}")
        logger.info(f"Temp directory: {settings.get_temp_path()}")


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings():
    """Reload settings from environment variables."""
    global settings
    settings = Settings()
    configure_logging()
    return settings


# NOTE: Logging is initialized in main.py after all imports are complete
# Do NOT call configure_logging() here - it will fail because Supabase client
# is not initialized yet at module import time
