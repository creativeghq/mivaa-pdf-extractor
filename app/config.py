"""
Configuration Management for PDF Processing Service

This module provides centralized configuration management that bridges
existing extractor.py functionality with the production FastAPI structure.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    This configuration bridges existing extractor.py functionality
    with the production FastAPI application structure.
    """
    
    # Application Settings
    app_name: str = Field(default="PDF Processing Service", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
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
    
    # Security Settings
    max_requests_per_minute: int = Field(default=60, env="MAX_REQUESTS_PER_MINUTE")
    
    # JWT Authentication Settings
    jwt_secret_key: str = Field(default="", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    jwt_issuer: str = Field(default="material-kai-platform", env="JWT_ISSUER")
    jwt_audience: str = Field(default="mivaa-pdf-extractor", env="JWT_AUDIENCE")
    
    # Supabase Settings
    supabase_url: str = Field(default="", env="SUPABASE_URL")
    supabase_key: str = Field(default="", env="SUPABASE_KEY")
    supabase_service_key: str = Field(default="", env="SUPABASE_SERVICE_KEY")
    supabase_storage_bucket: str = Field(default="pdf-documents", env="SUPABASE_STORAGE_BUCKET")
    
    # Database Settings
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    database_timeout: int = Field(default=30, env="DATABASE_TIMEOUT")
    
    # LlamaIndex RAG Settings
    llamaindex_embedding_model: str = Field(
        default="text-embedding-ada-002",
        env="LLAMAINDEX_EMBEDDING_MODEL"
    )
    llamaindex_llm_model: str = Field(
        default="gpt-3.5-turbo",
        env="LLAMAINDEX_LLM_MODEL"
    )
    llamaindex_chunk_size: int = Field(
        default=1024,
        env="LLAMAINDEX_CHUNK_SIZE"
    )
    llamaindex_chunk_overlap: int = Field(
        default=200,
        env="LLAMAINDEX_CHUNK_OVERLAP"
    )
    llamaindex_similarity_top_k: int = Field(
        default=5,
        env="LLAMAINDEX_SIMILARITY_TOP_K"
    )
    llamaindex_storage_dir: str = Field(
        default="./data/llamaindex",
        env="LLAMAINDEX_STORAGE_DIR"
    )
    llamaindex_enable_rag: bool = Field(
        default=True,
        env="LLAMAINDEX_ENABLE_RAG"
    )
    
    # Material Kai Vision Platform Settings
    material_kai_platform_url: str = Field(
        default="https://api.materialkai.vision",
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
        default=True,
        env="MATERIAL_KAI_SYNC_ENABLED"
    )
    material_kai_real_time_enabled: bool = Field(
        default=True,
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
    def parse_list_from_string(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("default_image_format")
    def validate_image_format(cls, v):
        """Validate image format."""
        valid_formats = ["png", "jpg", "jpeg", "webp"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Image format must be one of: {valid_formats}")
        return v.lower()
    
    @validator("default_table_strategy")
    def validate_table_strategy(cls, v):
        """Validate table extraction strategy."""
        valid_strategies = ["fast", "accurate"]
        if v.lower() not in valid_strategies:
            raise ValueError(f"Table strategy must be one of: {valid_strategies}")
        return v.lower()
    
    @validator("temp_dir", pre=True)
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
    def get_extractor_config(self) -> Dict[str, Any]:
        """
        Get configuration compatible with existing extractor.py functions.
        
        This ensures backward compatibility with the existing PyMuPDF4LLM
        implementation while providing new configuration options.
        """
        return {
            "output_dir": str(self.get_output_path()),
            "temp_dir": str(self.get_temp_path()),
            "image_format": self.default_image_format,
            "image_quality": self.default_image_quality,
            "table_strategy": self.default_table_strategy,
            "write_images": self.write_images,
            "extract_tables": self.extract_tables,
            "extract_images": self.extract_images,
            "max_file_size": self.max_file_size,
            "allowed_extensions": self.allowed_extensions,
        }
    def get_llamaindex_config(self) -> Dict[str, Any]:
        """
        Get LlamaIndex RAG configuration.
        
        This provides all necessary configuration for the LlamaIndex service
        including embedding models, LLM settings, and storage options.
        """
        return {
            "embedding_model": self.llamaindex_embedding_model,
            "llm_model": self.llamaindex_llm_model,
            "chunk_size": self.llamaindex_chunk_size,
            "chunk_overlap": self.llamaindex_chunk_overlap,
            "similarity_top_k": self.llamaindex_similarity_top_k,
            "storage_dir": self.llamaindex_storage_dir,
            "enable_rag": self.llamaindex_enable_rag,
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
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def configure_logging():
    """Configure application logging using the settings."""
    import logging.config
    
    logging_config = settings.get_logging_config()
    logging.config.dictConfig(logging_config)
    
    # Log startup information
    logger = logging.getLogger(__name__)
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


# Initialize logging on module import
configure_logging()