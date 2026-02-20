"""
Validation configuration management for the PDF2Markdown microservice.

This module provides centralized configuration for validation middleware,
including settings for security, performance, monitoring, and error handling.
"""

import os
from typing import Dict, List, Set, Optional, Any
from functools import lru_cache
import re

from pydantic import BaseModel, Field, field_validator
from app.config import settings


class SecurityConfig(BaseModel):
    """Security-related validation configuration."""
    
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting for requests"
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum requests per rate limit window"
    )
    rate_limit_window: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit window in seconds"
    )
    
    enable_input_sanitization: bool = Field(
        default=True,
        description="Enable input sanitization for security threats"
    )
    blocked_patterns: List[str] = Field(
        default=[
            r"<script[^>]*>.*?</script>",  # XSS prevention
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"on\w+\s*=",  # Event handlers
            r"expression\s*\(",  # CSS expressions
            r"@import",  # CSS imports
            r"<iframe[^>]*>",  # Iframe injection
            r"<object[^>]*>",  # Object injection
            r"<embed[^>]*>",  # Embed injection
        ],
        description="Regex patterns for blocked malicious content"
    )
    
    max_request_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        ge=1024,  # 1KB minimum
        le=500 * 1024 * 1024,  # 500MB maximum
        description="Maximum request size in bytes"
    )
    max_json_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum JSON nesting depth"
    )
    max_array_length: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum array length in JSON"
    )
    
    @field_validator('blocked_patterns')
    @classmethod
    def compile_patterns(cls, v):
        """Compile regex patterns for better performance."""
        compiled_patterns = []
        for pattern in v:
            try:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return compiled_patterns


class PerformanceConfig(BaseModel):
    """Performance-related validation configuration."""
    
    enable_caching: bool = Field(
        default=True,
        description="Enable validation result caching"
    )
    cache_ttl: int = Field(
        default=300,  # 5 minutes
        ge=60,  # 1 minute minimum
        le=3600,  # 1 hour maximum
        description="Cache time-to-live in seconds"
    )
    max_cache_size: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum number of cached validation results"
    )
    
    enable_compression: bool = Field(
        default=True,
        description="Enable response compression for large payloads"
    )
    compression_threshold: int = Field(
        default=1024,  # 1KB
        ge=100,
        le=10240,  # 10KB
        description="Minimum response size for compression in bytes"
    )
    
    async_validation: bool = Field(
        default=False,
        description="Enable asynchronous validation for non-blocking operations"
    )
    validation_timeout: float = Field(
        default=5.0,
        ge=0.1,
        le=30.0,
        description="Validation timeout in seconds"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""
    
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    log_validation_errors: bool = Field(
        default=True,
        description="Log validation errors for debugging"
    )
    log_performance_metrics: bool = Field(
        default=True,
        description="Log performance metrics"
    )
    log_security_violations: bool = Field(
        default=True,
        description="Log security violations"
    )
    
    slow_request_threshold: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Threshold for slow request logging in seconds"
    )
    metrics_retention_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 1 week
        description="How long to retain metrics in memory (hours)"
    )
    
    enable_health_checks: bool = Field(
        default=True,
        description="Enable validation middleware health checks"
    )


class ContentTypeConfig(BaseModel):
    """Content type validation configuration."""
    
    validate_content_type: bool = Field(
        default=True,
        description="Enable content type validation"
    )
    allowed_content_types: Set[str] = Field(
        default={
            "application/json",
            "multipart/form-data",
            "application/x-www-form-urlencoded",
            "text/plain",
            "application/pdf",  # For PDF uploads
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        },
        description="Allowed content types for requests"
    )
    strict_content_type_checking: bool = Field(
        default=False,
        description="Require exact content type matches (no charset, etc.)"
    )


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration."""
    
    include_error_details: bool = Field(
        default=False,  # Set to False in production for security
        description="Include detailed error information in responses"
    )
    max_error_message_length: int = Field(
        default=500,
        ge=50,
        le=2000,
        description="Maximum length of error messages"
    )
    
    custom_error_codes: Dict[str, str] = Field(
        default={
            "VALIDATION_FAILED": "VALIDATION_ERROR_422",
            "SECURITY_VIOLATION": "SECURITY_ERROR_403",
            "RATE_LIMIT_EXCEEDED": "RATE_LIMIT_ERROR_429",
            "REQUEST_TOO_LARGE": "SIZE_ERROR_413",
            "INVALID_CONTENT_TYPE": "CONTENT_TYPE_ERROR_415",
            "TIMEOUT": "TIMEOUT_ERROR_408",
        },
        description="Custom error codes for different validation failures"
    )
    
    enable_error_tracking: bool = Field(
        default=True,
        description="Enable error tracking and aggregation"
    )


class ValidationConfig(BaseModel):
    """
    Comprehensive validation configuration for the PDF2Markdown microservice.
    
    This configuration model combines all validation-related settings including
    security, performance, monitoring, content type validation, and error handling.
    """
    
    # Core settings
    enabled: bool = Field(
        default=True,
        description="Enable/disable validation middleware entirely"
    )
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    
    # Component configurations
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    content_type: ContentTypeConfig = Field(default_factory=ContentTypeConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    
    # Endpoint-specific settings
    endpoint_overrides: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="Per-endpoint configuration overrides"
    )
    
    # Schema validation settings
    enforce_schema_validation: bool = Field(
        default=True,
        description="Enforce Pydantic schema validation when schemas are registered"
    )
    allow_unknown_endpoints: bool = Field(
        default=True,
        description="Allow requests to endpoints without registered schemas"
    )
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        allowed_environments = {'development', 'staging', 'production'}
        if v.lower() not in allowed_environments:
            raise ValueError(f"Environment must be one of: {allowed_environments}")
        return v.lower()

    @field_validator('endpoint_overrides')
    @classmethod
    def validate_endpoint_overrides(cls, v):
        """Validate endpoint override format."""
        for endpoint, overrides in v.items():
            if not isinstance(endpoint, str) or not endpoint.startswith('/'):
                raise ValueError(f"Endpoint '{endpoint}' must be a string starting with '/'")
            if not isinstance(overrides, dict):
                raise ValueError(f"Overrides for endpoint '{endpoint}' must be a dictionary")
        return v
    
    def get_endpoint_config(self, endpoint: str, method: str = "GET") -> Dict[str, Any]:
        """
        Get configuration for a specific endpoint, including any overrides.
        
        Args:
            endpoint: The endpoint path (e.g., '/api/v1/documents')
            method: HTTP method (e.g., 'GET', 'POST')
            
        Returns:
            Dictionary containing the effective configuration for the endpoint
        """
        base_config = self.model_dump()
        
        # Check for endpoint-specific overrides
        endpoint_key = f"{method.upper()}:{endpoint}"
        if endpoint_key in self.endpoint_overrides:
            # Deep merge overrides
            overrides = self.endpoint_overrides[endpoint_key]
            return self._deep_merge(base_config, overrides)
        
        # Check for path-only overrides (applies to all methods)
        if endpoint in self.endpoint_overrides:
            overrides = self.endpoint_overrides[endpoint]
            return self._deep_merge(base_config, overrides)
        
        return base_config
    
    def _deep_merge(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


def create_validation_config_from_env() -> ValidationConfig:
    """
    Create validation configuration from environment variables.
    
    Environment variables follow the pattern: VALIDATION_<SECTION>_<SETTING>
    For example: VALIDATION_SECURITY_RATE_LIMIT_REQUESTS=200
    """
    config_dict = {}
    
    # Get environment
    config_dict['environment'] = os.getenv('ENVIRONMENT', 'development')
    config_dict['enabled'] = os.getenv('VALIDATION_ENABLED', 'true').lower() == 'true'
    
    # Security settings
    security_config = {}
    security_config['enable_rate_limiting'] = os.getenv('VALIDATION_SECURITY_RATE_LIMITING', 'true').lower() == 'true'
    security_config['rate_limit_requests'] = int(os.getenv('VALIDATION_SECURITY_RATE_LIMIT_REQUESTS', '100'))
    security_config['rate_limit_window'] = int(os.getenv('VALIDATION_SECURITY_RATE_LIMIT_WINDOW', '60'))
    security_config['enable_input_sanitization'] = os.getenv('VALIDATION_SECURITY_INPUT_SANITIZATION', 'true').lower() == 'true'
    security_config['max_request_size'] = int(os.getenv('VALIDATION_SECURITY_MAX_REQUEST_SIZE', str(50 * 1024 * 1024)))
    security_config['max_json_depth'] = int(os.getenv('VALIDATION_SECURITY_MAX_JSON_DEPTH', '10'))
    security_config['max_array_length'] = int(os.getenv('VALIDATION_SECURITY_MAX_ARRAY_LENGTH', '1000'))
    config_dict['security'] = security_config
    
    # Performance settings
    performance_config = {}
    performance_config['enable_caching'] = os.getenv('VALIDATION_PERFORMANCE_CACHING', 'true').lower() == 'true'
    performance_config['cache_ttl'] = int(os.getenv('VALIDATION_PERFORMANCE_CACHE_TTL', '300'))
    performance_config['max_cache_size'] = int(os.getenv('VALIDATION_PERFORMANCE_MAX_CACHE_SIZE', '1000'))
    performance_config['enable_compression'] = os.getenv('VALIDATION_PERFORMANCE_COMPRESSION', 'true').lower() == 'true'
    performance_config['compression_threshold'] = int(os.getenv('VALIDATION_PERFORMANCE_COMPRESSION_THRESHOLD', '1024'))
    performance_config['async_validation'] = os.getenv('VALIDATION_PERFORMANCE_ASYNC', 'false').lower() == 'true'
    performance_config['validation_timeout'] = float(os.getenv('VALIDATION_PERFORMANCE_TIMEOUT', '5.0'))
    config_dict['performance'] = performance_config
    
    # Monitoring settings
    monitoring_config = {}
    monitoring_config['enable_metrics'] = os.getenv('VALIDATION_MONITORING_METRICS', 'true').lower() == 'true'
    monitoring_config['log_validation_errors'] = os.getenv('VALIDATION_MONITORING_LOG_ERRORS', 'true').lower() == 'true'
    monitoring_config['log_performance_metrics'] = os.getenv('VALIDATION_MONITORING_LOG_PERFORMANCE', 'true').lower() == 'true'
    monitoring_config['log_security_violations'] = os.getenv('VALIDATION_MONITORING_LOG_SECURITY', 'true').lower() == 'true'
    monitoring_config['slow_request_threshold'] = float(os.getenv('VALIDATION_MONITORING_SLOW_THRESHOLD', '1.0'))
    monitoring_config['metrics_retention_hours'] = int(os.getenv('VALIDATION_MONITORING_RETENTION_HOURS', '24'))
    monitoring_config['enable_health_checks'] = os.getenv('VALIDATION_MONITORING_HEALTH_CHECKS', 'true').lower() == 'true'
    config_dict['monitoring'] = monitoring_config
    
    # Content type settings
    content_type_config = {}
    content_type_config['validate_content_type'] = os.getenv('VALIDATION_CONTENT_TYPE_VALIDATE', 'true').lower() == 'true'
    content_type_config['strict_content_type_checking'] = os.getenv('VALIDATION_CONTENT_TYPE_STRICT', 'false').lower() == 'true'
    config_dict['content_type'] = content_type_config
    
    # Error handling settings
    error_handling_config = {}
    error_handling_config['include_error_details'] = os.getenv('VALIDATION_ERROR_INCLUDE_DETAILS', 'false').lower() == 'true'
    error_handling_config['max_error_message_length'] = int(os.getenv('VALIDATION_ERROR_MAX_MESSAGE_LENGTH', '500'))
    error_handling_config['enable_error_tracking'] = os.getenv('VALIDATION_ERROR_TRACKING', 'true').lower() == 'true'
    config_dict['error_handling'] = error_handling_config
    
    # Schema validation settings
    config_dict['enforce_schema_validation'] = os.getenv('VALIDATION_ENFORCE_SCHEMA', 'true').lower() == 'true'
    config_dict['allow_unknown_endpoints'] = os.getenv('VALIDATION_ALLOW_UNKNOWN_ENDPOINTS', 'true').lower() == 'true'
    
    return ValidationConfig(**config_dict)


def create_production_config() -> ValidationConfig:
    """Create a production-optimized validation configuration."""
    return ValidationConfig(
        environment="production",
        security=SecurityConfig(
            enable_rate_limiting=True,
            rate_limit_requests=50,  # More restrictive in production
            rate_limit_window=60,
            enable_input_sanitization=True,
            max_request_size=25 * 1024 * 1024,  # 25MB limit
            max_json_depth=8,  # Stricter limits
            max_array_length=500,
        ),
        performance=PerformanceConfig(
            enable_caching=True,
            cache_ttl=600,  # 10 minutes
            max_cache_size=5000,  # Larger cache
            enable_compression=True,
            compression_threshold=512,  # Compress smaller responses
            async_validation=True,  # Enable async for better performance
            validation_timeout=3.0,  # Shorter timeout
        ),
        monitoring=MonitoringConfig(
            enable_metrics=True,
            log_validation_errors=True,
            log_performance_metrics=True,
            log_security_violations=True,
            slow_request_threshold=0.5,  # More sensitive
            metrics_retention_hours=72,  # 3 days
            enable_health_checks=True,
        ),
        content_type=ContentTypeConfig(
            validate_content_type=True,
            strict_content_type_checking=True,  # Strict in production
        ),
        error_handling=ErrorHandlingConfig(
            include_error_details=False,  # Hide details in production
            max_error_message_length=200,  # Shorter messages
            enable_error_tracking=True,
        ),
        enforce_schema_validation=True,
        allow_unknown_endpoints=False,  # Stricter in production
    )


def create_development_config() -> ValidationConfig:
    """Create a development-friendly validation configuration."""
    return ValidationConfig(
        environment="development",
        security=SecurityConfig(
            enable_rate_limiting=False,  # Disabled for development
            enable_input_sanitization=True,
            max_request_size=100 * 1024 * 1024,  # 100MB for testing
            max_json_depth=15,  # More lenient
            max_array_length=2000,
        ),
        performance=PerformanceConfig(
            enable_caching=False,  # Disabled for development
            enable_compression=False,  # Disabled for easier debugging
            async_validation=False,
            validation_timeout=10.0,  # Longer timeout
        ),
        monitoring=MonitoringConfig(
            enable_metrics=True,
            log_validation_errors=True,
            log_performance_metrics=False,  # Less noise
            log_security_violations=True,
            slow_request_threshold=2.0,  # More lenient
            metrics_retention_hours=12,
            enable_health_checks=True,
        ),
        content_type=ContentTypeConfig(
            validate_content_type=True,
            strict_content_type_checking=False,  # More lenient
        ),
        error_handling=ErrorHandlingConfig(
            include_error_details=True,  # Show details for debugging
            max_error_message_length=1000,  # Longer messages
            enable_error_tracking=True,
        ),
        enforce_schema_validation=True,
        allow_unknown_endpoints=True,  # Allow for development
    )


@lru_cache(maxsize=1)
def get_validation_config() -> ValidationConfig:
    """
    Get the validation configuration singleton.
    
    This function creates and caches the validation configuration based on
    the current environment. The configuration is cached to avoid repeated
    environment variable lookups.
    
    Returns:
        ValidationConfig: The cached validation configuration instance
    """
    # Check if we have explicit environment configuration
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        config = create_production_config()
    elif env == 'development':
        config = create_development_config()
    else:
        # For staging or other environments, use environment variables
        config = create_validation_config_from_env()
    
    return config


def reload_validation_config() -> ValidationConfig:
    """
    Reload the validation configuration from environment variables.
    
    This function clears the cache and reloads the configuration,
    useful for runtime configuration updates.
    
    Returns:
        ValidationConfig: The newly loaded configuration
    """
    get_validation_config.cache_clear()
    return get_validation_config()


# Export commonly used configurations
__all__ = [
    "ValidationConfig",
    "SecurityConfig", 
    "PerformanceConfig",
    "MonitoringConfig",
    "ContentTypeConfig",
    "ErrorHandlingConfig",
    "get_validation_config",
    "reload_validation_config",
    "create_validation_config_from_env",
    "create_production_config",
    "create_development_config",
]
