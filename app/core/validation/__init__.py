"""
Core validation package for the PDF2Markdown microservice.

This package provides comprehensive validation utilities, configuration management,
and error handling for request/response validation middleware.

Components:
- config: Validation configuration management
- errors: Custom validation error classes and handlers
- validators: Core validation logic and utilities
- registry: Schema registry for endpoint validation
"""

from .config import ValidationConfig, get_validation_config
from .errors import ValidationError, SecurityValidationError, ValidationErrorHandler
from .validators import SchemaValidator, SecurityValidator, StructureValidator
from .registry import ValidationRegistry

__all__ = [
    "ValidationConfig",
    "get_validation_config", 
    "ValidationError",
    "SecurityValidationError",
    "ValidationErrorHandler",
    "SchemaValidator",
    "SecurityValidator", 
    "StructureValidator",
    "ValidationRegistry"
]
