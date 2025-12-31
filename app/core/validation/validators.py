"""
Core validation utilities for the PDF2Markdown microservice.

This module provides specialized validators for schema validation, security checks,
and structural validation of requests and responses.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
import logging

from pydantic import BaseModel, ValidationError as PydanticValidationError

from .config import ValidationConfig
from .errors import (
    ValidationError,
    SecurityValidationError,
    JSONStructureError,
    ValidationTimeoutError,
    create_schema_validation_error,
    create_security_error
)

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Schema validation using Pydantic models.
    
    This validator handles validation of request/response data against
    registered Pydantic schemas with support for async validation and
    custom error handling.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize schema validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.timeout = config.performance.validation_timeout
        self.async_enabled = config.performance.async_validation
    
    async def validate_data(
        self,
        data: Dict[str, Any],
        schema_class: Type[BaseModel],
        context: Optional[Dict[str, Any]] = None
    ) -> BaseModel:
        """
        Validate data against a Pydantic schema.
        
        Args:
            data: Data to validate
            schema_class: Pydantic model class
            context: Additional validation context
            
        Returns:
            Validated Pydantic model instance
            
        Raises:
            ValidationError: If validation fails
            ValidationTimeoutError: If validation times out
        """
        try:
            if self.async_enabled:
                # Run validation in a separate task with timeout
                validation_task = asyncio.create_task(
                    self._validate_async(data, schema_class)
                )
                
                try:
                    validated_data = await asyncio.wait_for(
                        validation_task,
                        timeout=self.timeout
                    )
                    return validated_data
                except asyncio.TimeoutError:
                    validation_task.cancel()
                    raise ValidationTimeoutError(
                        f"Schema validation timed out after {self.timeout} seconds",
                        timeout_duration=self.timeout
                    )
            else:
                # Synchronous validation
                return schema_class(**data)
                
        except PydanticValidationError as e:
            # Convert Pydantic errors to our custom ValidationError
            field_errors = []
            for error in e.errors():
                field_errors.append({
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                    "value": error.get("input")
                })
            
            raise ValidationError(
                message=f"Schema validation failed: {len(field_errors)} field(s) invalid",
                field_errors=field_errors,
                context=context or {}
            )
        except Exception as e:
            logger.error("Unexpected error during schema validation: %s", str(e))
            raise ValidationError(
                message=f"Schema validation error: {str(e)}",
                context=context or {}
            )
    
    async def _validate_async(
        self,
        data: Dict[str, Any],
        schema_class: Type[BaseModel]
    ) -> BaseModel:
        """
        Asynchronous validation wrapper.
        
        Args:
            data: Data to validate
            schema_class: Pydantic model class
            
        Returns:
            Validated Pydantic model instance
        """
        # Run validation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: schema_class(**data)
        )
    
    def validate_partial(
        self,
        data: Dict[str, Any],
        schema_class: Type[BaseModel],
        fields: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate only specific fields of a schema.
        
        Args:
            data: Data to validate
            schema_class: Pydantic model class
            fields: List of field names to validate
            context: Additional validation context
            
        Returns:
            Dictionary of validated field values
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Create a partial model with only specified fields
            partial_data = {k: v for k, v in data.items() if k in fields}
            
            # Validate using the schema's field validators
            validated_fields = {}
            model_fields = schema_class.__fields__
            
            for field_name in fields:
                if field_name not in model_fields:
                    raise create_schema_validation_error(
                        field_name,
                        f"Field '{field_name}' not found in schema",
                        context=context
                    )
                
                field_info = model_fields[field_name]
                field_value = partial_data.get(field_name)
                
                # Apply field validation
                try:
                    validated_value = field_info.validate(
                        field_value,
                        values=partial_data,
                        loc=field_name,
                        cls=schema_class
                    )
                    validated_fields[field_name] = validated_value
                except Exception as e:
                    raise create_schema_validation_error(
                        field_name,
                        str(e),
                        value=field_value,
                        context=context
                    )
            
            return validated_fields
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("Unexpected error during partial validation: %s", str(e))
            raise ValidationError(
                message=f"Partial validation error: {str(e)}",
                context=context or {}
            )


class SecurityValidator:
    """
    Security validation for input sanitization and threat detection.
    
    This validator checks for common security threats including XSS,
    injection attacks, and malicious patterns in user input.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize security validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.blocked_patterns = config.security.blocked_patterns
        self.enabled = config.security.enable_input_sanitization
    
    def validate_input(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate input for security threats.
        
        Args:
            data: Data to validate
            context: Additional validation context
            
        Returns:
            True if input is safe, False otherwise
            
        Raises:
            SecurityValidationError: If security threat is detected
        """
        if not self.enabled:
            return True
        
        try:
            violations = self._scan_for_threats(data)
            
            if violations:
                violation_details = {
                    "violations": violations,
                    "data_type": type(data).__name__
                }
                
                raise create_security_error(
                    violation_type="malicious_input",
                    message=f"Security threat detected: {len(violations)} violation(s) found",
                    context={**(context or {}), **violation_details}
                )
            
            return True
            
        except SecurityValidationError:
            raise
        except Exception as e:
            logger.error("Unexpected error during security validation: %s", str(e))
            raise SecurityValidationError(
                message=f"Security validation error: {str(e)}",
                violation_type="validation_error",
                context=context or {}
            )
    
    def _scan_for_threats(self, data: Any, path: str = "") -> List[Dict[str, Any]]:
        """
        Recursively scan data for security threats.
        
        Args:
            data: Data to scan
            path: Current path in data structure
            
        Returns:
            List of detected violations
        """
        violations = []
        
        if isinstance(data, str):
            violations.extend(self._check_string_threats(data, path))
        elif isinstance(data, dict):
            for key, value in data.items():
                key_path = f"{path}.{key}" if path else key
                # Check key for threats
                if isinstance(key, str):
                    violations.extend(self._check_string_threats(key, f"{key_path}[key]"))
                # Check value recursively
                violations.extend(self._scan_for_threats(value, key_path))
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                violations.extend(self._scan_for_threats(item, item_path))
        
        return violations
    
    def _check_string_threats(self, text: str, path: str) -> List[Dict[str, Any]]:
        """
        Check a string for security threats.
        
        Args:
            text: String to check
            path: Path to the string in data structure
            
        Returns:
            List of detected violations
        """
        violations = []
        
        for pattern in self.blocked_patterns:
            matches = pattern.findall(text)
            if matches:
                violations.append({
                    "path": path,
                    "pattern": pattern.pattern,
                    "matches": matches[:5],  # Limit to first 5 matches
                    "match_count": len(matches)
                })
        
        return violations
    
    def sanitize_input(
        self,
        data: Any,
        aggressive: bool = False
    ) -> Any:
        """
        Sanitize input by removing or escaping dangerous content.
        
        Args:
            data: Data to sanitize
            aggressive: Whether to use aggressive sanitization
            
        Returns:
            Sanitized data
        """
        if not self.enabled:
            return data
        
        return self._sanitize_recursive(data, aggressive)
    
    def _sanitize_recursive(self, data: Any, aggressive: bool) -> Any:
        """
        Recursively sanitize data structure.
        
        Args:
            data: Data to sanitize
            aggressive: Whether to use aggressive sanitization
            
        Returns:
            Sanitized data
        """
        if isinstance(data, str):
            return self._sanitize_string(data, aggressive)
        elif isinstance(data, dict):
            return {
                self._sanitize_string(str(k), aggressive) if isinstance(k, str) else k:
                self._sanitize_recursive(v, aggressive)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_recursive(item, aggressive) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._sanitize_recursive(item, aggressive) for item in data)
        else:
            return data
    
    def _sanitize_string(self, text: str, aggressive: bool) -> str:
        """
        Sanitize a string by removing dangerous patterns.
        
        Args:
            text: String to sanitize
            aggressive: Whether to use aggressive sanitization
            
        Returns:
            Sanitized string
        """
        sanitized = text
        
        # Remove or escape dangerous patterns
        for pattern in self.blocked_patterns:
            if aggressive:
                # Remove matches entirely
                sanitized = pattern.sub("", sanitized)
            else:
                # Escape HTML/JS dangerous characters
                sanitized = sanitized.replace("<", "&lt;")
                sanitized = sanitized.replace(">", "&gt;")
                sanitized = sanitized.replace("\"", "&quot;")
                sanitized = sanitized.replace("'", "&#x27;")
                sanitized = sanitized.replace("&", "&amp;")
        
        return sanitized


class StructureValidator:
    """
    Validator for JSON structure and format validation.
    
    This validator checks JSON depth, array lengths, and overall
    structure compliance with configured limits.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize structure validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.max_depth = config.security.max_json_depth
        self.max_array_length = config.security.max_array_length
    
    def validate_structure(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate data structure against configured limits.
        
        Args:
            data: Data to validate
            context: Additional validation context
            
        Returns:
            True if structure is valid
            
        Raises:
            JSONStructureError: If structure validation fails
        """
        try:
            violations = []
            
            # Check depth
            depth = self._calculate_depth(data)
            if depth > self.max_depth:
                violations.append(f"JSON depth {depth} exceeds limit of {self.max_depth}")
            
            # Check array lengths
            array_violations = self._check_array_lengths(data)
            violations.extend(array_violations)
            
            if violations:
                raise JSONStructureError(
                    message=f"JSON structure validation failed: {'; '.join(violations)}",
                    violation_reason="; ".join(violations),
                    context=context or {}
                )
            
            return True
            
        except JSONStructureError:
            raise
        except Exception as e:
            logger.error("Unexpected error during structure validation: %s", str(e))
            raise JSONStructureError(
                message=f"Structure validation error: {str(e)}",
                context=context or {}
            )
    
    def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        """
        Calculate the maximum depth of a data structure.
        
        Args:
            data: Data to analyze
            current_depth: Current depth level
            
        Returns:
            Maximum depth found
        """
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                self._calculate_depth(value, current_depth + 1)
                for value in data.values()
            )
        elif isinstance(data, (list, tuple)):
            if not data:
                return current_depth
            return max(
                self._calculate_depth(item, current_depth + 1)
                for item in data
            )
        else:
            return current_depth
    
    def _check_array_lengths(
        self,
        data: Any,
        path: str = ""
    ) -> List[str]:
        """
        Check array lengths recursively.
        
        Args:
            data: Data to check
            path: Current path in data structure
            
        Returns:
            List of violation messages
        """
        violations = []
        
        if isinstance(data, (list, tuple)):
            if len(data) > self.max_array_length:
                violations.append(
                    f"Array at '{path}' has length {len(data)}, "
                    f"exceeds limit of {self.max_array_length}"
                )
            
            # Check nested arrays
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                violations.extend(self._check_array_lengths(item, item_path))
        
        elif isinstance(data, dict):
            for key, value in data.items():
                key_path = f"{path}.{key}" if path else key
                violations.extend(self._check_array_lengths(value, key_path))
        
        return violations
    
    def validate_json_string(
        self,
        json_string: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate and parse a JSON string.
        
        Args:
            json_string: JSON string to validate
            context: Additional validation context
            
        Returns:
            Parsed JSON data
            
        Raises:
            JSONStructureError: If JSON is invalid or violates structure rules
        """
        try:
            # Parse JSON
            data = json.loads(json_string)
            
            # Validate structure
            self.validate_structure(data, context)
            
            return data
            
        except json.JSONDecodeError as e:
            raise JSONStructureError(
                message=f"Invalid JSON: {str(e)}",
                violation_reason=f"JSON parsing failed: {str(e)}",
                context=context or {}
            )
        except JSONStructureError:
            raise
        except Exception as e:
            logger.error("Unexpected error during JSON validation: %s", str(e))
            raise JSONStructureError(
                message=f"JSON validation error: {str(e)}",
                context=context or {}
            )


# Export all validators
__all__ = [
    "SchemaValidator",
    "SecurityValidator", 
    "StructureValidator"
]
