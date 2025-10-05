"""
Comprehensive Request/Response Validation Middleware for PDF2Markdown Microservice.

This module implements production-ready validation middleware that leverages the existing
Pydantic schema foundation to provide comprehensive request/response validation with
security features, performance optimization, and monitoring capabilities.

Key Features:
- Request/response validation using existing Pydantic schemas
- Security validation (rate limiting, input sanitization, size limits)
- Performance optimization (caching, async processing)
- Comprehensive monitoring and metrics
- Configurable validation rules and error handling
- Integration with existing FastAPI middleware stack
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from functools import wraps
import hashlib
import re

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from pydantic import BaseModel, ValidationError, validator
from pydantic.error_wrappers import ErrorWrapper

from app.schemas.common import BaseResponse, ErrorResponse
from app.config import settings


# Configure logger for validation middleware
logger = logging.getLogger("validation_middleware")

class ValidationResult:
    """Result of validation operation."""
    
    def __init__(
        self,
        is_valid: bool,
        error_message: Optional[str] = None,
        details: Optional[List[Dict[str, Any]]] = None,
        validated_data: Optional[Any] = None
    ):
        self.is_valid = is_valid
        self.error_message = error_message
        self.details = details
        self.validated_data = validated_data


class ValidationConfig(BaseModel):
    """Configuration model for validation middleware settings."""
    
    # Request validation settings
    max_request_size: int = 50 * 1024 * 1024  # 50MB default
    max_json_depth: int = 10
    max_array_length: int = 1000
    validate_content_type: bool = True
    allowed_content_types: Set[str] = {
        "application/json",
        "multipart/form-data", 
        "application/x-www-form-urlencoded"
    }
    
    # Security settings
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100  # requests per window
    
    # Response validation settings (Phase 3)
    max_response_size: int = 100 * 1024 * 1024  # 100MB default
    strict_response_validation: bool = False  # Whether to replace invalid responses with errors
    required_response_headers: Dict[str, List[str]] = {}  # Required headers per endpoint
    validate_response_schemas: bool = True
    enable_response_sanitization: bool = True
    rate_limit_window: int = 60  # seconds
    enable_input_sanitization: bool = True
    blocked_patterns: List[str] = [
        r"<script[^>]*>.*?</script>",  # XSS prevention
        r"javascript:",
        r"data:text/html",
        r"vbscript:",
    ]
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    
    # Monitoring settings
    enable_metrics: bool = True
    log_validation_errors: bool = True
    log_performance_metrics: bool = True
    slow_request_threshold: float = 1.0  # seconds
    
    # Error handling settings
    include_error_details: bool = False  # Set to False in production
    max_error_message_length: int = 500
    
    @validator('blocked_patterns')
    def compile_patterns(cls, v):
        """Compile regex patterns for better performance."""
        return [re.compile(pattern, re.IGNORECASE) for pattern in v]


class ValidationMetrics:
    """Metrics collection for validation middleware."""
    
    def __init__(self):
        self.request_count = 0
        self.validation_errors = 0
        self.security_violations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.processing_times = deque(maxlen=1000)
        self.error_types = defaultdict(int)
        self.endpoint_metrics = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'avg_time': 0.0
        })
    
    def record_request(self, endpoint: str, processing_time: float, error_type: Optional[str] = None):
        """Record request metrics."""
        self.request_count += 1
        self.processing_times.append(processing_time)
        
        endpoint_data = self.endpoint_metrics[endpoint]
        endpoint_data['requests'] += 1
        
        # Update average processing time
        current_avg = endpoint_data['avg_time']
        request_count = endpoint_data['requests']
        endpoint_data['avg_time'] = (current_avg * (request_count - 1) + processing_time) / request_count
        
        if error_type:
            self.validation_errors += 1
            self.error_types[error_type] += 1
            endpoint_data['errors'] += 1
    
    def record_security_violation(self):
        """Record security violation."""
        self.security_violations += 1
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.cache_misses += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            'total_requests': self.request_count,
            'validation_errors': self.validation_errors,
            'security_violations': self.security_violations,
            'average_processing_time': round(avg_time, 4),
            'cache_hit_rate': round(cache_hit_rate, 4),
            'error_types': dict(self.error_types),
            'endpoint_metrics': dict(self.endpoint_metrics)
        }


class RateLimiter:
    """Rate limiting implementation with sliding window."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] <= now - self.window_seconds:
            client_requests.popleft()
        
        # Check if under limit
        if len(client_requests) < self.max_requests:
            client_requests.append(now)
            return True
        
        return False


class ValidationCache:
    """Simple in-memory cache for validation results."""
    
    def __init__(self, max_size: int, ttl: int):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, data: Any) -> Optional[Any]:
        """Get cached validation result."""
        key = self._generate_key(data)
        
        if key in self.cache:
            # Check if expired
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            return self.cache[key]
        
        return None
    
    def set(self, data: Any, result: Any):
        """Cache validation result."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        key = self._generate_key(data)
        self.cache[key] = result
        self.timestamps[key] = time.time()


class SecurityValidator:
    """Security validation utilities."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.blocked_patterns = config.blocked_patterns
    
    def validate_input(self, data: Any) -> bool:
        """Validate input for security threats."""
        if not self.config.enable_input_sanitization:
            return True
        
        def check_string(s: str) -> bool:
            for pattern in self.blocked_patterns:
                if pattern.search(s):
                    return False
            return True
        
        def validate_recursive(obj: Any) -> bool:
            if isinstance(obj, str):
                return check_string(obj)
            elif isinstance(obj, dict):
                return all(
                    check_string(str(k)) and validate_recursive(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, (list, tuple)):
                return all(validate_recursive(item) for item in obj)
            return True
        
        return validate_recursive(data)
    
    def validate_json_structure(self, data: Any, depth: int = 0) -> bool:
        """Validate JSON structure for depth and array length limits."""
        if depth > self.config.max_json_depth:
            return False
        
        if isinstance(data, dict):
            return all(
                self.validate_json_structure(v, depth + 1)
                for v in data.values()
            )
        elif isinstance(data, list):
            if len(data) > self.config.max_array_length:
                return False
            return all(
                self.validate_json_structure(item, depth + 1)
                for item in data
            )
        
        return True


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive validation middleware for FastAPI applications.
    
    Provides request/response validation, security checks, performance optimization,
    and monitoring capabilities while integrating with existing Pydantic schemas.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[ValidationConfig] = None,
        schema_registry: Optional[Dict[str, BaseModel]] = None
    ):
        super().__init__(app)
        self.config = config or ValidationConfig()
        self.schema_registry = schema_registry or {}
        
        # Initialize components
        self.metrics = ValidationMetrics()
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window
        ) if self.config.enable_rate_limiting else None
        
        self.cache = ValidationCache(
            self.config.max_cache_size,
            self.config.cache_ttl
        ) if self.config.enable_caching else None
        
        self.security_validator = SecurityValidator(self.config)
        
        logger.info("ValidationMiddleware initialized with config: %s", self.config.dict())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method."""
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        client_id = self._get_client_id(request)
        
        try:
            # Rate limiting check
            if self.rate_limiter and not self.rate_limiter.is_allowed(client_id):
                self.metrics.record_security_violation()
                return self._create_error_response(
                    "Rate limit exceeded",
                    status.HTTP_429_TOO_MANY_REQUESTS
                )
            
            # Request size validation
            if not await self._validate_request_size(request):
                return self._create_error_response(
                    "Request size exceeds limit",
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                )
            
            # Content type validation
            if not self._validate_content_type(request):
                return self._create_error_response(
                    "Invalid content type",
                    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
                )
            
            # Request validation
            validation_result = await self._validate_request(request)
            if not validation_result.is_valid:
                return self._create_error_response(
                    validation_result.error_message,
                    status.HTTP_422_UNPROCESSABLE_ENTITY,
                    validation_result.details
                )
            
            # Process request
            response = await call_next(request)
            
            # Response validation (if enabled)
            if hasattr(response, 'body'):
                await self._validate_response(response, endpoint)
            
            # Record successful request
            processing_time = time.time() - start_time
            self.metrics.record_request(endpoint, processing_time)
            
            if self.config.log_performance_metrics and processing_time > self.config.slow_request_threshold:
                logger.warning(
                    "Slow request detected: %s took %.2fs",
                    endpoint, processing_time
                )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_type = type(e).__name__
            
            self.metrics.record_request(endpoint, processing_time, error_type)
            
            if self.config.log_validation_errors:
                logger.error(
                    "Validation middleware error for %s: %s",
                    endpoint, str(e), exc_info=True
                )
            
            return self._create_error_response(
                "Internal validation error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier for rate limiting."""
        # Try to get from X-Forwarded-For header first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    async def _validate_request_size(self, request: Request) -> bool:
        """Validate request size limits."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                return size <= self.config.max_request_size
            except ValueError:
                return False
        return True
    
    def _validate_content_type(self, request: Request) -> bool:
        """Validate request content type."""
        if not self.config.validate_content_type:
            return True
        
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        return content_type in self.config.allowed_content_types or content_type == ""
    
    async def _validate_request(self, request: Request) -> 'ValidationResult':
        """Validate request data against schemas and security rules."""
        try:
            # Check cache first
            if self.cache:
                # Create cache key from request data
                cache_key_data = {
                    'method': request.method,
                    'path': request.url.path,
                    'headers': dict(request.headers),
                }
                
                # Add body to cache key if present
                if request.method in ['POST', 'PUT', 'PATCH']:
                    body = await request.body()
                    if body:
                        cache_key_data['body'] = body.decode('utf-8', errors='ignore')
                
                cached_result = self.cache.get(cache_key_data)
                if cached_result:
                    self.metrics.record_cache_hit()
                    return cached_result
                else:
                    self.metrics.record_cache_miss()
            
            # Parse request data
            request_data = await self._parse_request_data(request)
            
            # Security validation
            if not self.security_validator.validate_input(request_data):
                self.metrics.record_security_violation()
                return ValidationResult(
                    is_valid=False,
                    error_message="Security validation failed: potentially malicious input detected"
                )
            
            if not self.security_validator.validate_json_structure(request_data):
                return ValidationResult(
                    is_valid=False,
                    error_message="JSON structure validation failed: depth or array length limits exceeded"
                )
            
            # Perform comprehensive validation using all validation components
            endpoint_path = request.url.path
            validation_results = []
            
            # 1. Schema validation (request body)
            endpoint_key = f"{request.method}:{endpoint_path}"
            if endpoint_key in self.schema_registry:
                schema_class = self.schema_registry[endpoint_key]
                try:
                    validated_data = schema_class(**request_data)
                    schema_result = ValidationResult(is_valid=True, validated_data={'body': validated_data})
                except ValidationError as e:
                    schema_result = ValidationResult(
                        is_valid=False,
                        error_message="Schema validation failed",
                        details=self._format_validation_errors(e)
                    )
            else:
                schema_result = ValidationResult(is_valid=True, validated_data={'body': request_data})
            validation_results.append(schema_result)
            
            # 2. Query parameter validation
            query_result = await self._validate_query_parameters(request, endpoint_path)
            validation_results.append(query_result)
            
            # 3. Path parameter validation
            path_result = await self._validate_path_parameters(request, endpoint_path)
            validation_results.append(path_result)
            
            # 4. Header validation
            header_result = await self._validate_headers(request, endpoint_path)
            validation_results.append(header_result)
            
            # 5. File upload validation
            file_result = await self._validate_file_uploads(request, endpoint_path)
            validation_results.append(file_result)
            
            # 6. Aggregate all validation results
            result = await self._aggregate_validation_errors(*validation_results)
            
            # Cache the result
            if self.cache:
                self.cache.set(cache_key_data, result)
            
            return result
            
        except Exception as e:
            logger.error("Request validation error: %s", str(e))
            return ValidationResult(
                is_valid=False,
                error_message="Request validation failed"
            )
    
    async def _parse_request_data(self, request: Request) -> Dict[str, Any]:
        """Parse request data from various content types."""
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        
        if content_type == "application/json":
            try:
                body = await request.body()
                if body:
                    return json.loads(body.decode('utf-8'))
                return {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ValueError("Invalid JSON in request body")
        
        elif content_type == "application/x-www-form-urlencoded":
            form_data = await request.form()
            return dict(form_data)
        
        elif content_type == "multipart/form-data":
            form_data = await request.form()
            return dict(form_data)
        
        # For other content types or GET requests, return query parameters
        return dict(request.query_params)
    
    async def _validate_query_parameters(self, request: Request, endpoint: str) -> ValidationResult:
        """Validate query parameters with type coercion and validation rules."""
        try:
            query_params = dict(request.query_params)
            
            # Get query parameter validation rules for this endpoint
            endpoint_key = f"{request.method}:{endpoint}"
            if hasattr(self, 'query_param_rules') and endpoint_key in self.query_param_rules:
                rules = self.query_param_rules[endpoint_key]
                validated_params = {}
                errors = []
                
                for param_name, rule in rules.items():
                    param_value = query_params.get(param_name)
                    
                    # Check required parameters
                    if rule.get('required', False) and param_value is None:
                        errors.append({
                            'field': f'query.{param_name}',
                            'message': f'Required query parameter "{param_name}" is missing',
                            'type': 'missing'
                        })
                        continue
                    
                    # Skip validation if parameter is optional and not provided
                    if param_value is None:
                        continue
                    
                    # Type coercion and validation
                    param_type = rule.get('type', str)
                    try:
                        if param_type == int:
                            validated_params[param_name] = int(param_value)
                        elif param_type == float:
                            validated_params[param_name] = float(param_value)
                        elif param_type == bool:
                            validated_params[param_name] = param_value.lower() in ('true', '1', 'yes', 'on')
                        elif param_type == list:
                            # Handle comma-separated values
                            validated_params[param_name] = [item.strip() for item in param_value.split(',')]
                        else:
                            validated_params[param_name] = str(param_value)
                        
                        # Apply validation constraints
                        if 'min_value' in rule and validated_params[param_name] < rule['min_value']:
                            errors.append({
                                'field': f'query.{param_name}',
                                'message': f'Value must be at least {rule["min_value"]}',
                                'type': 'value_error'
                            })
                        
                        if 'max_value' in rule and validated_params[param_name] > rule['max_value']:
                            errors.append({
                                'field': f'query.{param_name}',
                                'message': f'Value must be at most {rule["max_value"]}',
                                'type': 'value_error'
                            })
                        
                        if 'choices' in rule and validated_params[param_name] not in rule['choices']:
                            errors.append({
                                'field': f'query.{param_name}',
                                'message': f'Value must be one of: {", ".join(map(str, rule["choices"]))}',
                                'type': 'value_error'
                            })
                        
                        if 'pattern' in rule and param_type == str:
                            import re
                            if not re.match(rule['pattern'], validated_params[param_name]):
                                errors.append({
                                    'field': f'query.{param_name}',
                                    'message': f'Value does not match required pattern',
                                    'type': 'value_error'
                                })
                    
                    except (ValueError, TypeError) as e:
                        errors.append({
                            'field': f'query.{param_name}',
                            'message': f'Invalid {param_type.__name__} value: {param_value}',
                            'type': 'type_error'
                        })
                
                if errors:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Query parameter validation failed",
                        details=errors
                    )
                
                return ValidationResult(
                    is_valid=True,
                    validated_data={'query_params': validated_params}
                )
            
            # No specific rules, return success with original params
            return ValidationResult(is_valid=True, validated_data={'query_params': query_params})
            
        except Exception as e:
            logger.error("Query parameter validation error: %s", str(e))
            return ValidationResult(
                is_valid=False,
                error_message="Query parameter validation failed"
            )
    
    async def _validate_path_parameters(self, request: Request, endpoint: str) -> ValidationResult:
        """Validate path parameters with custom validation rules."""
        try:
            # Extract path parameters from the request
            path_params = {}
            if hasattr(request, 'path_params'):
                path_params = dict(request.path_params)
            
            # Get path parameter validation rules for this endpoint
            endpoint_key = f"{request.method}:{endpoint}"
            if hasattr(self, 'path_param_rules') and endpoint_key in self.path_param_rules:
                rules = self.path_param_rules[endpoint_key]
                validated_params = {}
                errors = []
                
                for param_name, rule in rules.items():
                    param_value = path_params.get(param_name)
                    
                    # Path parameters are typically always required
                    if param_value is None:
                        errors.append({
                            'field': f'path.{param_name}',
                            'message': f'Required path parameter "{param_name}" is missing',
                            'type': 'missing'
                        })
                        continue
                    
                    # Type coercion and validation
                    param_type = rule.get('type', str)
                    try:
                        if param_type == int:
                            validated_params[param_name] = int(param_value)
                        elif param_type == float:
                            validated_params[param_name] = float(param_value)
                        else:
                            validated_params[param_name] = str(param_value)
                        
                        # Apply validation constraints
                        if 'min_value' in rule and validated_params[param_name] < rule['min_value']:
                            errors.append({
                                'field': f'path.{param_name}',
                                'message': f'Value must be at least {rule["min_value"]}',
                                'type': 'value_error'
                            })
                        
                        if 'max_value' in rule and validated_params[param_name] > rule['max_value']:
                            errors.append({
                                'field': f'path.{param_name}',
                                'message': f'Value must be at most {rule["max_value"]}',
                                'type': 'value_error'
                            })
                        
                        if 'pattern' in rule and param_type == str:
                            import re
                            if not re.match(rule['pattern'], validated_params[param_name]):
                                errors.append({
                                    'field': f'path.{param_name}',
                                    'message': f'Value does not match required pattern',
                                    'type': 'value_error'
                                })
                        
                        # Custom validation function
                        if 'validator' in rule and callable(rule['validator']):
                            try:
                                if not rule['validator'](validated_params[param_name]):
                                    errors.append({
                                        'field': f'path.{param_name}',
                                        'message': rule.get('validator_message', 'Custom validation failed'),
                                        'type': 'validation_error'
                                    })
                            except Exception as ve:
                                errors.append({
                                    'field': f'path.{param_name}',
                                    'message': f'Validation error: {str(ve)}',
                                    'type': 'validation_error'
                                })
                    
                    except (ValueError, TypeError) as e:
                        errors.append({
                            'field': f'path.{param_name}',
                            'message': f'Invalid {param_type.__name__} value: {param_value}',
                            'type': 'type_error'
                        })
                
                if errors:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Path parameter validation failed",
                        details=errors
                    )
                
                return ValidationResult(
                    is_valid=True,
                    validated_data={'path_params': validated_params}
                )
            
            # No specific rules, return success with original params
            return ValidationResult(is_valid=True, validated_data={'path_params': path_params})
            
        except Exception as e:
            logger.error("Path parameter validation error: %s", str(e))
            return ValidationResult(
                is_valid=False,
                error_message="Path parameter validation failed"
            )
    
    async def _validate_headers(self, request: Request, endpoint: str) -> ValidationResult:
        """Validate headers for required/optional headers with security checks."""
        try:
            headers = dict(request.headers)
            
            # Get header validation rules for this endpoint
            endpoint_key = f"{request.method}:{endpoint}"
            if hasattr(self, 'header_rules') and endpoint_key in self.header_rules:
                rules = self.header_rules[endpoint_key]
                validated_headers = {}
                errors = []
                
                for header_name, rule in rules.items():
                    # Headers are case-insensitive, so normalize
                    header_value = None
                    for h_name, h_value in headers.items():
                        if h_name.lower() == header_name.lower():
                            header_value = h_value
                            break
                    
                    # Check required headers
                    if rule.get('required', False) and header_value is None:
                        errors.append({
                            'field': f'headers.{header_name}',
                            'message': f'Required header "{header_name}" is missing',
                            'type': 'missing'
                        })
                        continue
                    
                    # Skip validation if header is optional and not provided
                    if header_value is None:
                        continue
                    
                    # Validate header value
                    if 'pattern' in rule:
                        import re
                        if not re.match(rule['pattern'], header_value):
                            errors.append({
                                'field': f'headers.{header_name}',
                                'message': f'Header value does not match required pattern',
                                'type': 'value_error'
                            })
                    
                    if 'choices' in rule and header_value not in rule['choices']:
                        errors.append({
                            'field': f'headers.{header_name}',
                            'message': f'Header value must be one of: {", ".join(rule["choices"])}',
                            'type': 'value_error'
                        })
                    
                    # Security checks
                    if rule.get('security_check', False):
                        # Check for potentially malicious header values
                        if self.security_validator.contains_malicious_patterns(header_value):
                            errors.append({
                                'field': f'headers.{header_name}',
                                'message': f'Header contains potentially malicious content',
                                'type': 'security_error'
                            })
                    
                    validated_headers[header_name] = header_value
                
                if errors:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Header validation failed",
                        details=errors
                    )
                
                return ValidationResult(
                    is_valid=True,
                    validated_data={'headers': validated_headers}
                )
            
            # Perform basic security checks on common headers
            security_errors = []
            
            # Check Content-Type for suspicious values
            content_type = headers.get('content-type', '')
            if content_type and self.security_validator.contains_malicious_patterns(content_type):
                security_errors.append({
                    'field': 'headers.content-type',
                    'message': 'Content-Type header contains potentially malicious content',
                    'type': 'security_error'
                })
            
            # Check User-Agent for suspicious patterns
            user_agent = headers.get('user-agent', '')
            if user_agent and len(user_agent) > 1000:  # Unusually long user agent
                security_errors.append({
                    'field': 'headers.user-agent',
                    'message': 'User-Agent header is unusually long',
                    'type': 'security_error'
                })
            
            if security_errors:
                return ValidationResult(
                    is_valid=False,
                    error_message="Header security validation failed",
                    details=security_errors
                )
            
            return ValidationResult(is_valid=True, validated_data={'headers': headers})
            
        except Exception as e:
            logger.error("Header validation error: %s", str(e))
            return ValidationResult(
                is_valid=False,
                error_message="Header validation failed"
            )
    
    async def _validate_response(self, response: Response, endpoint: str):
        """
        Comprehensive response validation system.
        
        Validates response data against schemas, headers, size limits, and security requirements.
        Implements SLA compliance checking and response sanitization.
        """
        start_time = time.time()
        
        try:
            # Skip validation for certain response types
            if response.status_code in [204, 304]:  # No Content, Not Modified
                return
            
            # 1. Response Size Validation
            await self._validate_response_size(response)
            
            # 2. Response Headers Validation
            await self._validate_response_headers(response, endpoint)
            
            # 3. Content-Type Validation
            await self._validate_response_content_type(response, endpoint)
            
            # 4. Response Schema Validation (if JSON)
            if self._is_json_response(response):
                await self._validate_response_schema(response, endpoint)
            
            # 5. Response Time SLA Validation
            await self._validate_response_time_sla(response, endpoint, start_time)
            
            # 6. Security Response Validation
            await self._validate_response_security(response)
            
            # 7. Response Sanitization
            await self._sanitize_response_data(response)
            
            # Update metrics
            self.metrics.response_validations_passed += 1
            
        except ValidationError as e:
            self.metrics.response_validation_errors += 1
            logger.warning(f"Response validation failed for {endpoint}: {str(e)}")
            
            if self.config.strict_response_validation:
                # Replace response with error response
                error_response = ErrorResponse(
                    success=False,
                    message="Response validation failed",
                    error_code="RESPONSE_VALIDATION_ERROR",
                    error_type="ValidationError",
                    details={"validation_errors": self._format_validation_errors(e)},
                    timestamp=datetime.utcnow()
                )
                
                # Update response with error
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                response.headers["content-type"] = "application/json"
                response._content = json.dumps(error_response.dict()).encode()
                
        except Exception as e:
            self.metrics.response_validation_errors += 1
            logger.error(f"Unexpected error in response validation for {endpoint}: {str(e)}")
            
            if self.config.strict_response_validation:
                # Replace with generic error response
                error_response = ErrorResponse(
                    success=False,
                    message="Internal response validation error",
                    error_code="RESPONSE_VALIDATION_INTERNAL_ERROR",
                    error_type="InternalError",
                    timestamp=datetime.utcnow()
                )
                
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                response.headers["content-type"] = "application/json"
                response._content = json.dumps(error_response.dict()).encode()
    
    async def _validate_response_size(self, response: Response):
        """Validate response size against configured limits."""
        if hasattr(response, '_content') and response._content:
            content_length = len(response._content)
            
            if content_length > self.config.max_response_size:
                raise ValidationError(
                    f"Response size {content_length} exceeds maximum allowed {self.config.max_response_size}",
                    model=BaseResponse
                )
    
    async def _validate_response_headers(self, response: Response, endpoint: str):
        """Validate response headers for security and compliance."""
        required_headers = self.config.required_response_headers.get(endpoint, [])
        
        for header in required_headers:
            if header.lower() not in [h.lower() for h in response.headers.keys()]:
                raise ValidationError(
                    f"Required response header '{header}' missing",
                    model=BaseResponse
                )
        
        # Validate security headers
        security_headers = {
            'x-content-type-options': 'nosniff',
            'x-frame-options': ['DENY', 'SAMEORIGIN'],
            'x-xss-protection': '1; mode=block'
        }
        
        for header, expected_values in security_headers.items():
            if header in response.headers:
                actual_value = response.headers[header].lower()
                if isinstance(expected_values, list):
                    if actual_value not in [v.lower() for v in expected_values]:
                        logger.warning(f"Security header '{header}' has unexpected value: {actual_value}")
                else:
                    if actual_value != expected_values.lower():
                        logger.warning(f"Security header '{header}' has unexpected value: {actual_value}")
    
    async def _validate_response_content_type(self, response: Response, endpoint: str):
        """Validate response content-type matches expected format."""
        if 'content-type' not in response.headers:
            return  # Some responses may not have content-type
        
        content_type = response.headers['content-type'].lower()
        
        # Define expected content types for different endpoints
        expected_content_types = {
            '/health': 'application/json',
            '/api/v1/': 'application/json',  # Default for API endpoints
            '/docs': 'text/html',
            '/openapi.json': 'application/json'
        }
        
        # Check if endpoint matches any pattern
        for pattern, expected_type in expected_content_types.items():
            if endpoint.startswith(pattern):
                if not content_type.startswith(expected_type):
                    logger.warning(
                        f"Content-type mismatch for {endpoint}: "
                        f"expected {expected_type}, got {content_type}"
                    )
                break
    
    async def _validate_response_schema(self, response: Response, endpoint: str):
        """Validate JSON response against Pydantic schemas."""
        if not hasattr(response, '_content') or not response._content:
            return
        
        try:
            response_data = json.loads(response._content.decode())
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in response: {str(e)}", model=BaseResponse)
        
        # Determine expected schema based on endpoint and response structure
        schema_class = self._get_response_schema(endpoint, response_data, response.status_code)
        
        if schema_class:
            try:
                # Validate against schema
                validated_data = schema_class(**response_data)
                
                # Update response with validated data (ensures consistency)
                response._content = json.dumps(validated_data.dict()).encode()
                
            except ValidationError as e:
                # Log validation details
                logger.error(f"Response schema validation failed for {endpoint}: {str(e)}")
                raise e
    
    def _get_response_schema(self, endpoint: str, response_data: dict, status_code: int) -> Optional[type]:
        """Determine appropriate Pydantic schema for response validation."""
        # Error responses
        if status_code >= 400 or not response_data.get('success', True):
            return ErrorResponse
        
        # Health check responses
        if endpoint.startswith('/health'):
            from app.schemas.common import HealthResponse
            return HealthResponse
        
        # Check if response follows BaseResponse pattern
        if 'success' in response_data and 'timestamp' in response_data:
            return BaseResponse
        
        # For paginated responses
        if 'items' in response_data and 'total_count' in response_data:
            from app.schemas.common import PaginationResponse
            return PaginationResponse
        
        # Default to BaseResponse for API endpoints
        if endpoint.startswith('/api/'):
            return BaseResponse
        
        return None
    
    async def _validate_response_time_sla(self, response: Response, endpoint: str, start_time: float):
        """Validate response time against SLA requirements."""
        response_time = time.time() - start_time
        
        # Define SLA thresholds (in seconds)
        sla_thresholds = {
            '/health': 0.1,      # Health checks should be very fast
            '/api/v1/': 2.0,     # API endpoints
            '/docs': 1.0,        # Documentation
            'default': 5.0       # Default threshold
        }
        
        # Find applicable threshold
        threshold = sla_thresholds['default']
        for pattern, sla_time in sla_thresholds.items():
            if pattern != 'default' and endpoint.startswith(pattern):
                threshold = sla_time
                break
        
        if response_time > threshold:
            logger.warning(
                f"Response time SLA violation for {endpoint}: "
                f"{response_time:.3f}s > {threshold}s"
            )
            
            # Add performance warning header
            response.headers['x-performance-warning'] = f"slow-response-{response_time:.3f}s"
            
            # Update metrics
            self.metrics.sla_violations += 1
    
    async def _validate_response_security(self, response: Response):
        """Validate response for security compliance."""
        if not hasattr(response, '_content') or not response._content:
            return
        
        try:
            content = response._content.decode()
            
            # Check for potential sensitive data exposure
            sensitive_patterns = [
                r'password["\s]*[:=]["\s]*[^"\s,}]+',  # Password fields
                r'secret["\s]*[:=]["\s]*[^"\s,}]+',    # Secret fields
                r'token["\s]*[:=]["\s]*[^"\s,}]+',     # Token fields (except safe ones)
                r'key["\s]*[:=]["\s]*[^"\s,}]+',       # Key fields
                r'\b[A-Za-z0-9+/]{40,}\b',             # Base64 encoded secrets
                r'\b[0-9a-f]{32,}\b',                  # Hex encoded secrets
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    logger.warning(f"Potential sensitive data detected in response")
                    
                    # Add security warning header
                    response.headers['x-security-warning'] = 'potential-sensitive-data'
                    break
            
            # Validate against XSS in JSON strings
            if self._is_json_response(response):
                self._check_xss_in_json(content)
                
        except UnicodeDecodeError:
            # Binary content, skip text-based security checks
            pass
    
    async def _sanitize_response_data(self, response: Response):
        """Sanitize response data to remove or mask sensitive information."""
        if not hasattr(response, '_content') or not response._content:
            return
        
        if not self._is_json_response(response):
            return
        
        try:
            response_data = json.loads(response._content.decode())
            sanitized_data = self._sanitize_json_data(response_data)
            
            # Update response with sanitized data
            response._content = json.dumps(sanitized_data).encode()
            
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Skip sanitization for non-JSON or binary content
            pass
    
    def _sanitize_json_data(self, data: Any) -> Any:
        """Recursively sanitize JSON data."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Mask sensitive fields
                if key.lower() in ['password', 'secret', 'token', 'key', 'api_key']:
                    sanitized[key] = '***MASKED***'
                else:
                    sanitized[key] = self._sanitize_json_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_json_data(item) for item in data]
        elif isinstance(data, str):
            # Check for potential sensitive data patterns
            if re.match(r'^[A-Za-z0-9+/]{40,}={0,2}$', data):  # Base64 pattern
                return '***MASKED_BASE64***'
            elif re.match(r'^[0-9a-f]{32,}$', data):  # Hex pattern
                return '***MASKED_HEX***'
            return data
        else:
            return data
    
    def _is_json_response(self, response: Response) -> bool:
        """Check if response is JSON content."""
        content_type = response.headers.get('content-type', '').lower()
        return 'application/json' in content_type
    
    def _check_xss_in_json(self, content: str):
        """Check for potential XSS patterns in JSON content."""
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                logger.warning("Potential XSS content detected in JSON response")
                break
    
    def _format_validation_errors(self, validation_error: ValidationError) -> List[Dict[str, Any]]:
        """Format Pydantic validation errors for API response."""
        errors = []
        for error in validation_error.errors():
            error_dict = {
                'field': '.'.join(str(loc) for loc in error['loc']),
                'message': error['msg'],
                'type': error['type']
            }
            if self.config.include_error_details and 'ctx' in error:
                error_dict['context'] = error['ctx']
            errors.append(error_dict)
        return errors
    
    def _create_error_response(
        self,
        message: str,
        status_code: int,
        details: Optional[List[Dict[str, Any]]] = None
    ) -> JSONResponse:
        """Create standardized error response."""
        # Truncate message if too long
        if len(message) > self.config.max_error_message_length:
            message = message[:self.config.max_error_message_length] + "..."
        
        error_response = ErrorResponse(
            success=False,
            message=message,
            error_code=f"VALIDATION_ERROR_{status_code}",
            timestamp=datetime.utcnow(),
            details=details if self.config.include_error_details else None
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.dict(exclude_none=True)
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current validation metrics."""
        return self.metrics.get_summary()
    
    def register_schema(self, endpoint: str, method: str, schema_class: BaseModel):
        """Register a Pydantic schema for endpoint validation."""
        key = f"{method.upper()}:{endpoint}"
        self.schema_registry[key] = schema_class
        logger.info("Registered schema %s for endpoint %s", schema_class.__name__, key)
    
    def update_config(self, new_config: ValidationConfig):
        """Update middleware configuration at runtime."""
        self.config = new_config
        
        # Reinitialize components with new config
        if self.config.enable_rate_limiting:
            self.rate_limiter = RateLimiter(
                self.config.rate_limit_requests,
                self.config.rate_limit_window
            )
        else:
            self.rate_limiter = None
        
        if self.config.enable_caching:
            self.cache = ValidationCache(
                self.config.max_cache_size,
                self.config.cache_ttl
            )
        else:
            self.cache = None
        
        self.security_validator = SecurityValidator(self.config)
        logger.info("ValidationMiddleware configuration updated")




# Utility functions for easy integration

def create_validation_middleware(
    app: ASGIApp,
    config: Optional[Dict[str, Any]] = None,
    schema_registry: Optional[Dict[str, BaseModel]] = None
) -> ValidationMiddleware:
    """
    Factory function to create validation middleware with configuration.
    
    Args:
        app: FastAPI application instance
        config: Configuration dictionary for ValidationConfig
        schema_registry: Dictionary mapping endpoints to Pydantic schemas
    
    Returns:
        Configured ValidationMiddleware instance
    """
    validation_config = ValidationConfig(**(config or {}))
    return ValidationMiddleware(app, validation_config, schema_registry)


def validation_decorator(schema_class: BaseModel):
    """
    Decorator for endpoint functions to register validation schemas.
    
    Usage:
        @validation_decorator(MyRequestSchema)
        async def my_endpoint(request: Request):
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        # Store schema information for later registration
        wrapper._validation_schema = schema_class
        return wrapper
    
    return decorator
    
    async def _validate_file_uploads(self, request: Request, endpoint: str) -> ValidationResult:
        """Validate file uploads with size/type constraints and security scanning."""
        try:
            # Check if request has file uploads
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("multipart/form-data"):
                return ValidationResult(is_valid=True)  # No files to validate
            
            form_data = await request.form()
            files = []
            errors = []
            
            # Extract file uploads from form data
            for field_name, field_value in form_data.items():
                if hasattr(field_value, 'filename') and hasattr(field_value, 'file'):
                    files.append((field_name, field_value))
            
            if not files:
                return ValidationResult(is_valid=True)  # No files found
            
            # Get file upload validation rules for this endpoint
            endpoint_key = f"{request.method}:{endpoint}"
            file_rules = {}
            if hasattr(self, 'file_upload_rules') and endpoint_key in self.file_upload_rules:
                file_rules = self.file_upload_rules[endpoint_key]
            
            # Apply default file validation rules if no specific rules exist
            default_rules = {
                'max_file_size': self.config.max_request_size or 10 * 1024 * 1024,  # 10MB default
                'allowed_extensions': ['.pdf', '.txt', '.doc', '.docx', '.jpg', '.jpeg', '.png'],
                'allowed_mime_types': [
                    'application/pdf', 'text/plain', 'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'image/jpeg', 'image/png'
                ],
                'max_files': 10,
                'scan_for_malware': True
            }
            
            # Merge with endpoint-specific rules
            validation_rules = {**default_rules, **file_rules}
            
            # Check maximum number of files
            if len(files) > validation_rules.get('max_files', 10):
                errors.append({
                    'field': 'files',
                    'message': f'Too many files uploaded. Maximum allowed: {validation_rules["max_files"]}',
                    'type': 'count_error'
                })
            
            validated_files = []
            
            for field_name, file_upload in files:
                file_errors = []
                
                # Check file size
                file_size = 0
                if hasattr(file_upload, 'size'):
                    file_size = file_upload.size
                else:
                    # Read file to get size
                    content = await file_upload.read()
                    file_size = len(content)
                    # Reset file pointer
                    await file_upload.seek(0)
                
                max_size = validation_rules.get('max_file_size', 10 * 1024 * 1024)
                if file_size > max_size:
                    file_errors.append({
                        'field': f'files.{field_name}',
                        'message': f'File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)',
                        'type': 'size_error'
                    })
                
                # Check file extension
                filename = getattr(file_upload, 'filename', '')
                if filename:
                    file_ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
                    allowed_extensions = validation_rules.get('allowed_extensions', [])
                    if allowed_extensions and file_ext not in allowed_extensions:
                        file_errors.append({
                            'field': f'files.{field_name}',
                            'message': f'File extension "{file_ext}" not allowed. Allowed: {", ".join(allowed_extensions)}',
                            'type': 'extension_error'
                        })
                
                # Check MIME type
                content_type = getattr(file_upload, 'content_type', '')
                allowed_mime_types = validation_rules.get('allowed_mime_types', [])
                if allowed_mime_types and content_type not in allowed_mime_types:
                    file_errors.append({
                        'field': f'files.{field_name}',
                        'message': f'MIME type "{content_type}" not allowed. Allowed: {", ".join(allowed_mime_types)}',
                        'type': 'mime_type_error'
                    })
                
                # Security scanning
                if validation_rules.get('scan_for_malware', True):
                    try:
                        # Read file content for security scanning
                        file_content = await file_upload.read()
                        await file_upload.seek(0)  # Reset file pointer
                        
                        # Check for malicious patterns in file content
                        if self.security_validator.contains_malicious_patterns(file_content.decode('utf-8', errors='ignore')):
                            file_errors.append({
                                'field': f'files.{field_name}',
                                'message': 'File contains potentially malicious content',
                                'type': 'security_error'
                            })
                        
                        # Check file header/magic bytes for consistency with extension
                        if filename and not self._validate_file_magic_bytes(file_content, file_ext):
                            file_errors.append({
                                'field': f'files.{field_name}',
                                'message': 'File content does not match file extension',
                                'type': 'content_mismatch_error'
                            })
                    
                    except UnicodeDecodeError:
                        # Binary file - perform basic checks only
                        pass
                    except Exception as e:
                        logger.warning("File security scan failed for %s: %s", filename, str(e))
                
                # Custom validation function
                if 'validator' in validation_rules and callable(validation_rules['validator']):
                    try:
                        if not validation_rules['validator'](file_upload):
                            file_errors.append({
                                'field': f'files.{field_name}',
                                'message': validation_rules.get('validator_message', 'Custom file validation failed'),
                                'type': 'validation_error'
                            })
                    except Exception as ve:
                        file_errors.append({
                            'field': f'files.{field_name}',
                            'message': f'File validation error: {str(ve)}',
                            'type': 'validation_error'
                        })
                
                if file_errors:
                    errors.extend(file_errors)
                else:
                    validated_files.append({
                        'field_name': field_name,
                        'filename': filename,
                        'content_type': content_type,
                        'size': file_size,
                        'file': file_upload
                    })
            
            if errors:
                return ValidationResult(
                    is_valid=False,
                    error_message="File upload validation failed",
                    details=errors
                )
            
            return ValidationResult(
                is_valid=True,
                validated_data={'files': validated_files}
            )
            
        except Exception as e:
            logger.error("File upload validation error: %s", str(e))
            return ValidationResult(
                is_valid=False,
                error_message="File upload validation failed"
            )
    
    def _validate_file_magic_bytes(self, file_content: bytes, file_extension: str) -> bool:
        """Validate file content matches the expected file type based on magic bytes."""
        if not file_content or len(file_content) < 4:
            return True  # Cannot validate, assume valid
        
        # Common file magic bytes
        magic_bytes = {
            '.pdf': [b'%PDF'],
            '.jpg': [b'\xff\xd8\xff'],
            '.jpeg': [b'\xff\xd8\xff'],
            '.png': [b'\x89PNG\r\n\x1a\n'],
            '.gif': [b'GIF87a', b'GIF89a'],
            '.zip': [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'],
            '.docx': [b'PK\x03\x04'],  # DOCX is a ZIP file
            '.xlsx': [b'PK\x03\x04'],  # XLSX is a ZIP file
        }
        
        expected_magic = magic_bytes.get(file_extension.lower(), [])
        if not expected_magic:
            return True  # No magic bytes defined for this extension
        
        # Check if file starts with any of the expected magic bytes
        for magic in expected_magic:
            if file_content.startswith(magic):
                return True
        
        return False
    
    async def _aggregate_validation_errors(self, *validation_results: ValidationResult) -> ValidationResult:
        """Aggregate multiple validation results into a single result."""
        all_errors = []
        all_validated_data = {}
        
        for result in validation_results:
            if not result.is_valid:
                if result.details:
                    if isinstance(result.details, list):
                        all_errors.extend(result.details)
                    else:
                        all_errors.append(result.details)
                else:
                    # Create error from error_message if no details
                    all_errors.append({
                        'field': 'general',
                        'message': result.error_message or 'Validation failed',
                        'type': 'validation_error'
                    })
            
            # Merge validated data
            if result.validated_data:
                all_validated_data.update(result.validated_data)
        
        if all_errors:
            # Group errors by field for better organization
            grouped_errors = {}
            for error in all_errors:
                field = error.get('field', 'general')
                if field not in grouped_errors:
                    grouped_errors[field] = []
                grouped_errors[field].append(error)
            
            # Create summary message
            error_count = len(all_errors)
            field_count = len(grouped_errors)
            summary_message = f"Validation failed with {error_count} error(s) across {field_count} field(s)"
            
            return ValidationResult(
                is_valid=False,
                error_message=summary_message,
                details={
                    'summary': summary_message,
                    'error_count': error_count,
                    'field_count': field_count,
                    'errors_by_field': grouped_errors,
                    'all_errors': all_errors
                }
            )
        
        return ValidationResult(
            is_valid=True,
            validated_data=all_validated_data
        )