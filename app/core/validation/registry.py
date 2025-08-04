"""
Validation schema registry for the PDF2Markdown microservice.

This module provides a centralized registry for managing Pydantic schemas
associated with specific endpoints, enabling automatic validation based
on route patterns and HTTP methods.
"""

import re
from typing import Dict, List, Optional, Type, Union, Pattern
from collections import defaultdict
import logging

from pydantic import BaseModel

from .config import ValidationConfig
from .errors import ValidationConfigurationError

logger = logging.getLogger(__name__)


class EndpointPattern:
    """
    Represents an endpoint pattern for schema registration.
    
    Supports exact matches, wildcard patterns, and regex patterns
    for flexible endpoint matching.
    """
    
    def __init__(
        self,
        pattern: str,
        method: str = "*",
        is_regex: bool = False
    ):
        """
        Initialize endpoint pattern.
        
        Args:
            pattern: Endpoint pattern (e.g., "/api/v1/documents", "/api/*/users")
            method: HTTP method ("GET", "POST", "*" for all)
            is_regex: Whether pattern is a regex
        """
        self.pattern = pattern
        self.method = method.upper()
        self.is_regex = is_regex
        
        if is_regex:
            try:
                self.compiled_pattern = re.compile(pattern)
            except re.error as e:
                raise ValidationConfigurationError(
                    f"Invalid regex pattern '{pattern}': {e}",
                    config_issue=f"Regex compilation failed: {e}"
                )
        else:
            # Convert wildcard pattern to regex
            escaped_pattern = re.escape(pattern)
            # Replace escaped wildcards with regex equivalents
            regex_pattern = escaped_pattern.replace(r"\*", "[^/]*")
            regex_pattern = f"^{regex_pattern}$"
            
            try:
                self.compiled_pattern = re.compile(regex_pattern)
            except re.error as e:
                raise ValidationConfigurationError(
                    f"Failed to compile pattern '{pattern}': {e}",
                    config_issue=f"Pattern compilation failed: {e}"
                )
    
    def matches(self, endpoint: str, method: str) -> bool:
        """
        Check if endpoint and method match this pattern.
        
        Args:
            endpoint: Endpoint path
            method: HTTP method
            
        Returns:
            True if matches, False otherwise
        """
        # Check method match
        if self.method != "*" and self.method != method.upper():
            return False
        
        # Check endpoint match
        return bool(self.compiled_pattern.match(endpoint))
    
    def __str__(self) -> str:
        return f"{self.method}:{self.pattern}"
    
    def __repr__(self) -> str:
        return f"EndpointPattern(pattern='{self.pattern}', method='{self.method}', is_regex={self.is_regex})"


class SchemaRegistration:
    """
    Represents a schema registration with metadata.
    """
    
    def __init__(
        self,
        schema_class: Type[BaseModel],
        endpoint_pattern: EndpointPattern,
        description: Optional[str] = None,
        priority: int = 0,
        enabled: bool = True
    ):
        """
        Initialize schema registration.
        
        Args:
            schema_class: Pydantic model class
            endpoint_pattern: Endpoint pattern for matching
            description: Optional description
            priority: Priority for pattern matching (higher = more priority)
            enabled: Whether registration is enabled
        """
        self.schema_class = schema_class
        self.endpoint_pattern = endpoint_pattern
        self.description = description or f"Schema for {schema_class.__name__}"
        self.priority = priority
        self.enabled = enabled
        self.registration_id = f"{endpoint_pattern}:{schema_class.__name__}"
    
    def __str__(self) -> str:
        return f"{self.endpoint_pattern} -> {self.schema_class.__name__}"
    
    def __repr__(self) -> str:
        return (
            f"SchemaRegistration("
            f"schema={self.schema_class.__name__}, "
            f"pattern={self.endpoint_pattern}, "
            f"priority={self.priority}, "
            f"enabled={self.enabled})"
        )


class ValidationRegistry:
    """
    Centralized registry for validation schemas.
    
    Manages the mapping between endpoints and their corresponding
    Pydantic schemas for automatic validation.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validation registry.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self._registrations: List[SchemaRegistration] = []
        self._method_registrations: Dict[str, List[SchemaRegistration]] = defaultdict(list)
        self._exact_matches: Dict[str, SchemaRegistration] = {}
        
        # Statistics
        self._stats = {
            'total_registrations': 0,
            'enabled_registrations': 0,
            'pattern_matches': 0,
            'exact_matches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Simple cache for recent lookups
        self._lookup_cache: Dict[str, Optional[SchemaRegistration]] = {}
        self._max_cache_size = 100
    
    def register(
        self,
        endpoint_pattern: Union[str, EndpointPattern],
        schema_class: Type[BaseModel],
        method: str = "*",
        description: Optional[str] = None,
        priority: int = 0,
        enabled: bool = True,
        is_regex: bool = False
    ) -> str:
        """
        Register a schema for an endpoint pattern.
        
        Args:
            endpoint_pattern: Endpoint pattern or EndpointPattern object
            schema_class: Pydantic model class
            method: HTTP method
            description: Optional description
            priority: Priority for pattern matching
            enabled: Whether registration is enabled
            is_regex: Whether pattern is regex (ignored if endpoint_pattern is EndpointPattern)
            
        Returns:
            Registration ID
            
        Raises:
            ValidationConfigurationError: If registration fails
        """
        try:
            # Create EndpointPattern if needed
            if isinstance(endpoint_pattern, str):
                pattern = EndpointPattern(endpoint_pattern, method, is_regex)
            else:
                pattern = endpoint_pattern
            
            # Validate schema class
            if not issubclass(schema_class, BaseModel):
                raise ValidationConfigurationError(
                    f"Schema class {schema_class.__name__} must inherit from BaseModel",
                    config_issue="Invalid schema class type"
                )
            
            # Create registration
            registration = SchemaRegistration(
                schema_class=schema_class,
                endpoint_pattern=pattern,
                description=description,
                priority=priority,
                enabled=enabled
            )
            
            # Check for duplicate exact matches
            exact_key = f"{pattern.method}:{pattern.pattern}"
            if not pattern.is_regex and "*" not in pattern.pattern:
                if exact_key in self._exact_matches:
                    existing = self._exact_matches[exact_key]
                    logger.warning(
                        "Overriding existing exact match registration: %s -> %s (was %s)",
                        exact_key, schema_class.__name__, existing.schema_class.__name__
                    )
                self._exact_matches[exact_key] = registration
                self._stats['exact_matches'] += 1
            
            # Add to registrations list (sorted by priority)
            self._registrations.append(registration)
            self._registrations.sort(key=lambda r: r.priority, reverse=True)
            
            # Add to method-specific registrations
            self._method_registrations[pattern.method].append(registration)
            if pattern.method != "*":
                self._method_registrations["*"].append(registration)
            
            # Update statistics
            self._stats['total_registrations'] += 1
            if enabled:
                self._stats['enabled_registrations'] += 1
            
            # Clear cache since we have new registrations
            self._lookup_cache.clear()
            
            logger.info(
                "Registered schema %s for pattern %s (priority: %d, enabled: %s)",
                schema_class.__name__, pattern, priority, enabled
            )
            
            return registration.registration_id
            
        except Exception as e:
            logger.error("Failed to register schema: %s", str(e))
            raise ValidationConfigurationError(
                f"Schema registration failed: {str(e)}",
                config_issue=f"Registration error: {str(e)}"
            )
    
    def register_multiple(
        self,
        registrations: List[Dict[str, any]]
    ) -> List[str]:
        """
        Register multiple schemas at once.
        
        Args:
            registrations: List of registration dictionaries
            
        Returns:
            List of registration IDs
        """
        registration_ids = []
        
        for reg_config in registrations:
            try:
                reg_id = self.register(**reg_config)
                registration_ids.append(reg_id)
            except Exception as e:
                logger.error("Failed to register schema from config %s: %s", reg_config, str(e))
                # Continue with other registrations
                continue
        
        return registration_ids
    
    def unregister(self, registration_id: str) -> bool:
        """
        Unregister a schema by registration ID.
        
        Args:
            registration_id: Registration ID to remove
            
        Returns:
            True if unregistered, False if not found
        """
        # Find and remove registration
        for i, registration in enumerate(self._registrations):
            if registration.registration_id == registration_id:
                # Remove from main list
                removed_registration = self._registrations.pop(i)
                
                # Remove from method registrations
                method = removed_registration.endpoint_pattern.method
                if method in self._method_registrations:
                    self._method_registrations[method] = [
                        r for r in self._method_registrations[method]
                        if r.registration_id != registration_id
                    ]
                
                # Remove from exact matches if present
                exact_key = f"{method}:{removed_registration.endpoint_pattern.pattern}"
                if exact_key in self._exact_matches:
                    del self._exact_matches[exact_key]
                    self._stats['exact_matches'] -= 1
                
                # Update statistics
                self._stats['total_registrations'] -= 1
                if removed_registration.enabled:
                    self._stats['enabled_registrations'] -= 1
                
                # Clear cache
                self._lookup_cache.clear()
                
                logger.info("Unregistered schema: %s", registration_id)
                return True
        
        logger.warning("Registration ID not found: %s", registration_id)
        return False
    
    def lookup(
        self,
        endpoint: str,
        method: str = "GET"
    ) -> Optional[Type[BaseModel]]:
        """
        Look up schema for an endpoint and method.
        
        Args:
            endpoint: Endpoint path
            method: HTTP method
            
        Returns:
            Pydantic schema class if found, None otherwise
        """
        # Create cache key
        cache_key = f"{method.upper()}:{endpoint}"
        
        # Check cache first
        if cache_key in self._lookup_cache:
            self._stats['cache_hits'] += 1
            cached_registration = self._lookup_cache[cache_key]
            return cached_registration.schema_class if cached_registration else None
        
        self._stats['cache_misses'] += 1
        
        # Try exact match first
        exact_key = f"{method.upper()}:{endpoint}"
        if exact_key in self._exact_matches:
            registration = self._exact_matches[exact_key]
            if registration.enabled:
                self._cache_result(cache_key, registration)
                return registration.schema_class
        
        # Try wildcard exact match
        wildcard_key = f"*:{endpoint}"
        if wildcard_key in self._exact_matches:
            registration = self._exact_matches[wildcard_key]
            if registration.enabled:
                self._cache_result(cache_key, registration)
                return registration.schema_class
        
        # Pattern matching
        method_upper = method.upper()
        candidates = []
        
        # Get method-specific registrations
        candidates.extend(self._method_registrations.get(method_upper, []))
        
        # Add wildcard registrations if not already included
        if method_upper != "*":
            candidates.extend(self._method_registrations.get("*", []))
        
        # Find matching patterns (sorted by priority)
        for registration in candidates:
            if not registration.enabled:
                continue
            
            if registration.endpoint_pattern.matches(endpoint, method):
                self._stats['pattern_matches'] += 1
                self._cache_result(cache_key, registration)
                return registration.schema_class
        
        # No match found
        self._cache_result(cache_key, None)
        return None
    
    def _cache_result(self, cache_key: str, registration: Optional[SchemaRegistration]):
        """Cache lookup result."""
        if len(self._lookup_cache) >= self._max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._lookup_cache))
            del self._lookup_cache[oldest_key]
        
        self._lookup_cache[cache_key] = registration
    
    def list_registrations(
        self,
        enabled_only: bool = False,
        method: Optional[str] = None
    ) -> List[SchemaRegistration]:
        """
        List all registrations.
        
        Args:
            enabled_only: Only return enabled registrations
            method: Filter by HTTP method
            
        Returns:
            List of registrations
        """
        registrations = self._registrations
        
        if method:
            method_upper = method.upper()
            registrations = [
                r for r in registrations
                if r.endpoint_pattern.method == method_upper or r.endpoint_pattern.method == "*"
            ]
        
        if enabled_only:
            registrations = [r for r in registrations if r.enabled]
        
        return registrations
    
    def enable_registration(self, registration_id: str) -> bool:
        """
        Enable a registration.
        
        Args:
            registration_id: Registration ID
            
        Returns:
            True if enabled, False if not found
        """
        for registration in self._registrations:
            if registration.registration_id == registration_id:
                if not registration.enabled:
                    registration.enabled = True
                    self._stats['enabled_registrations'] += 1
                    self._lookup_cache.clear()
                    logger.info("Enabled registration: %s", registration_id)
                return True
        
        return False
    
    def disable_registration(self, registration_id: str) -> bool:
        """
        Disable a registration.
        
        Args:
            registration_id: Registration ID
            
        Returns:
            True if disabled, False if not found
        """
        for registration in self._registrations:
            if registration.registration_id == registration_id:
                if registration.enabled:
                    registration.enabled = False
                    self._stats['enabled_registrations'] -= 1
                    self._lookup_cache.clear()
                    logger.info("Disabled registration: %s", registration_id)
                return True
        
        return False
    
    def clear(self):
        """Clear all registrations."""
        self._registrations.clear()
        self._method_registrations.clear()
        self._exact_matches.clear()
        self._lookup_cache.clear()
        
        # Reset statistics
        self._stats = {
            'total_registrations': 0,
            'enabled_registrations': 0,
            'pattern_matches': 0,
            'exact_matches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("Cleared all schema registrations")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary of statistics
        """
        cache_hit_rate = 0.0
        total_lookups = self._stats['cache_hits'] + self._stats['cache_misses']
        if total_lookups > 0:
            cache_hit_rate = self._stats['cache_hits'] / total_lookups
        
        return {
            **self._stats,
            'cache_hit_rate': round(cache_hit_rate, 4),
            'cache_size': len(self._lookup_cache),
            'method_registrations': {
                method: len(regs) for method, regs in self._method_registrations.items()
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate registry configuration and return any issues.
        
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check for duplicate patterns with same priority
        pattern_priorities = defaultdict(list)
        for registration in self._registrations:
            if registration.enabled:
                key = str(registration.endpoint_pattern)
                pattern_priorities[key].append(registration.priority)
        
        for pattern, priorities in pattern_priorities.items():
            if len(set(priorities)) < len(priorities):
                issues.append(f"Duplicate priorities for pattern {pattern}: {priorities}")
        
        # Check for unreachable patterns
        for i, registration in enumerate(self._registrations):
            if not registration.enabled:
                continue
            
            # Check if this pattern is overshadowed by higher priority patterns
            for j, other_registration in enumerate(self._registrations[:i]):
                if not other_registration.enabled:
                    continue
                
                if (other_registration.priority > registration.priority and
                    self._patterns_overlap(registration.endpoint_pattern, other_registration.endpoint_pattern)):
                    issues.append(
                        f"Pattern {registration.endpoint_pattern} may be unreachable due to "
                        f"higher priority pattern {other_registration.endpoint_pattern}"
                    )
        
        return issues
    
    def _patterns_overlap(self, pattern1: EndpointPattern, pattern2: EndpointPattern) -> bool:
        """
        Check if two patterns might overlap.
        
        This is a simple heuristic check - not exhaustive.
        """
        # If methods don't overlap, patterns don't overlap
        if (pattern1.method != "*" and pattern2.method != "*" and 
            pattern1.method != pattern2.method):
            return False
        
        # Simple string comparison for basic overlap detection
        # This could be made more sophisticated
        return (pattern1.pattern == pattern2.pattern or
                "*" in pattern1.pattern or "*" in pattern2.pattern or
                pattern1.is_regex or pattern2.is_regex)


# Global registry instance
_global_registry: Optional[ValidationRegistry] = None


def get_global_registry() -> ValidationRegistry:
    """
    Get the global validation registry instance.
    
    Returns:
        Global ValidationRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ValidationRegistry()
    return _global_registry


def set_global_registry(registry: ValidationRegistry):
    """
    Set the global validation registry instance.
    
    Args:
        registry: ValidationRegistry instance to set as global
    """
    global _global_registry
    _global_registry = registry


# Export all classes and functions
__all__ = [
    "EndpointPattern",
    "SchemaRegistration", 
    "ValidationRegistry",
    "get_global_registry",
    "set_global_registry"
]