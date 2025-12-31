"""
Middleware package for the PDF2Markdown microservice.

This package contains all middleware components including validation,
authentication, monitoring, and error handling middleware.
"""

from .validation import ValidationMiddleware

__all__ = ["ValidationMiddleware"]
