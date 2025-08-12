"""
Enhanced Logging Integration for PDF Processing Service

This module provides enhanced logging capabilities that integrate existing
extractor.py functionality with the production FastAPI application structure.
"""

import logging
import logging.config
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import wraps
import inspect

from app.config import get_settings


class PDFProcessingLogger:
    """
    Enhanced logger for PDF processing operations.
    
    Provides structured logging with context information, performance tracking,
    and integration between existing extractor.py and new FastAPI components.
    """
    
    def __init__(self, name: str = __name__):
        """
        Initialize the PDF processing logger.
        
        Args:
            name: Logger name, typically __name__ from calling module
        """
        self.logger = logging.getLogger(name)
        self.settings = get_settings()
        
    def log_operation_start(self, operation: str, **context) -> str:
        """
        Log the start of a PDF processing operation.
        
        Args:
            operation: Name of the operation (e.g., "extract_markdown", "extract_tables")
            **context: Additional context information
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.info(
            f"🚀 Starting operation: {operation} [ID: {operation_id}] | Context: {context_str}"
        )
        
        return operation_id
    
    def log_operation_success(self, operation_id: str, operation: str, **results):
        """
        Log successful completion of a PDF processing operation.
        
        Args:
            operation_id: Operation ID from log_operation_start
            operation: Name of the operation
            **results: Results and metrics from the operation
        """
        results_str = ", ".join([f"{k}={v}" for k, v in results.items()])
        self.logger.info(
            f"✅ Completed operation: {operation} [ID: {operation_id}] | Results: {results_str}"
        )
    
    def log_operation_error(self, operation_id: str, operation: str, error: Exception, **context):
        """
        Log error during PDF processing operation.
        
        Args:
            operation_id: Operation ID from log_operation_start
            operation: Name of the operation
            error: Exception that occurred
            **context: Additional context information
        """
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(
            f"❌ Failed operation: {operation} [ID: {operation_id}] | "
            f"Error: {type(error).__name__}: {str(error)} | Context: {context_str}"
        )
        
        # Log full traceback at debug level
        self.logger.debug(f"Full traceback for {operation_id}:\n{traceback.format_exc()}")
    
    def log_file_processing(self, filename: str, file_size: int, operation: str):
        """
        Log file processing information.
        
        Args:
            filename: Name of the file being processed
            file_size: Size of the file in bytes
            operation: Type of processing operation
        """
        size_mb = file_size / (1024 * 1024)
        self.logger.info(
            f"📄 Processing file: {filename} ({size_mb:.2f} MB) | Operation: {operation}"
        )
    
    def log_extractor_integration(self, function_name: str, **params):
        """
        Log integration with existing extractor.py functions.
        
        Args:
            function_name: Name of the extractor function being called
            **params: Parameters passed to the function
        """
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.logger.debug(
            f"🔗 Calling extractor function: {function_name} | Params: {params_str}"
        )
    
    def log_performance_metric(self, operation: str, metric_name: str, value: Union[int, float], unit: str = ""):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (optional)
        """
        unit_str = f" {unit}" if unit else ""
        self.logger.info(
            f"📊 Performance metric | Operation: {operation} | {metric_name}: {value}{unit_str}"
        )
    
    def log_configuration_info(self):
        """Log current configuration information."""
        config = {
            "output_dir": str(self.settings.get_output_path()),
            "temp_dir": str(self.settings.get_temp_path()),
            "image_format": self.settings.default_image_format,
            "image_quality": self.settings.default_image_quality,
            "table_strategy": self.settings.default_table_strategy,
            "write_images": self.settings.write_images,
            "extract_tables": self.settings.extract_tables,
            "extract_images": self.settings.extract_images,
            "max_file_size": self.settings.max_file_size,
            "allowed_extensions": self.settings.allowed_extensions,
        }
        self.logger.info(f"⚙️ Configuration loaded: {config}")
    
    def log_health_check(self, status: str, **details):
        """
        Log health check information.
        
        Args:
            status: Health status (healthy, degraded, unhealthy)
            **details: Additional health check details
        """
        details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
        status_emoji = {"healthy": "💚", "degraded": "💛", "unhealthy": "❤️"}.get(status, "❓")
        
        self.logger.info(f"{status_emoji} Health check: {status} | Details: {details_str}")


def log_function_call(logger: Optional[PDFProcessingLogger] = None):
    """
    Decorator to automatically log function calls with parameters and results.
    
    Args:
        logger: Optional logger instance. If None, creates a new one.
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            func_logger = logger or PDFProcessingLogger(func.__module__)
            
            # Get function signature for better logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Filter out sensitive or large parameters for logging
            safe_params = {}
            for name, value in bound_args.arguments.items():
                if name in ['file_content', 'file_data', 'upload_file']:
                    safe_params[name] = f"<{type(value).__name__}>"
                elif isinstance(value, (str, int, float, bool, type(None))):
                    safe_params[name] = value
                else:
                    safe_params[name] = f"<{type(value).__name__}>"
            
            # Log function entry
            operation_id = func_logger.log_operation_start(
                f"{func.__module__}.{func.__name__}",
                **safe_params
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful completion
                result_info = {}
                if hasattr(result, '__len__') and not isinstance(result, str):
                    result_info['result_length'] = len(result)
                elif isinstance(result, (int, float, bool)):
                    result_info['result'] = result
                else:
                    result_info['result_type'] = type(result).__name__
                
                func_logger.log_operation_success(
                    operation_id,
                    f"{func.__module__}.{func.__name__}",
                    **result_info
                )
                
                return result
                
            except Exception as e:
                # Log error
                func_logger.log_operation_error(
                    operation_id,
                    f"{func.__module__}.{func.__name__}",
                    e,
                    **safe_params
                )
                raise
                
        return wrapper
    return decorator


def setup_logging_for_extractor():
    """
    Set up logging configuration for existing extractor.py integration.
    
    This function ensures that existing extractor functions have proper
    logging integration with the new FastAPI application.
    """
    # Get the extractor module logger
    extractor_logger = logging.getLogger('extractor')
    
    # Ensure it uses the same configuration as the main app
    settings = get_settings()
    
    # Add a handler if none exists
    if not extractor_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(settings.log_format)
        handler.setFormatter(formatter)
        extractor_logger.addHandler(handler)
        extractor_logger.setLevel(getattr(logging, settings.log_level))
    
    # Log integration setup
    pdf_logger = PDFProcessingLogger(__name__)
    pdf_logger.logger.info("🔧 Logging integration set up for extractor.py")


def create_request_logger(request_id: str) -> PDFProcessingLogger:
    """
    Create a logger instance for a specific request.
    
    Args:
        request_id: Unique identifier for the request
        
    Returns:
        PDFProcessingLogger instance configured for the request
    """
    logger_name = f"pdf_processing.request.{request_id}"
    return PDFProcessingLogger(logger_name)


def log_api_request(endpoint: str, method: str, **params):
    """
    Log API request information.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        **params: Request parameters (sensitive data will be filtered)
    """
    logger = PDFProcessingLogger("api")
    
    # Filter sensitive parameters
    safe_params = {}
    for key, value in params.items():
        if key in ['file', 'file_content', 'upload_file']:
            if hasattr(value, 'filename'):
                safe_params[key] = f"<File: {value.filename}>"
            else:
                safe_params[key] = f"<{type(value).__name__}>"
        else:
            safe_params[key] = value
    
    params_str = ", ".join([f"{k}={v}" for k, v in safe_params.items()])
    logger.logger.info(f"🌐 API Request: {method} {endpoint} | Params: {params_str}")


def log_api_response(endpoint: str, status_code: int, **response_info):
    """
    Log API response information.
    
    Args:
        endpoint: API endpoint path
        status_code: HTTP status code
        **response_info: Response information
    """
    logger = PDFProcessingLogger("api")
    
    status_emoji = "✅" if 200 <= status_code < 300 else "❌"
    info_str = ", ".join([f"{k}={v}" for k, v in response_info.items()])
    
    logger.logger.info(f"{status_emoji} API Response: {endpoint} | Status: {status_code} | Info: {info_str}")


class LoggingMiddleware:
    """
    FastAPI middleware for request/response logging.
    """
    
    def __init__(self, app):
        self.app = app
        self.logger = PDFProcessingLogger("middleware")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate request ID
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Log request start
            self.logger.log_operation_start(
                "http_request",
                request_id=request_id,
                method=scope.get("method"),
                path=scope.get("path"),
                client=scope.get("client")
            )
            
            # Process request
            try:
                await self.app(scope, receive, send)
                self.logger.log_operation_success(
                    request_id,
                    "http_request"
                )
            except Exception as e:
                self.logger.log_operation_error(
                    request_id,
                    "http_request",
                    e
                )
                raise
        else:
            await self.app(scope, receive, send)


# Initialize logging setup on module import
setup_logging_for_extractor()