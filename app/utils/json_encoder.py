"""
Custom JSON encoder for handling datetime and other non-serializable objects.

This module provides a custom JSON encoder that properly handles datetime objects
and other common Python types that are not natively JSON serializable.
"""

import json
from datetime import datetime, date, time
from decimal import Decimal
from uuid import UUID
from typing import Any


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime objects and other common types.
    
    This encoder automatically converts:
    - datetime objects to ISO format strings
    - date objects to ISO format strings  
    - time objects to ISO format strings
    - Decimal objects to float
    - UUID objects to string
    - Sets to lists
    """
    
    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-serializable format.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return obj.__dict__
        
        # Let the base class handle other types
        return super().default(obj)


def json_dumps(obj: Any, **kwargs) -> str:
    """
    JSON dumps with custom encoder.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation
    """
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)


def safe_json_response(data: Any) -> dict:
    """
    Safely convert data to JSON-serializable format.
    
    Args:
        data: Data to make JSON-safe
        
    Returns:
        JSON-serializable dictionary
    """
    try:
        # Test if data is already JSON serializable
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        # Use custom encoder to make it serializable
        json_str = json_dumps(data)
        return json.loads(json_str)
