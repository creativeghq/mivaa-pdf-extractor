"""
Timestamp utility functions for handling PostgreSQL timestamps.
"""

def normalize_timestamp(ts_string: str) -> str:
    """
    Normalize PostgreSQL timestamp to Python datetime format.
    
    Handles variable microsecond precision (1-6 digits) from PostgreSQL.
    Python's datetime.fromisoformat() requires exactly 6 digits or none.
    
    Args:
        ts_string: Timestamp string from PostgreSQL
        
    Returns:
        Normalized timestamp string compatible with datetime.fromisoformat()
        
    Examples:
        >>> normalize_timestamp("2025-11-18T18:36:51.92242+00:00")
        "2025-11-18T18:36:51.922420+00:00"
        
        >>> normalize_timestamp("2025-11-18T18:36:51.9Z")
        "2025-11-18T18:36:51.900000+00:00"
        
        >>> normalize_timestamp("2025-11-18T18:36:51Z")
        "2025-11-18T18:36:51+00:00"
    """
    if not ts_string:
        return ts_string
        
    # Replace 'Z' with '+00:00'
    ts_string = ts_string.replace("Z", "+00:00")
    
    # Handle variable microsecond precision
    if '.' in ts_string and '+' in ts_string:
        parts = ts_string.split('+')
        datetime_part = parts[0]
        tz_part = '+' + parts[1]
        
        if '.' in datetime_part:
            date_time, microseconds = datetime_part.rsplit('.', 1)
            # Pad or truncate to exactly 6 digits
            microseconds = microseconds.ljust(6, '0')[:6]
            ts_string = f"{date_time}.{microseconds}{tz_part}"
    
    return ts_string


