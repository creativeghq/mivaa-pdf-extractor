"""
Configuration module for MIVAA PDF Extractor.

This module re-exports configuration from the parent config.py file
to maintain backward compatibility with the directory structure.
"""

# Import from the parent config.py file
import sys
from pathlib import Path

# Add parent directory to path to import config.py
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import from config.py
from config import (
    Settings,
    get_settings,
    configure_logging,
    reload_settings,
    settings
)

# Also import the sub-modules
from . import ai_pricing
from . import confidence_thresholds

__all__ = [
    'Settings',
    'get_settings',
    'configure_logging',
    'reload_settings',
    'settings',
    'ai_pricing',
    'confidence_thresholds',
]

