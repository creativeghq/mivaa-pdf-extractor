"""
Configuration module for MIVAA PDF Extractor.

This module re-exports configuration from the parent config.py file
to maintain backward compatibility with the directory structure.
"""

# Import from the parent config.py file using absolute import
import sys
import importlib.util
from pathlib import Path

# Load config.py directly to avoid circular imports
config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("_config", config_path)
_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_config)

# Re-export the functions and classes
Settings = _config.Settings
get_settings = _config.get_settings
configure_logging = _config.configure_logging
reload_settings = _config.reload_settings
settings = _config.settings

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

