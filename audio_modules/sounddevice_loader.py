"""Centralized sounddevice loader with automatic dependency management.

This module handles sounddevice import with fallback to dependency manager.
"""

import logging

# Global variables to store sounddevice module and availability status
_sounddevice_module = None
_sounddevice_available = None


def ensure_sounddevice():
    """Ensure sounddevice is available, downloading if necessary."""
    global _sounddevice_module, _sounddevice_available

    if _sounddevice_available is not None:
        return _sounddevice_available

    try:
        import sounddevice as sd

        _sounddevice_module = sd
        _sounddevice_available = True
        logging.getLogger(__name__).info("sounddevice loaded successfully")
        return True
    except (ImportError, OSError) as e:
        logging.getLogger(__name__).warning(f"sounddevice not available: {e}")
        _sounddevice_available = False
        return False


def get_sounddevice():
    """Get the sounddevice module, ensuring it's available first."""
    ensure_sounddevice()
    return _sounddevice_module


def is_sounddevice_available():
    """Check if sounddevice is available."""
    ensure_sounddevice()
    return _sounddevice_available
