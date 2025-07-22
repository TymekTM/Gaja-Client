"""Minimal overlay stub for Tauri-based interface.

No tkinter dependencies - only console logging.
"""

import logging

logger = logging.getLogger(__name__)


class TauriOverlay:
    """Minimal overlay interface for Tauri-based UI."""

    def __init__(self):
        self.is_active = False
        logger.info("TauriOverlay initialized (no tkinter)")

    def show_response(self, text):
        """Show AI response - handled by Tauri frontend"""
        logger.info(f"AI Response: {text}")
        print(f"AI Response: {text}")

    def show_listening(self):
        """Show listening indicator - handled by Tauri frontend"""
        logger.info("Listening for user input...")
        print("🎤 Listening...")

    def hide_overlay(self):
        """Hide overlay - handled by Tauri frontend"""
        logger.info("Overlay hidden")
        print("Overlay hidden")

    def cleanup(self):
        """Cleanup overlay resources."""
        self.is_active = False
        logger.info("TauriOverlay cleaned up")


# For backward compatibility
class Overlay(TauriOverlay):
    """Compatibility alias."""

    pass
