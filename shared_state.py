# Stub shared_state module for client
import logging

logger = logging.getLogger(__name__)


def save_assistant_state(state):
    """Stub function for saving assistant state."""
    logger.debug(f"save_assistant_state called with: {state}")


def update_wake_word_state(detected=False):
    """Stub function for updating wake word state."""
    logger.debug(f"update_wake_word_state called with detected={detected}")


def update_listening_state(listening=False):
    """Stub function for updating listening state."""
    logger.debug(f"update_listening_state called with listening={listening}")


# Stub utils module
class AudioUtils:
    @staticmethod
    def get_assistant_instance():
        """Stub function for getting assistant instance."""
        return None


# Create utils module structure
import types

utils = types.ModuleType("utils")
utils.audio_utils = AudioUtils()

import sys

sys.modules["utils"] = utils
sys.modules["utils.audio_utils"] = utils.audio_utils
