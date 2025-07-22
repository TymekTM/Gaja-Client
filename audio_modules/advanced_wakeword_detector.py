"""Adapter for the full wakeword detector to work with client-server architecture."""

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class WakewordDetector:
    """Adapter for the full wakeword detector implementation."""

    def __init__(self, config: dict[str, Any], callback: Callable):
        """Initialize the wakeword detector adapter.

        Args:
            config: Wakeword configuration with keys:
                - enabled: bool
                - sensitivity: float (0.0-1.0)
                - keyword: str
                - device_id: int or None
                - stt_silence_threshold_ms: int
            callback: Callback function to call when wakeword is detected
        """
        self.config = config
        self.callback = callback
        self.is_running = False
        self.detection_thread = None
        self.keyword = config.get("keyword", "gaja").lower()
        self.sensitivity = config.get("sensitivity", 0.6)
        self.device_id = config.get("device_id")
        self.stt_silence_threshold_ms = config.get("stt_silence_threshold_ms", 2000)

        # Threading events for control
        self.stop_event = threading.Event()
        self.manual_listen_event = threading.Event()

        # Audio components placeholders
        self.whisper_asr = None
        self.tts_module = None

        logger.info(
            f"WakewordDetector adapter initialized for keyword '{self.keyword}' with sensitivity {self.sensitivity}"
        )

    async def start(self):
        """Start the wakeword detection."""
        if not self.config.get("enabled", True):
            logger.info("Wakeword detection is disabled in config")
            return

        if self.is_running:
            logger.warning("Wakeword detector is already running")
            return

        self.is_running = True
        self.stop_event.clear()
        # Import and start the full wakeword detection in a separate thread
        self.detection_thread = threading.Thread(
            target=self._run_detection_wrapper, daemon=True
        )
        self.detection_thread.start()

        logger.info("Wakeword detection started")

    async def start_monitoring(self):
        """Start monitoring for wakewords (alias for start)."""
        await self.start()

    async def stop(self):
        """Stop the wakeword detection."""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=3.0)

        logger.info("Wakeword detection stopped")

    def _run_detection_wrapper(self):
        """Wrapper to run the full wakeword detection."""
        try:
            # Import the full implementation
            from .wakeword_detector_full import run_wakeword_detection

            # Create event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Setup callback for query processing
            async def process_query_callback(query: str):
                """Process detected query."""
                logger.info(f"🎯 Processing query in callback: {query}")
                if self.callback:
                    if asyncio.iscoroutinefunction(self.callback):
                        logger.info("🚀 Calling async callback...")
                        await self.callback(query)
                        logger.info("✅ Async callback completed")
                    else:
                        logger.info("🚀 Calling sync callback...")
                        self.callback(query)
                        logger.info("✅ Sync callback completed")
                else:
                    logger.warning("❌ No callback set!")

            async def main_detection():
                """Main detection coroutine."""
                logger.info("🔄 Starting wakeword detection in event loop...")
                # Use run_in_executor to run the blocking wakeword detection in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    await loop.run_in_executor(
                        executor,
                        run_wakeword_detection,
                        self.device_id,
                        self.stt_silence_threshold_ms,
                        self.keyword,
                        self.tts_module,
                        process_query_callback,
                        loop,
                        self.sensitivity,
                        self.whisper_asr,
                        self.manual_listen_event,
                        self.stop_event,
                    )

            # Run the event loop
            loop.run_until_complete(main_detection())
        except Exception as e:
            logger.error(f"Error in wakeword detection: {e}", exc_info=True)
        finally:
            if "loop" in locals():
                logger.info("🔄 Closing event loop...")
                loop.close()
            logger.info("Wakeword detection thread finished")

    def trigger_manual_detection(self):
        """Manually trigger wakeword detection for testing."""
        logger.info("Manual wakeword detection triggered")
        self.manual_listen_event.set()

    def set_whisper_asr(self, whisper_asr):
        """Set the Whisper ASR instance."""
        self.whisper_asr = whisper_asr
        logger.info("Whisper ASR instance set for wakeword detector")

    def set_tts_module(self, tts_module):
        """Set the TTS module."""
        self.tts_module = tts_module
        logger.info("TTS module set for wakeword detector")

    def start_detection(self):
        """Start wakeword detection."""
        if not self.is_running:
            logger.info("Starting wakeword detection...")
            self.is_running = True
            self.detection_thread = threading.Thread(
                target=self._run_detection_wrapper, daemon=True
            )
            self.detection_thread.start()
        else:
            logger.warning("Wakeword detection already running")

    def stop_detection(self):
        """Stop wakeword detection."""
        if self.is_running:
            logger.info("Stopping wakeword detection...")
            self.is_running = False
            if hasattr(self, "detection_thread") and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)
        else:
            logger.debug("Wakeword detection already stopped")


def create_wakeword_detector(
    config: dict[str, Any], callback: Callable
) -> WakewordDetector:
    """Create and return a wakeword detector instance.

    Args:
        config: Wakeword configuration
        callback: Callback function to call when wakeword is detected

    Returns:
        WakewordDetector instance
    """
    return WakewordDetector(config, callback)
