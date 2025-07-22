"""Simple wakeword detector for testing."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WakewordDetector:
    """Simple wakeword detector for testing."""

    def __init__(self, config: dict[str, Any], callback: Callable):
        self.config = config
        self.callback = callback
        self.is_running = False
        self.detection_thread = None
        self.keyword = config.get("keyword", "gaja").lower()
        self.sensitivity = config.get("sensitivity", 0.6)
        self.device_id = config.get("device_id")

        # Audio recording parameters
        self.sample_rate = 16000
        self.chunk_duration = 0.1  # 100ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Basic detection parameters
        self.voice_threshold = 0.01
        self.silence_chunks = 0
        self.max_silence_chunks = 10
        self.audio_buffer = []
        self.buffer_length = 30

        # Try to import sounddevice
        try:
            import sounddevice as sd

            self.sd = sd
            self.sounddevice_available = True
            logger.info("sounddevice loaded successfully")
        except ImportError:
            self.sd = None
            self.sounddevice_available = False
            logger.warning("sounddevice not available - wakeword detection disabled")

        logger.info(
            f"WakewordDetector initialized for keyword '{self.keyword}' with sensitivity {self.sensitivity}"
        )

    async def start(self):
        """Start the wakeword detection."""
        if not self.config.get("enabled", True):
            logger.info("Wakeword detection is disabled in config")
            return

        if not self.sounddevice_available:
            logger.warning(
                "Cannot start wakeword detection - sounddevice not available"
            )
            return

        if self.is_running:
            logger.warning("Wakeword detector is already running")
            return

        self.is_running = True

        # Start detection in a separate thread
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
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

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

        logger.info("Wakeword detection stopped")

    def _detection_loop(self):
        """Main detection loop with actual audio processing."""
        try:
            logger.info("Wakeword detection loop started")

            # Set up audio stream
            with self.sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                device=self.device_id,
                callback=self._audio_callback,
            ):
                logger.info("Audio stream started for wakeword detection")

                while self.is_running:
                    # TODO: Replace with proper async/await pattern
                    # Small delay to prevent busy waiting - acceptable in audio thread
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in wakeword detection loop: {e}")
        finally:
            logger.info("Wakeword detection loop finished")

    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio stream - processes incoming audio chunks."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        try:
            # Convert to numpy array
            audio_data = np.frombuffer(indata, dtype=np.float32).flatten()

            # Simple voice activity detection
            amplitude = np.sqrt(np.mean(audio_data**2))

            if amplitude > self.voice_threshold:
                # Voice detected - add to buffer
                self.audio_buffer.append(audio_data)
                self.silence_chunks = 0

                # Keep buffer at reasonable length
                if len(self.audio_buffer) > self.buffer_length:
                    self.audio_buffer.pop(0)

                # Simple keyword detection
                if len(self.audio_buffer) >= 10:  # Need at least 1 second of audio
                    self._check_for_keyword()
            else:
                # Silence detected
                self.silence_chunks += 1
                if self.silence_chunks > self.max_silence_chunks:
                    # Reset buffer after prolonged silence
                    self.audio_buffer.clear()
                    self.silence_chunks = 0

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def _check_for_keyword(self):
        """Simple keyword detection based on audio patterns."""
        try:
            # Combine audio buffer
            combined_audio = np.concatenate(self.audio_buffer)

            # Calculate energy
            total_energy = np.sum(combined_audio**2)

            # Simple energy-based detection
            if total_energy > (self.sensitivity * 0.1):
                # Check for voice patterns
                std_dev = np.std(combined_audio)
                if std_dev > 0.005:  # Some variation indicates speech
                    detection_score = min(total_energy * 10, 1.0)

                    if detection_score > self.sensitivity:
                        logger.info(f"Wakeword detected (score: {detection_score:.2f})")
                        self._trigger_wakeword_callback()

                        # Clear buffer after detection to avoid repeated triggers
                        self.audio_buffer.clear()

        except Exception as e:
            logger.error(f"Error in keyword checking: {e}")

    def _trigger_wakeword_callback(self):
        """Trigger the wakeword callback."""
        logger.info("Wakeword detected! Triggering callback.")
        if self.callback:
            try:
                # Call the callback asynchronously if it's a coroutine
                if asyncio.iscoroutinefunction(self.callback):
                    # Schedule the coroutine to run in the event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.callback())
                        else:
                            loop.run_until_complete(self.callback())
                    except RuntimeError:
                        # If no event loop, run in thread
                        asyncio.run(self.callback())
                else:
                    self.callback()
            except Exception as e:
                logger.error(f"Error calling wakeword callback: {e}")

    def trigger_manual_detection(self):
        """Manually trigger wakeword detection for testing."""
        logger.info("Manual wakeword detection triggered")
        self._trigger_wakeword_callback()


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
