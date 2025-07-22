#!/usr/bin/env python3
"""Optimized wake word detection and speech processing for GAJA Assistant.

This module provides high-performance, async-compatible wake word detection with
advanced VAD, efficient audio buffering, and optimized Whisper integration.

Follows AGENTS.md guidelines: async, testable, modular architecture.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Performance constants optimized for real-time processing
SAMPLE_RATE = 16000  # Hz - optimal for Whisper and OpenWakeWord
OPTIMAL_CHUNK_SIZE = 1024  # 64ms chunks for optimal latency/quality balance
BUFFER_DURATION = 3.0  # seconds of audio history
COOLDOWN_PERIOD = 2.0  # seconds between detections to prevent spam

# VAD (Voice Activity Detection) optimized thresholds
VAD_ENERGY_THRESHOLD = 0.002  # Tuned for better sensitivity
VAD_SPECTRAL_THRESHOLD = 0.015  # Spectral centroid threshold
VAD_SILENCE_FRAMES = 30  # Frames of silence to end recording (~2 seconds)
VAD_MIN_SPEECH_FRAMES = 10  # Minimum frames for valid speech (~0.64 seconds)

# Wake word detection parameters
WW_SENSITIVITY_DEFAULT = 0.65  # Balanced sensitivity
WW_FRAME_OVERLAP = 0.5  # 50% overlap for better detection
WW_ENERGY_GATE = 0.001  # Energy gate to prevent processing silence

# Audio recording parameters for commands
COMMAND_MAX_DURATION = 8.0  # seconds
COMMAND_MIN_DURATION = 0.5  # seconds


class OptimizedAudioBuffer:
    """Efficient circular audio buffer for real-time processing."""

    def __init__(self, max_duration: float, sample_rate: int):
        """Initialize the audio buffer.

        Args:
            max_duration: Maximum duration in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0
        self._lock = threading.Lock()

    def append(self, audio_data: np.ndarray) -> None:
        """Append audio data to the circular buffer.

        Args:
            audio_data: Audio samples to append
        """
        with self._lock:
            data_len = len(audio_data)

            # Handle wrap-around for circular buffer
            if self.write_pos + data_len <= self.max_samples:
                self.buffer[self.write_pos : self.write_pos + data_len] = audio_data
            else:
                # Split write across buffer boundary
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos :] = audio_data[:first_part]
                self.buffer[: data_len - first_part] = audio_data[first_part:]

            self.write_pos = (self.write_pos + data_len) % self.max_samples
            self.samples_written += data_len

    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get the most recent audio data.

        Args:
            duration: Duration in seconds to retrieve

        Returns:
            Recent audio samples
        """
        with self._lock:
            samples_needed = int(duration * SAMPLE_RATE)
            samples_available = min(
                samples_needed, self.samples_written, self.max_samples
            )

            if samples_available == 0:
                return np.array([], dtype=np.float32)

            # Calculate read position
            read_pos = (self.write_pos - samples_available) % self.max_samples

            if read_pos + samples_available <= self.max_samples:
                return self.buffer[read_pos : read_pos + samples_available].copy()
            else:
                # Handle wrap-around
                first_part = self.max_samples - read_pos
                result = np.zeros(samples_available, dtype=np.float32)
                result[:first_part] = self.buffer[read_pos:]
                result[first_part:] = self.buffer[: samples_available - first_part]
                return result


class AdvancedVAD:
    """Advanced Voice Activity Detection with spectral features."""

    def __init__(self):
        """Initialize VAD with optimized parameters."""
        self.energy_threshold = VAD_ENERGY_THRESHOLD
        self.spectral_threshold = VAD_SPECTRAL_THRESHOLD
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speech_active = False

        # Smoothing for more stable detection
        self.energy_history = deque(maxlen=5)
        self.spectral_history = deque(maxlen=5)

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Process audio frame for voice activity.

        Args:
            audio_frame: Audio samples for this frame

        Returns:
            True if voice activity detected
        """
        # Energy-based features
        energy = np.mean(audio_frame**2)
        self.energy_history.append(energy)

        # Spectral centroid for speech detection
        fft = np.fft.rfft(audio_frame)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio_frame), 1 / SAMPLE_RATE)

        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            spectral_centroid_norm = spectral_centroid / (SAMPLE_RATE / 2)
        else:
            spectral_centroid_norm = 0

        self.spectral_history.append(spectral_centroid_norm)

        # Smoothed features
        avg_energy = np.mean(self.energy_history)
        avg_spectral = np.mean(self.spectral_history)

        # Combined voice activity decision
        energy_speech = avg_energy > self.energy_threshold
        spectral_speech = avg_spectral > self.spectral_threshold

        voice_detected = energy_speech and spectral_speech

        # State tracking with hysteresis
        if voice_detected:
            self.speech_frames += 1
            self.silence_frames = 0
            if self.speech_frames >= 3:  # Require sustained speech
                self.is_speech_active = True
        else:
            self.silence_frames += 1
            self.speech_frames = max(0, self.speech_frames - 1)
            if self.silence_frames >= VAD_SILENCE_FRAMES:
                self.is_speech_active = False

        return self.is_speech_active

    def reset(self) -> None:
        """Reset VAD state."""
        self.silence_frames = 0
        self.speech_frames = 0
        self.is_speech_active = False
        self.energy_history.clear()
        self.spectral_history.clear()


class OptimizedWakeWordDetector:
    """High-performance wake word detector with advanced features."""

    def __init__(
        self,
        sensitivity: float = WW_SENSITIVITY_DEFAULT,
        keyword: str = "gaja",
        device_id: int | None = None,
    ):
        """Initialize the wake word detector.

        Args:
            sensitivity: Detection sensitivity (0.0-1.0)
            keyword: Target wake word
            device_id: Audio input device ID
        """
        self.sensitivity = sensitivity
        self.keyword = keyword.lower()
        self.device_id = device_id

        # Audio processing components
        self.audio_buffer = OptimizedAudioBuffer(BUFFER_DURATION, SAMPLE_RATE)
        self.vad = AdvancedVAD()

        # Wake word model (loaded on demand)
        self.ww_model = None
        self.model_loaded = False

        # Whisper ASR integration
        self.whisper_asr = None

        # State management
        self.is_running = False
        self.last_detection_time = 0
        self.detection_callbacks = []

        # Performance tracking
        self.frames_processed = 0
        self.detections_count = 0

        # Threading
        self._stop_event = threading.Event()
        self._audio_thread = None

    def add_detection_callback(self, callback: Callable[[str], None]) -> None:
        """Add a callback for wake word detections.

        Args:
            callback: Function to call when wake word is detected
        """
        self.detection_callbacks.append(callback)

    async def start_async(self) -> None:
        """Start wake word detection asynchronously."""
        if self.is_running:
            return

        logger.info("Starting optimized wake word detection")

        # Load wake word model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_wake_word_model)

        # Start audio processing thread
        self.is_running = True
        self._stop_event.clear()
        self._audio_thread = threading.Thread(target=self._audio_processing_loop)
        self._audio_thread.start()

        logger.info("Wake word detection started successfully")

    async def stop_async(self) -> None:
        """Stop wake word detection asynchronously."""
        if not self.is_running:
            return

        logger.info("Stopping wake word detection")

        self.is_running = False
        self._stop_event.set()

        if self._audio_thread and self._audio_thread.is_alive():
            # Use executor to avoid blocking the async loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._audio_thread.join, 3.0)

        logger.info("Wake word detection stopped")

    def _load_wake_word_model(self) -> None:
        """Load the wake word model (runs in thread pool)."""
        try:
            # Try to load OpenWakeWord model
            import os
            from pathlib import Path

            # Locate model directory
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            model_dir = project_root / "resources" / "openWakeWord"

            if not model_dir.exists():
                model_dir = project_root / "client" / "resources" / "openWakeWord"

            if model_dir.exists():
                try:
                    from openwakeword import Model

                    # Find available models - use only .onnx files for onnx framework
                    model_files = []
                    for file in os.listdir(model_dir):
                        if file.endswith(".onnx"):  # Only use .onnx files
                            if not any(
                                x in file.lower()
                                for x in ["preprocessor", "embedding", "melspectrogram"]
                            ):
                                model_files.append(str(model_dir / file))

                    if model_files:
                        logger.info(
                            f"Loading {len(model_files)} wake word models (ONNX)"
                        )
                        self.ww_model = Model(
                            wakeword_models=model_files[
                                :4
                            ],  # Limit to 4 models for performance
                            inference_framework="onnx",
                        )
                        self.model_loaded = True
                        logger.info("OpenWakeWord model loaded successfully")
                    else:
                        logger.warning("No suitable wake word models found")

                except ImportError:
                    logger.warning(
                        "OpenWakeWord not available, using energy-based detection"
                    )
            else:
                logger.warning("Wake word model directory not found")

        except Exception as e:
            logger.error(f"Error loading wake word model: {e}")

    def _audio_processing_loop(self) -> None:
        """Main audio processing loop using legacy wake word detection."""
        try:
            # Use the working legacy wake word detection code
            from .sounddevice_loader import is_sounddevice_available
            from .wakeword_detector import run_wakeword_detection

            # Check if sounddevice is available
            if not is_sounddevice_available():
                logger.error("sounddevice not available for wake word detection")
                return

            logger.info("Starting legacy wake word detection from optimized detector")

            # Use our working legacy implementation
            run_wakeword_detection(
                mic_device_id=self.device_id,
                stt_silence_threshold_ms=2000,
                wake_word_config_name="gaja",
                tts_module=None,
                process_query_callback_async=self._handle_wake_word_detection_async,
                async_event_loop=self._get_event_loop(),
                oww_sensitivity_threshold=self.sensitivity,
                whisper_asr_instance=getattr(self, "whisper_asr", None),
                manual_listen_trigger_event=threading.Event(),
                stop_detector_event=self._stop_event,
            )

        except Exception as e:
            logger.error(f"Error in audio processing loop: {e}")
        finally:
            logger.info("Audio processing loop finished")

    def _handle_wake_word_detection(self, transcribed_text: str) -> None:
        """Handle wake word detection and transcribed text."""
        try:
            logger.info(f"Wake word detected with text: '{transcribed_text}'")

            # Trigger callbacks
            for callback in self.detection_callbacks:
                try:
                    # Check if callback is async
                    import asyncio
                    import inspect

                    if inspect.iscoroutinefunction(callback):
                        # If callback is async, schedule it on the event loop
                        try:
                            loop = asyncio.get_event_loop()
                            asyncio.create_task(callback(transcribed_text))
                        except RuntimeError:
                            # No event loop in this thread, create new one
                            asyncio.run(callback(transcribed_text))
                    else:
                        # Synchronous callback
                        callback(transcribed_text)
                except Exception as e:
                    logger.error(f"Error in detection callback: {e}")

        except Exception as e:
            logger.error(f"Error handling wake word detection: {e}")

    async def _handle_wake_word_detection_async(self, transcribed_text: str) -> None:
        """Async wrapper for wake word detection handling."""
        try:
            logger.info(f"Wake word detected with text: '{transcribed_text}'")

            # Trigger callbacks
            for callback in self.detection_callbacks:
                try:
                    import inspect

                    if inspect.iscoroutinefunction(callback):
                        # If callback is async, await it
                        await callback(transcribed_text)
                    else:
                        # Synchronous callback
                        callback(transcribed_text)
                except Exception as e:
                    logger.error(f"Error in detection callback: {e}")

        except Exception as e:
            logger.error(f"Error handling wake word detection: {e}")

    def _get_event_loop(self):
        """Get the current event loop or None if not available."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    def _process_audio_frame(self, audio_frame: np.ndarray) -> None:
        """Process a single audio frame for wake word detection.

        Args:
            audio_frame: Audio samples for this frame
        """
        # VAD processing
        voice_detected = self.vad.process_frame(audio_frame)

        if not voice_detected:
            return

        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_detection_time < COOLDOWN_PERIOD:
            return

        # Get recent audio for wake word detection
        recent_audio = self.audio_buffer.get_recent_audio(1.5)  # 1.5 seconds

        if len(recent_audio) < SAMPLE_RATE * 0.5:  # Need at least 0.5 seconds
            return

        # Perform wake word detection
        if self._detect_wake_word(recent_audio):
            self.last_detection_time = current_time
            self.detections_count += 1
            self._trigger_detection_callbacks()

    def _detect_wake_word(self, audio_data: np.ndarray) -> bool:
        """Detect wake word in audio data.

        Args:
            audio_data: Audio samples to analyze

        Returns:
            True if wake word detected
        """
        try:
            if self.model_loaded and self.ww_model:
                # Use OpenWakeWord model
                audio_int16 = (audio_data * 32767).astype(np.int16)

                # Process with overlap for better detection
                chunk_size = int(SAMPLE_RATE * 0.8)  # 0.8 second chunks
                overlap = int(chunk_size * WW_FRAME_OVERLAP)

                for start in range(
                    0, len(audio_int16) - chunk_size, chunk_size - overlap
                ):
                    chunk = audio_int16[start : start + chunk_size]

                    if len(chunk) < chunk_size:
                        continue

                    predictions = self.ww_model.predict(chunk)

                    for keyword, score in predictions.items():
                        if score >= self.sensitivity:
                            logger.info(
                                f"Wake word '{keyword}' detected (score: {score:.3f})"
                            )
                            return True
            else:
                # Fallback to energy-based detection
                return self._energy_based_detection(audio_data)

        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return self._energy_based_detection(audio_data)

        return False

    def _energy_based_detection(self, audio_data: np.ndarray) -> bool:
        """Energy-based wake word detection fallback.

        Args:
            audio_data: Audio samples to analyze

        Returns:
            True if potential wake word detected
        """
        try:
            # Calculate energy and spectral features
            energy = np.mean(audio_data**2)

            # Spectral analysis for speech characteristics
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1 / SAMPLE_RATE)

            # Focus on speech frequency range (80Hz - 8kHz)
            speech_mask = (freqs >= 80) & (freqs <= 8000)
            speech_energy = np.sum(magnitude[speech_mask] ** 2)
            total_energy = np.sum(magnitude**2)

            if total_energy > 0:
                speech_ratio = speech_energy / total_energy
            else:
                speech_ratio = 0

            # Dynamic thresholds based on sensitivity
            energy_threshold = self.sensitivity * 0.01
            speech_threshold = self.sensitivity * 0.3

            if energy > energy_threshold and speech_ratio > speech_threshold:
                # Additional pattern matching for "gaja"
                score = min(energy * 100 * speech_ratio, 1.0)
                if score > self.sensitivity:
                    logger.info(f"Energy-based wake word detected (score: {score:.3f})")
                    return True

        except Exception as e:
            logger.error(f"Error in energy-based detection: {e}")

        return False

    def _trigger_detection_callbacks(self) -> None:
        """Trigger all registered detection callbacks."""
        for callback in self.detection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Schedule async callback
                    asyncio.create_task(callback(self.keyword))
                else:
                    callback(self.keyword)
            except Exception as e:
                logger.error(f"Error in detection callback: {e}")

    def set_whisper_asr(self, whisper_asr) -> None:
        """Set Whisper ASR instance for integration (legacy compatibility).

        Args:
            whisper_asr: Whisper ASR instance
        """
        # Store ASR reference for potential future use
        self.whisper_asr = whisper_asr
        logger.debug("Whisper ASR reference stored in wake word detector")

    def start_detection(self) -> None:
        """Legacy-compatible synchronous start method.

        Starts wake word detection in a background thread.
        """
        import asyncio
        import threading

        def run_async_detection():
            """Run async detection in new event loop."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.start_async())
            except Exception as e:
                logger.error(f"Error in async detection thread: {e}")

        # Start detection in background thread
        self._detection_thread = threading.Thread(
            target=run_async_detection, daemon=True
        )
        self._detection_thread.start()
        logger.info("Wake word detection started in background thread")

    def stop_detection(self) -> None:
        """Legacy-compatible synchronous stop method."""
        import asyncio

        try:
            # Try to get running event loop
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.stop_async())
        except RuntimeError:
            # No running loop, create new one
            asyncio.run(self.stop_async())

        logger.info("Wake word detection stopped")


class OptimizedSpeechRecorder:
    """Optimized speech recorder with advanced VAD for command capture."""

    def __init__(self, device_id: int | None = None):
        """Initialize the speech recorder.

        Args:
            device_id: Audio input device ID
        """
        self.device_id = device_id
        self.vad = AdvancedVAD()

        # Recording state
        self.is_recording = False
        self.recorded_audio = []
        self._stop_event = threading.Event()

    async def record_command_async(
        self, max_duration: float = COMMAND_MAX_DURATION, silence_timeout: float = 2.0
    ) -> np.ndarray | None:
        """Record a voice command asynchronously.

        Args:
            max_duration: Maximum recording duration in seconds
            silence_timeout: Stop recording after this much silence

        Returns:
            Recorded audio samples or None if failed
        """
        logger.info(f"Starting command recording (max: {max_duration}s)")

        # Run recording in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        try:
            audio_data = await loop.run_in_executor(
                None, self._record_blocking, max_duration, silence_timeout
            )

            if audio_data is not None and len(audio_data) > 0:
                duration = len(audio_data) / SAMPLE_RATE
                logger.info(f"Command recorded: {duration:.2f}s")
                return audio_data
            else:
                logger.warning("No valid command audio recorded")
                return None

        except Exception as e:
            logger.error(f"Error recording command: {e}")
            return None

    def _record_blocking(
        self, max_duration: float, silence_timeout: float
    ) -> np.ndarray | None:
        """Blocking recording implementation (runs in thread pool).

        Args:
            max_duration: Maximum recording duration
            silence_timeout: Silence timeout

        Returns:
            Recorded audio or None
        """
        try:
            import sounddevice as sd

            self.recorded_audio = []
            self.vad.reset()
            self._stop_event.clear()

            frames_recorded = 0
            silence_frames = 0
            speech_started = False

            max_frames = int(max_duration * SAMPLE_RATE / OPTIMAL_CHUNK_SIZE)
            silence_frame_limit = int(
                silence_timeout * SAMPLE_RATE / OPTIMAL_CHUNK_SIZE
            )

            def recording_callback(indata, frames, time_info, status):
                nonlocal frames_recorded, silence_frames, speech_started

                if status:
                    logger.debug(f"Recording status: {status}")

                if self._stop_event.is_set():
                    raise sd.CallbackStop()

                try:
                    audio_data = indata.astype(np.float32).flatten()

                    # VAD processing
                    voice_active = self.vad.process_frame(audio_data)

                    if voice_active:
                        speech_started = True
                        silence_frames = 0
                        self.recorded_audio.append(audio_data)
                    else:
                        if speech_started:
                            silence_frames += 1
                            self.recorded_audio.append(
                                audio_data
                            )  # Keep recording during pauses

                    frames_recorded += 1

                    # Stop conditions
                    if frames_recorded >= max_frames:
                        logger.info("Recording stopped: maximum duration reached")
                        raise sd.CallbackStop()

                    if speech_started and silence_frames >= silence_frame_limit:
                        logger.info("Recording stopped: silence detected")
                        raise sd.CallbackStop()

                except sd.CallbackStop:
                    raise
                except Exception as e:
                    logger.error(f"Error in recording callback: {e}")

            # Start recording stream
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=OPTIMAL_CHUNK_SIZE,
                device=self.device_id,
                callback=recording_callback,
                latency="low",
            ):
                # Wait for recording to complete
                while not self._stop_event.wait(0.1):
                    if frames_recorded >= max_frames:
                        break
                    if speech_started and silence_frames >= silence_frame_limit:
                        break

            # Process recorded audio
            if self.recorded_audio:
                audio_data = np.concatenate(self.recorded_audio)

                # Validate recording
                if len(audio_data) / SAMPLE_RATE >= COMMAND_MIN_DURATION:
                    return audio_data
                else:
                    logger.info("Recording too short, discarding")
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Error in blocking recording: {e}")
            return None


async def create_optimized_detector(
    sensitivity: float = WW_SENSITIVITY_DEFAULT,
    keyword: str = "gaja",
    device_id: int | None = None,
) -> OptimizedWakeWordDetector:
    """Create and initialize an optimized wake word detector.

    Args:
        sensitivity: Detection sensitivity (0.0-1.0)
        keyword: Target wake word
        device_id: Audio input device ID

    Returns:
        Configured wake word detector
    """
    detector = OptimizedWakeWordDetector(
        sensitivity=sensitivity, keyword=keyword, device_id=device_id
    )

    await detector.start_async()
    return detector


async def create_optimized_recorder(
    device_id: int | None = None,
) -> OptimizedSpeechRecorder:
    """Create an optimized speech recorder.

    Args:
        device_id: Audio input device ID

    Returns:
        Configured speech recorder
    """
    return OptimizedSpeechRecorder(device_id=device_id)


# Compatibility functions for existing code
def run_optimized_wakeword_detection(
    mic_device_id: int | None,
    stt_silence_threshold_ms: int,
    wake_word_config_name: str,
    tts_module: Any,
    process_query_callback_async: Callable,
    async_event_loop: asyncio.AbstractEventLoop,
    oww_sensitivity_threshold: float,
    whisper_asr_instance: Any,
    manual_listen_trigger_event: threading.Event,
    stop_detector_event: threading.Event,
) -> None:
    """Compatibility wrapper for the optimized wake word detection.

    This function provides a bridge between the legacy interface and the new optimized
    implementation, maintaining backward compatibility while offering improved
    performance.
    """

    async def detection_wrapper():
        """Async wrapper for wake word detection."""
        try:
            # Create optimized detector
            detector = await create_optimized_detector(
                sensitivity=oww_sensitivity_threshold,
                keyword=wake_word_config_name.lower(),
                device_id=mic_device_id,
            )

            # Create speech recorder
            recorder = await create_optimized_recorder(device_id=mic_device_id)

            async def handle_wake_word_detection(keyword: str):
                """Handle wake word detection with speech recording."""
                logger.info(
                    f"Wake word '{keyword}' detected, starting speech recording"
                )

                try:
                    # Cancel TTS if active
                    if tts_module and hasattr(tts_module, "cancel"):
                        tts_module.cancel()

                    # Record command
                    silence_timeout = stt_silence_threshold_ms / 1000.0
                    audio_data = await recorder.record_command_async(
                        silence_timeout=silence_timeout
                    )

                    if audio_data is not None and whisper_asr_instance:
                        # Transcribe with Whisper
                        logger.info("Transcribing command with optimized Whisper...")
                        transcription = whisper_asr_instance.transcribe(
                            audio_data, sample_rate=SAMPLE_RATE
                        )

                        if transcription and transcription.strip():
                            logger.info(f"Transcription: '{transcription}'")
                            # Call the original callback
                            await process_query_callback_async(transcription)
                        else:
                            logger.warning("Empty transcription received")
                    else:
                        logger.warning("No audio recorded or Whisper not available")

                except Exception as e:
                    logger.error(f"Error handling wake word detection: {e}")

            # Add wake word detection callback
            detector.add_detection_callback(handle_wake_word_detection)

            # Handle manual listen trigger
            async def check_manual_trigger():
                """Check for manual listen trigger events."""
                while not stop_detector_event.is_set():
                    if manual_listen_trigger_event.is_set():
                        manual_listen_trigger_event.clear()
                        await handle_wake_word_detection("manual")
                    await asyncio.sleep(0.1)

            # Start manual trigger monitoring
            manual_task = asyncio.create_task(check_manual_trigger())

            # Keep running until stop event
            while not stop_detector_event.is_set():
                await asyncio.sleep(0.1)

            # Cleanup
            manual_task.cancel()
            await detector.stop_async()

        except Exception as e:
            logger.error(f"Error in optimized wake word detection: {e}")

    # Run the async detection in the provided event loop
    if async_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(detection_wrapper(), async_event_loop)
    else:
        async_event_loop.run_until_complete(detection_wrapper())


def create_wakeword_detector(
    config: dict, callback: callable
) -> OptimizedWakeWordDetector:
    """Legacy-compatible synchronous factory function for backward compatibility.

    Args:
        config: Wake word configuration dictionary
        callback: Callback function when wake word is detected

    Returns:
        Configured OptimizedWakeWordDetector instance
    """
    # Extract parameters from config
    sensitivity = config.get("sensitivity", WW_SENSITIVITY_DEFAULT)
    keyword = config.get("keyword", "gaja")
    device_id = config.get("device_id", None)

    # Create detector synchronously
    detector = OptimizedWakeWordDetector(
        sensitivity=sensitivity, keyword=keyword, device_id=device_id
    )

    # Add callback
    if callback:
        detector.add_detection_callback(callback)

    return detector
