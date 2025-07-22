#!/usr/bin/env python3
"""Full wakeword detection implementation for GAJA Assistant.

Follows AGENTS.md guidelines: async, testable, modular. Uses heavy ML dependencies for
advanced wake word detection.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Global flag to track if heavy dependencies are available
_HEAVY_DEPS_AVAILABLE = None


def _check_heavy_dependencies() -> bool:
    """Check if heavy ML dependencies are available."""
    global _HEAVY_DEPS_AVAILABLE

    if _HEAVY_DEPS_AVAILABLE is not None:
        return _HEAVY_DEPS_AVAILABLE

    try:
        # Check for sounddevice (required for audio input)
        # Check for numpy (should always be available)
        import numpy as np
        import sounddevice as sd

        # Try to import optional heavy dependencies
        try:
            import torch

            logger.info("PyTorch available for advanced wake word detection")
        except ImportError:
            logger.info("PyTorch not available - using basic detection")

        try:
            import openwakeword

            logger.info("OpenWakeWord available for advanced detection")
        except ImportError:
            logger.info("OpenWakeWord not available - using basic detection")

        _HEAVY_DEPS_AVAILABLE = True
        return True

    except ImportError as e:
        logger.warning(f"Heavy dependencies not available: {e}")
        _HEAVY_DEPS_AVAILABLE = False
        return False


def run_wakeword_detection(
    device_id: int | None = None,
    stt_silence_threshold_ms: int = 2000,
    keyword: str = "gaja",
    tts_module: Any | None = None,
    process_query_callback: Callable | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
    sensitivity: float = 0.6,
    whisper_asr: Any | None = None,
    manual_listen_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Run full wakeword detection with advanced ML models.

    This function runs the complete wake word detection pipeline:
    1. Audio capture from microphone
    2. Voice activity detection
    3. Wake word detection using ML models
    4. Speech-to-text conversion for queries
    5. Callback execution for detected queries

    Args:
        device_id: Audio input device ID (None for default)
        stt_silence_threshold_ms: Silence threshold for STT in milliseconds
        keyword: Wake word to detect (default: "gaja")
        tts_module: Text-to-speech module for feedback
        process_query_callback: Async callback for processing detected queries
        event_loop: Event loop for async operations
        sensitivity: Detection sensitivity (0.0-1.0)
        whisper_asr: Whisper ASR instance for speech recognition
        manual_listen_event: Event to manually trigger listening
        stop_event: Event to stop detection
    """
    logger.info(
        f"Starting full wakeword detection for '{keyword}' (sensitivity: {sensitivity})"
    )

    if not _check_heavy_dependencies():
        logger.warning(
            "Heavy dependencies not available - falling back to basic detection"
        )
        _run_basic_detection(
            device_id=device_id,
            keyword=keyword,
            sensitivity=sensitivity,
            process_query_callback=process_query_callback,
            event_loop=event_loop,
            manual_listen_event=manual_listen_event,
            stop_event=stop_event,
        )
        return

    try:
        _run_advanced_detection(
            device_id=device_id,
            stt_silence_threshold_ms=stt_silence_threshold_ms,
            keyword=keyword,
            tts_module=tts_module,
            process_query_callback=process_query_callback,
            event_loop=event_loop,
            sensitivity=sensitivity,
            whisper_asr=whisper_asr,
            manual_listen_event=manual_listen_event,
            stop_event=stop_event,
        )
    except Exception as e:
        logger.error(f"Error in advanced detection, falling back to basic: {e}")
        _run_basic_detection(
            device_id=device_id,
            keyword=keyword,
            sensitivity=sensitivity,
            process_query_callback=process_query_callback,
            event_loop=event_loop,
            manual_listen_event=manual_listen_event,
            stop_event=stop_event,
        )


def _run_advanced_detection(
    device_id: int | None = None,
    stt_silence_threshold_ms: int = 2000,
    keyword: str = "gaja",
    tts_module: Any | None = None,
    process_query_callback: Callable | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
    sensitivity: float = 0.6,
    whisper_asr: Any | None = None,
    manual_listen_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Run advanced ML-based wakeword detection."""
    import sounddevice as sd

    logger.info("Using advanced ML-based wake word detection")

    # Audio parameters
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.1  # 100ms chunks
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    BUFFER_DURATION = 3.0  # 3 seconds of audio buffer
    BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

    # Detection parameters
    VOICE_THRESHOLD = 0.01
    SILENCE_CHUNKS = 0
    MAX_SILENCE_CHUNKS = int(stt_silence_threshold_ms / (CHUNK_DURATION * 1000))

    # Cooldown to prevent continuous triggering
    COOLDOWN_PERIOD = 3.0  # 3 seconds between detections
    last_detection_time = 0

    # Audio buffer for processing
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    buffer_index = 0

    # Wake word model - real OpenWakeWord implementation
    wakeword_model = None
    try:
        logger.info("Loading OpenWakeWord model...")
        wakeword_model = _load_openwakeword_model(keyword)
        if wakeword_model is None:
            logger.warning("OpenWakeWord model failed to load, using basic detection")
            wakeword_model = "basic"
    except Exception as e:
        logger.warning(f"Could not load OpenWakeWord model: {e}")
        wakeword_model = "basic"

    def audio_callback(indata, frames, time_info, status):
        """Process incoming audio chunks."""
        nonlocal buffer_index, audio_buffer, last_detection_time

        if status:
            logger.warning(f"Audio callback status: {status}")

        if stop_event and stop_event.is_set():
            return

        try:
            # Convert to numpy array
            audio_data = np.frombuffer(indata, dtype=np.float32).flatten()

            # Add to circular buffer
            end_index = buffer_index + len(audio_data)
            if end_index <= BUFFER_SIZE:
                audio_buffer[buffer_index:end_index] = audio_data
            else:
                # Wrap around
                first_part = BUFFER_SIZE - buffer_index
                audio_buffer[buffer_index:] = audio_data[:first_part]
                audio_buffer[: end_index - BUFFER_SIZE] = audio_data[first_part:]

            buffer_index = end_index % BUFFER_SIZE

            # Voice activity detection
            amplitude = np.sqrt(np.mean(audio_data**2))
            import time as time_module

            current_time = time_module.time()

            # Check cooldown period
            if current_time - last_detection_time < COOLDOWN_PERIOD:
                return

            if amplitude > VOICE_THRESHOLD or (
                manual_listen_event and manual_listen_event.is_set()
            ):
                # Voice detected or manual trigger
                if manual_listen_event and manual_listen_event.is_set():
                    logger.info("Manual listen triggered")
                    manual_listen_event.clear()
                    last_detection_time = current_time

                # Check for wake word
                if _detect_wakeword(audio_buffer, keyword, sensitivity, wakeword_model):
                    logger.info(f"Wake word '{keyword}' detected!")
                    last_detection_time = current_time

                    # Start listening for query
                    if whisper_asr and process_query_callback:
                        _process_speech_query(
                            whisper_asr=whisper_asr,
                            callback=process_query_callback,
                            event_loop=event_loop,
                            device_id=device_id,
                            silence_threshold_ms=stt_silence_threshold_ms,
                            stop_event=stop_event,
                        )
                    else:
                        # Log warning if no Whisper ASR available
                        logger.warning(
                            "Wake word detected but no Whisper ASR available - cannot process query"
                        )
                        # Don't trigger callback with empty query

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    # Start audio stream
    logger.info(
        f"Starting audio stream (device: {device_id}, sample_rate: {SAMPLE_RATE})"
    )

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SIZE,
            device=device_id,
            callback=audio_callback,
        ):
            logger.info("Audio stream started for advanced wake word detection")

            # Keep running until stop event is set
            while not (stop_event and stop_event.is_set()):
                time.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in audio stream: {e}")
    finally:
        logger.info("Advanced wake word detection stopped")


def _run_basic_detection(
    device_id: int | None = None,
    keyword: str = "gaja",
    sensitivity: float = 0.6,
    process_query_callback: Callable | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
    manual_listen_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Run basic energy-based wakeword detection as fallback."""
    logger.info("Using basic energy-based wake word detection")

    try:
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice not available - cannot run wake word detection")
        return

    # Audio parameters
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 0.1  # 100ms
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

    # Simple detection parameters
    VOICE_THRESHOLD = 0.01
    audio_buffer = []
    BUFFER_LENGTH = 30  # Keep last 3 seconds
    silence_chunks = 0
    MAX_SILENCE_CHUNKS = 10

    def audio_callback(indata, frames, time, status):
        """Simple audio callback for basic detection."""
        nonlocal audio_buffer, silence_chunks

        if status:
            logger.warning(f"Audio callback status: {status}")

        if stop_event and stop_event.is_set():
            return

        try:
            audio_data = np.frombuffer(indata, dtype=np.float32).flatten()
            amplitude = np.sqrt(np.mean(audio_data**2))

            if amplitude > VOICE_THRESHOLD or (
                manual_listen_event and manual_listen_event.is_set()
            ):
                # Voice detected
                audio_buffer.append(audio_data)
                silence_chunks = 0

                if manual_listen_event and manual_listen_event.is_set():
                    logger.info("Manual listen triggered (basic)")
                    manual_listen_event.clear()
                    # Always trigger on manual
                    if process_query_callback:
                        _trigger_callback(process_query_callback, "", event_loop)
                    return

                # Keep buffer at reasonable length
                if len(audio_buffer) > BUFFER_LENGTH:
                    audio_buffer.pop(0)

                # Simple detection
                if len(audio_buffer) >= 10:
                    if _simple_keyword_detection(audio_buffer, sensitivity):
                        logger.info("Wake word detected (basic method)")
                        if process_query_callback:
                            _trigger_callback(process_query_callback, "", event_loop)
                        audio_buffer.clear()
            else:
                silence_chunks += 1
                if silence_chunks > MAX_SILENCE_CHUNKS:
                    audio_buffer.clear()
                    silence_chunks = 0

        except Exception as e:
            logger.error(f"Error in basic audio callback: {e}")

    # Start basic audio stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SIZE,
            device=device_id,
            callback=audio_callback,
        ):
            logger.info("Audio stream started for basic wake word detection")

            while not (stop_event and stop_event.is_set()):
                time.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in basic audio stream: {e}")
    finally:
        logger.info("Basic wake word detection stopped")


def _load_openwakeword_model(keyword: str) -> Any:
    """Load OpenWakeWord model for the specified keyword."""
    try:
        import os
        from pathlib import Path

        from openwakeword import Model

        # Get the base directory of the project
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up to f:\Asystent\

        # First try client/resources/openWakeWord
        model_dir = project_root / "client" / "resources" / "openWakeWord"
        if not model_dir.exists():
            # Fallback to resources/openWakeWord
            model_dir = project_root / "resources" / "openWakeWord"

        if not model_dir.exists():
            logger.error(f"OpenWakeWord model directory not found at {model_dir}")
            return None

        # Load ALL available models, not just keyword-specific ones
        model_files = []
        all_tflite_files = []

        for file in os.listdir(model_dir):
            if file.endswith(".tflite"):
                file_path = str(model_dir / file)
                all_tflite_files.append(file_path)

                # Still prioritize keyword-specific models
                if keyword.lower() in file.lower():
                    model_files.append(file_path)

        # If no keyword-specific models, use all available models
        if not model_files:
            model_files = all_tflite_files[
                :4
            ]  # Load up to 4 models for better detection
            logger.info(
                "No keyword-specific models found, loading first 4 available models"
            )
        else:
            # Add additional models if we have less than 4
            for file_path in all_tflite_files:
                if file_path not in model_files and len(model_files) < 4:
                    model_files.append(file_path)

        if not model_files:
            logger.error(f"No .tflite models found in {model_dir}")
            return None

        logger.info(
            f"Loading {len(model_files)} OpenWakeWord models: {[os.path.basename(f) for f in model_files]}"
        )

        # Initialize OpenWakeWord model with multiple models for better detection
        model = Model(wakeword_models=model_files, inference_framework="tflite")

        logger.info(
            f"OpenWakeWord model loaded successfully with {len(model_files)} models"
        )
        return model

    except ImportError:
        logger.error("OpenWakeWord library not available")
        return None
    except Exception as e:
        logger.error(f"Error loading OpenWakeWord model: {e}")
        return None


def _detect_wakeword(
    audio_buffer: np.ndarray, keyword: str, sensitivity: float, model: Any
) -> bool:
    """Detect wake word in audio buffer using ML model.

    Args:
        audio_buffer: Audio data buffer
        keyword: Target wake word
        sensitivity: Detection sensitivity
        model: Wake word detection model

    Returns:
        True if wake word detected, False otherwise
    """
    try:
        if model == "basic":
            # Use simple energy-based detection
            return _simple_energy_detection(audio_buffer, sensitivity)

        # Use actual OpenWakeWord model
        if model and hasattr(model, "predict"):
            try:
                # OpenWakeWord expects int16 PCM audio
                # Convert from float32 to int16
                audio_int16 = (audio_buffer * 32767).astype(np.int16)

                # Get prediction scores
                prediction_scores = model.predict(audio_int16)

                # Check if any keyword in the model was detected
                detected = False
                for keyword_name, score in prediction_scores.items():
                    if score > sensitivity:
                        logger.info(
                            f"OpenWakeWord detected '{keyword_name}' with score {score:.4f} (threshold: {sensitivity})"
                        )
                        detected = True
                        break

                return detected

            except Exception as e:
                logger.error(f"Error in OpenWakeWord prediction: {e}")
                # Fallback to basic detection
                return _simple_energy_detection(audio_buffer, sensitivity)

        # If no model available, fallback to basic detection
        return _simple_energy_detection(audio_buffer, sensitivity)

    except Exception as e:
        logger.error(f"Error in wake word detection: {e}")
        return False


def _simple_energy_detection(audio_buffer: np.ndarray, sensitivity: float) -> bool:
    """Simple energy-based wake word detection."""
    try:
        # Calculate total energy
        total_energy = np.sum(audio_buffer**2)

        # Check for voice patterns
        std_dev = np.std(audio_buffer)

        # Calculate RMS amplitude
        rms_amplitude = np.sqrt(np.mean(audio_buffer**2))

        # More strict thresholds to avoid false positives
        energy_threshold = sensitivity * 0.05  # Lower threshold
        variation_threshold = 0.01  # Higher variation threshold
        rms_threshold = 0.02  # Higher RMS threshold

        # Require ALL conditions to be met
        if (
            total_energy > energy_threshold
            and std_dev > variation_threshold
            and rms_amplitude > rms_threshold
        ):
            detection_score = min(total_energy * 5, 1.0)  # Reduced multiplier

            # Add additional check for energy consistency
            if detection_score > (sensitivity * 1.5):  # Require higher score
                logger.debug(
                    f"Wake word detection: energy={total_energy:.4f}, std={std_dev:.4f}, rms={rms_amplitude:.4f}, score={detection_score:.4f}"
                )
                return True

        return False

    except Exception as e:
        logger.error(f"Error in simple energy detection: {e}")
        return False


def _simple_keyword_detection(audio_buffer: list, sensitivity: float) -> bool:
    """Simple keyword detection for basic mode."""
    try:
        if not audio_buffer:
            return False

        # Combine audio buffer
        combined_audio = np.concatenate(audio_buffer)
        return _simple_energy_detection(combined_audio, sensitivity)

    except Exception as e:
        logger.error(f"Error in simple keyword detection: {e}")
        return False


def _process_speech_query(
    whisper_asr: Any,
    callback: Callable,
    event_loop: asyncio.AbstractEventLoop | None = None,
    device_id: int | None = None,
    silence_threshold_ms: int = 2000,
    stop_event: threading.Event | None = None,
) -> None:
    """Process speech query after wake word detection."""
    logger.info("Processing speech query...")

    try:
        # Record audio for transcription
        import sounddevice as sd

        # Record for a few seconds to capture the user's query
        SAMPLE_RATE = 16000
        RECORD_DURATION = 3.0  # Record for 3 seconds

        logger.info(f"Recording audio for {RECORD_DURATION} seconds...")

        # Record audio
        audio_data = sd.rec(
            int(SAMPLE_RATE * RECORD_DURATION),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=device_id,
        )
        sd.wait()  # Wait for recording to complete

        # Use Whisper ASR to transcribe if available
        if whisper_asr:
            try:
                # Convert to the format expected by Whisper
                audio_data_flat = audio_data.flatten()
                query = whisper_asr.transcribe(audio_data_flat, sample_rate=SAMPLE_RATE)
                logger.info(f"Transcribed query: {query}")
            except Exception as e:
                logger.error(f"Error transcribing with Whisper: {e}")
                query = ""
        else:
            logger.warning("No Whisper ASR available - cannot transcribe query")
            query = ""

        if query and query.strip():
            logger.info(f"Transcribed query: {query}")
            _trigger_callback(callback, query, event_loop)
        else:
            logger.info("No query detected")

    except Exception as e:
        logger.error(f"Error processing speech query: {e}")


def _trigger_callback(
    callback: Callable, query: str, event_loop: asyncio.AbstractEventLoop | None = None
) -> None:
    """Trigger the callback function with the detected query."""
    if not callback:
        return

    # Don't trigger callback if query is empty or only whitespace
    if not query or not query.strip():
        logger.warning(f"Not triggering callback - query is empty: '{query}'")
        return

    logger.info(f"Triggering callback with query: '{query}'")

    try:
        if asyncio.iscoroutinefunction(callback):
            # Async callback
            if event_loop and event_loop.is_running():
                # Schedule in the event loop
                future = asyncio.run_coroutine_threadsafe(callback(query), event_loop)
                try:
                    future.result(timeout=5.0)  # Wait up to 5 seconds
                except Exception as e:
                    logger.error(f"Error in async callback: {e}")
            else:
                # Run in new event loop
                asyncio.run(callback(query))
        else:
            # Sync callback
            callback(query)

    except Exception as e:
        logger.error(f"Error triggering callback: {e}")
