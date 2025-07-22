import asyncio
import logging
import os
import queue
import sys
import threading
import time

import numpy as np

from .sounddevice_loader import get_sounddevice, is_sounddevice_available

# Import shared state management
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_state import update_listening_state, update_wake_word_state

# Import BASE_DIR from client config
try:
    from client.config import BASE_DIR
except ImportError:
    # Fallback if client.config not available
    if getattr(sys, "frozen", False):
        BASE_DIR = os.path.dirname(sys.executable)
    else:
        BASE_DIR = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

# Load sounddevice using our centralized loader
sd = get_sounddevice()
SOUNDDEVICE_AVAILABLE = is_sounddevice_available()
if SOUNDDEVICE_AVAILABLE:
    logging.getLogger(__name__).info("sounddevice loaded successfully via loader")
else:
    logging.getLogger(__name__).warning(
        "sounddevice not available - will be installed on demand"
    )

Model = None
OPENWAKEWORD_AVAILABLE = False
# openwakeword will be imported dynamically inside run_wakeword_detection after ensuring dependencies

from .beep_sounds import play_beep

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Helper function to determine the base path for resources (use config.BASE_DIR when frozen)
def get_base_path():
    """Returns the base path for resources, whether running after bundle or in
    development."""
    if getattr(sys, "frozen", False):
        # In bundled mode, resources live next to the launcher executable
        return BASE_DIR
    # Development mode: Go to project root instead of client
    # This ensures we find resources/openWakeWord correctly
    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # audio_modules
    client_dir = os.path.dirname(current_file_dir)  # client
    project_root = os.path.dirname(client_dir)  # Asystent
    return project_root


# Constants for audio recording
SAMPLE_RATE = 16000  # Hz (openWakeWord and Whisper typically use 16kHz)
CHUNK_DURATION_MS = 50  # openWakeWord processes audio in chunks (adjust if oww expects different chunk size)
CHUNK_SAMPLES = int(
    SAMPLE_RATE * CHUNK_DURATION_MS / 1000
)  # Samples per chunk for VAD and command recording
COMMAND_RECORD_TIMEOUT_SECONDS = 7  # Max duration for command recording
MIN_COMMAND_AUDIO_CHUNKS = 40  # Minimum audio chunks (2000ms) to ensure more time for command capture before silence detection
VAD_SILENCE_AMPLITUDE_THRESHOLD = (
    0.002  # Lowered threshold for better sensitivity (float32 audio).
)
# stt_silence_threshold (from config, in ms) is now used as duration of silence for VAD.


async def record_command_audio_async(
    mic_device_id: int, vad_silence_duration_ms: int, stop_event: threading.Event
) -> np.ndarray | None:
    """Records audio from the microphone until silence is detected or timeout. Uses a
    simple VAD based on amplitude.

    This async wrapper ensures compatibility with AGENTS.md requirements.
    """
    # Use executor to run the blocking audio recording in a separate thread
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, record_command_audio, mic_device_id, vad_silence_duration_ms, stop_event
    )


def record_command_audio(
    mic_device_id: int, vad_silence_duration_ms: int, stop_event: threading.Event
) -> np.ndarray | None:
    """Records audio from the microphone until silence is detected or timeout.

    Uses a simple VAD based on amplitude.
    """
    # Check if sounddevice is available
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("SoundDevice not available - cannot record command audio")
        play_beep("error", loop=False)
        return None

    logger.info(f"Recording command audio from device ID: {mic_device_id}...")
    audio_buffer = []

    # Calculate how many consecutive silent chunks constitute "silence" for VAD
    vad_silence_chunks_limit = max(1, vad_silence_duration_ms // CHUNK_DURATION_MS)
    silent_chunks_count = 0

    # Max command duration in chunks
    max_recording_chunks = COMMAND_RECORD_TIMEOUT_SECONDS * (1000 // CHUNK_DURATION_MS)

    try:
        # Using float32 for Whisper, ensure openWakeWord also handles it or convert if necessary.
        # openWakeWord's Model.predict() can take float32.
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=mic_device_id,
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            logger.info("Listening for command...")
            for _ in range(
                0, int(SAMPLE_RATE / CHUNK_SAMPLES * COMMAND_RECORD_TIMEOUT_SECONDS)
            ):  # Max recording duration
                if stop_event.is_set():
                    logger.debug("Stop event received during command recording.")
                    break
                audio_chunk, overflowed = stream.read(CHUNK_SAMPLES)
                if overflowed:
                    logger.warning("Input overflowed during command recording!")
                audio_buffer.append(audio_chunk)

                # Simple VAD: check RMS of the last few chunks
                if len(audio_buffer) > vad_silence_chunks_limit:
                    # Consider last 'vad_silence_chunks_limit' chunks for silence detection
                    current_segment = np.concatenate(
                        audio_buffer[-vad_silence_chunks_limit:]
                    )
                    rms = np.sqrt(np.mean(current_segment**2))
                    logger.debug(
                        f"VAD RMS: {rms:.6f} vs threshold {VAD_SILENCE_AMPLITUDE_THRESHOLD:.6f}, chunks: {len(audio_buffer)}"
                    )
                    if rms < VAD_SILENCE_AMPLITUDE_THRESHOLD:
                        silent_chunks_count += 1
                    else:
                        silent_chunks_count = 0  # Reset on sound

                    if (
                        silent_chunks_count >= vad_silence_chunks_limit
                        and len(audio_buffer) >= MIN_COMMAND_AUDIO_CHUNKS
                    ):
                        logger.info(
                            f"Silence detected after {len(audio_buffer) * CHUNK_DURATION_MS / 1000:.2f}s of audio."
                        )
                        break
                if len(audio_buffer) >= max_recording_chunks:
                    logger.info("Command recording reached maximum duration.")
                    break
    except Exception as e:
        if hasattr(sd, "PortAudioError") and isinstance(e, sd.PortAudioError):
            logger.error(f"PortAudio error during command recording: {e}")
            if "Invalid input device" in str(e):
                logger.error(
                    f"Invalid microphone device ID: {mic_device_id}. Please check your configuration."
                )
        else:
            logger.error(f"Error during command recording: {e}", exc_info=True)
        play_beep("error", loop=False)  # Play error beep
        return None

    if (
        not audio_buffer or len(audio_buffer) < MIN_COMMAND_AUDIO_CHUNKS / 2
    ):  # Check if enough audio was captured (e.g. more than 1 second)
        logger.info("No valid command audio recorded (too short or empty).")
        # play_beep("timeout", loop=False) # Consider if a timeout beep is desired here
        return None

    recorded_audio = np.concatenate(audio_buffer, axis=0)
    logger.info(
        f"Command audio recorded: {len(recorded_audio)/SAMPLE_RATE:.2f} seconds."
    )
    # Log audio characteristics
    logger.debug(
        f"Command audio stats: Max amp={np.max(np.abs(recorded_audio)):.4f}, Mean amp={np.mean(np.abs(recorded_audio)):.4f}, dtype={recorded_audio.dtype}"
    )
    return recorded_audio


async def run_wakeword_detection_async(
    mic_device_id: int | None,
    stt_silence_threshold_ms: int,  # Used for VAD in command recording
    wake_word_config_name: str,  # Name of wake word from config (for logging)
    tts_module,
    process_query_callback_async,
    async_event_loop: asyncio.AbstractEventLoop,
    oww_sensitivity_threshold: float,
    whisper_asr_instance,
    manual_listen_trigger_event: threading.Event,
    stop_detector_event: threading.Event,
):
    """Async wrapper for wakeword detection to comply with AGENTS.md requirements.

    Listens for wake word using openWakeWord and handles command
    recording/transcription.
    """
    # Use executor to run the blocking wakeword detection in a separate thread
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_wakeword_detection,
        mic_device_id,
        stt_silence_threshold_ms,
        wake_word_config_name,
        tts_module,
        process_query_callback_async,
        async_event_loop,
        oww_sensitivity_threshold,
        whisper_asr_instance,
        manual_listen_trigger_event,
        stop_detector_event,
    )


def run_wakeword_detection(
    mic_device_id: int | None,
    stt_silence_threshold_ms: int,  # Used for VAD in command recording
    wake_word_config_name: str,  # Name of wake word from config (for logging)
    tts_module,
    process_query_callback_async,
    async_event_loop: asyncio.AbstractEventLoop,
    oww_sensitivity_threshold: float,
    whisper_asr_instance,
    manual_listen_trigger_event: threading.Event,
    stop_detector_event: threading.Event,
):
    """Listens for wake word using openWakeWord and handles command
    recording/transcription."""

    # Get assistant instance function (may return None if not available)
    def get_assistant_instance():
        """Get assistant instance if available."""
        return None  # Stub implementation - will be filled by actual assistant if available

    # Check if sounddevice is available
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("SoundDevice not available - wake word detection disabled")
        play_beep("error", loop=False)
        return None

    # Dynamic import of openwakeword Model, with fallback to accommodate package structure
    _imported_model = None
    try:
        # Try importing from submodule
        from openwakeword.model import Model as _ImportedModel

        _imported_model = _ImportedModel
    except ImportError:
        try:
            # Fallback: import directly from package
            from openwakeword import Model as _ImportedModel

            _imported_model = _ImportedModel
        except ImportError as e:
            logger.error(f"openwakeword library import failed: {e}")
            _imported_model = None
    if _imported_model is None:
        OPENWAKEWORD_AVAILABLE = False
        logger.warning("openwakeword Model unavailable: wakeword detection disabled.")
        return None
    # Set the Model and availability flag
    Model = _imported_model
    OPENWAKEWORD_AVAILABLE = True
    if mic_device_id is None:
        try:
            default_devices = sd.query_devices()
            default_input_device_index = None
            for i, device in enumerate(default_devices):
                # Heuristic to find default input device
                if device["max_input_channels"] > 0:
                    try:
                        host_api_info = sd.query_hostapis()[device["hostapi"]]
                        if (
                            "default_input_device_name" in host_api_info
                            and device["name"]
                            == host_api_info["default_input_device_name"]
                        ):
                            default_input_device_index = i
                            logger.info(
                                f"Found host API default input device: ID {i} ({device['name']})"
                            )
                            break
                    except KeyError:
                        pass  # Continue if host API info is not as expected

            if (
                default_input_device_index is None
            ):  # Fallback to sounddevice's default if host API default not found or error
                try:
                    default_sd_device = sd.default.device
                    if (
                        isinstance(default_sd_device, (list, tuple))
                        and len(default_sd_device) > 0
                    ):
                        potential_default_idx = (
                            default_sd_device[0]
                            if isinstance(default_sd_device, (list, tuple))
                            else default_sd_device
                        )
                        if (
                            default_devices[potential_default_idx]["max_input_channels"]
                            > 0
                        ):
                            default_input_device_index = potential_default_idx
                            logger.info(
                                f"Using sounddevice default input device: ID {default_input_device_index} ({default_devices[default_input_device_index]['name']})"
                            )
                except Exception as e_sd_default:
                    logger.warning(
                        f"Could not determine sounddevice default input: {e_sd_default}. Will try first available."
                    )

            if (
                default_input_device_index is None
            ):  # Fallback to first available input device
                for i, device in enumerate(default_devices):
                    if device["max_input_channels"] > 0:
                        default_input_device_index = i
                        logger.info(
                            f"Using first available input device: ID {i} ({device['name']})"
                        )
                        break

            if default_input_device_index is not None:
                mic_device_id = default_input_device_index
                logger.info(
                    f"No microphone device ID specified, using determined default input device: ID {mic_device_id} ({sd.query_devices(mic_device_id)['name']})"
                )
            else:
                logger.error(
                    "Could not find any input audio device. Microphone device ID is required."
                )
                play_beep("error", loop=False)
                return
        except Exception as e:
            logger.error(
                f"Could not get default input device: {e}. Microphone device ID is required.",
                exc_info=True,
            )
            play_beep("error", loop=False)
            return

    # --- openWakeWord Model Initialization ---
    # Construct path to 'resources/openWakeWord'
    base_project_dir = (
        get_base_path()
    )  # Use the new helper function to get the project's base directory
    logger.info(f"[Debug] base_project_dir for oww: {base_project_dir}")
    # The model_dir should point to a directory named 'openWakeWord' directly inside 'resources'
    # which is itself at the same level as the executable (or main script).
    model_dir = os.path.join(base_project_dir, "resources", "openWakeWord")
    logger.info(f"[Debug] Attempting to use oww model_dir: {model_dir}")

    wakeword_model_instance = None
    try:
        # Initialize openWakeWord model using custom models from resources/openWakeWord
        if not os.path.exists(model_dir):
            logger.warning(
                "openWakeWord directory not found, wake word detection disabled"
            )
            return None  # Gather custom ONNX models (exclude helper files)
        custom_models = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".onnx")
            and not any(
                x in f.lower() for x in ["preprocessor", "embedding", "melspectrogram"]
            )
        ]

        if not custom_models:
            logger.warning("No custom ONNX models found, wake word detection disabled")
            return None

        logger.info(
            f"Found {len(custom_models)} custom ONNX wake word models: {[os.path.basename(m) for m in custom_models]}"
        )

        # Attempt to initialize openWakeWord model
        try:
            # Initialize with custom models only - avoid embedding model issues
            # by using only minimal required parameters
            model_kwargs = {
                "wakeword_models": custom_models,
                "inference_framework": "onnx",
            }

            # Melspectrogram model - musi pochodzić z naszego folderu resources
            melspec_path = os.path.join(model_dir, "melspectrogram.onnx")
            if os.path.exists(melspec_path):
                model_kwargs["melspec_model_path"] = melspec_path
                logger.info(
                    f"Używanie niestandardowego modelu melspektrogramu z: {melspec_path}"
                )
            else:
                logger.error(
                    f"KRYTYCZNY BŁĄD: Model melspektrogramu nie został znaleziony w {melspec_path}. "
                    "Detekcja słowa kluczowego nie może zostać uruchomiona."
                )
                return None  # Przerywamy inicjalizację            # Model embedding: Próbujemy wyłączyć używając None
            embedding_path = os.path.join(model_dir, "embedding_model.onnx")
            if os.path.exists(embedding_path):
                logger.warning(
                    f"Znaleziono embedding_model.onnx w {embedding_path}, ale jest niekompatybilny. Pomijamy go."
                )
            else:
                logger.info("Brak embedding_model.onnx w resources/openWakeWord")

            # Sprawdźmy czy możemy pominąć embedding_model_path całkowicie
            logger.info("Inicjalizacja OpenWakeWord bez parametru embedding_model_path")
            wakeword_model_instance = Model(**model_kwargs)
            logger.info("openWakeWord Model initialized with custom ONNX models")
        except Exception as e:
            logger.error(f"Error initializing openWakeWord Model: {e}", exc_info=True)
            logger.warning("Wake word detection disabled due to initialization failure")
            return None

        logger.info(
            f"openWakeWord Model initialized. Expected frame length: {getattr(wakeword_model_instance, 'expected_frame_length', 'N/A')}"
        )

    except ImportError:
        logger.error(
            "openwakeword library is required. Install via `pip install openwakeword`.",
            exc_info=True,
        )
        play_beep("error", loop=False)
        return
    except Exception as e:
        logger.error(f"Error initializing openWakeWord Model: {e}", exc_info=True)
        play_beep("error", loop=False)
        return

    if wakeword_model_instance is None:
        logger.error("Failed to create wakeword_model_instance. Aborting.")
        return

    logger.info(
        f"Starting wake word detection for '{wake_word_config_name}' using openWakeWord."
    )
    logger.info("Gaja jest załadowana i nasłuchuje.")

    audio_data_queue = queue.Queue()

    def sd_callback(indata, frames, time_info, status_flags):
        if status_flags:
            logger.warning(f"sounddevice InputStream status: {status_flags}")
        audio_data_queue.put(indata.copy())

    # Determine the correct chunk size for openWakeWord
    oww_chunk_size = getattr(wakeword_model_instance, "expected_frame_length", None)
    if oww_chunk_size is None:
        logger.warning(
            "Could not determine expected_frame_length from oww model. Defaulting to 1280 samples (80ms @ 16kHz)."
        )
        oww_chunk_size = 1280  # A common default for oww
    else:
        logger.info(
            f"Using openWakeWord expected_frame_length for audio chunks: {oww_chunk_size} samples."
        )

    audio_stream = None  # Initialize to None for finally block
    try:
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=mic_device_id,
            channels=1,
            dtype="int16",  # openWakeWord expects int16 PCM
            blocksize=oww_chunk_size,  # Use oww_chunk_size
            callback=sd_callback,
        )
        with audio_stream:
            logger.info(
                f"Audio stream started on device ID {mic_device_id} for wake word detection with chunk size {oww_chunk_size}."
            )
            while not stop_detector_event.is_set():
                try:
                    if manual_listen_trigger_event.is_set():
                        logger.info("Manual listen trigger activated.")
                        play_beep("listening_start", loop=False)
                        if tts_module:
                            tts_module.cancel()

                        # Update shared state for manual trigger
                        update_wake_word_state(detected=True)
                        update_listening_state(listening=True)

                        assistant = get_assistant_instance()
                        if assistant:
                            assistant.is_listening = True
                        command_audio_data_np = record_command_audio(
                            mic_device_id, stt_silence_threshold_ms, stop_detector_event
                        )
                        manual_listen_trigger_event.clear()

                        if (
                            command_audio_data_np is not None
                            and command_audio_data_np.size > 0
                        ):
                            if whisper_asr_instance:
                                logger.info(
                                    "Transcribing command with Whisper (manual trigger)..."
                                )
                                audio_to_transcribe = command_audio_data_np.flatten()
                                logger.debug(
                                    f"Audio to Whisper (manual): Max amp={np.max(np.abs(audio_to_transcribe)):.4f}, Mean amp={np.mean(np.abs(audio_to_transcribe)):.4f}, dtype={audio_to_transcribe.dtype}, shape={audio_to_transcribe.shape}"
                                )
                                transcribed_text = whisper_asr_instance.transcribe(
                                    audio_to_transcribe, sample_rate=SAMPLE_RATE
                                )
                                logger.info(
                                    f"Whisper transcription (manual): '{transcribed_text}'"
                                )
                                if transcribed_text:
                                    if async_event_loop and hasattr(
                                        async_event_loop, "call_soon_threadsafe"
                                    ):
                                        # Use event loop if available
                                        asyncio.run_coroutine_threadsafe(
                                            process_query_callback_async(
                                                transcribed_text
                                            ),
                                            async_event_loop,
                                        )
                                    else:
                                        # Fallback: run in thread pool or directly
                                        logger.warning(
                                            "No event loop available, running callback directly"
                                        )
                                        try:
                                            # Try to run async callback in new loop
                                            import asyncio

                                            loop = asyncio.new_event_loop()
                                            asyncio.set_event_loop(loop)
                                            loop.run_until_complete(
                                                process_query_callback_async(
                                                    transcribed_text
                                                )
                                            )
                                            loop.close()
                                        except Exception as e:
                                            logger.error(
                                                f"Error running async callback: {e}"
                                            )
                                else:
                                    logger.warning(
                                        "Whisper returned empty transcription (manual)."
                                    )
                                    play_beep("error", loop=False)
                            else:
                                logger.error(
                                    "Whisper ASR instance missing for manual trigger."
                                )
                        else:
                            logger.warning(
                                "No audio recorded for manual trigger command."
                            )

                        # Reset shared state after manual trigger
                        update_listening_state(listening=False)
                        update_wake_word_state(detected=False)

                        if assistant:
                            assistant.is_listening = False
                        if wakeword_model_instance:
                            wakeword_model_instance.reset()
                        continue

                    try:
                        audio_chunk_int16 = audio_data_queue.get(timeout=0.1)
                    except queue.Empty:
                        if stop_detector_event.is_set():
                            break
                        continue

                    if wakeword_model_instance is None:
                        logger.error("wakeword_model_instance is None, cannot predict.")
                        time.sleep(
                            0.1
                        )  # Avoid busy loop if model failed to init but thread continued
                        continue

                    prediction_scores = wakeword_model_instance.predict(
                        audio_chunk_int16.flatten()
                    )
                    for model_name_key, score_value in prediction_scores.items():
                        if score_value >= oww_sensitivity_threshold:
                            logger.info(
                                f"Wake word '{model_name_key}' detected with score: {score_value:.2f} (threshold: {oww_sensitivity_threshold:.2f})"
                            )
                            play_beep("listening_start", loop=False)
                            if tts_module:
                                tts_module.cancel()

                            # Update shared state for wake word detection
                            update_wake_word_state(detected=True)
                            update_listening_state(listening=True)

                            assistant = get_assistant_instance()
                            if assistant:
                                assistant.is_listening = True
                            command_audio_data_np = record_command_audio(
                                mic_device_id,
                                stt_silence_threshold_ms,
                                stop_detector_event,
                            )
                            if (
                                command_audio_data_np is not None
                                and command_audio_data_np.size > 0
                            ):
                                logger.info(
                                    f"Recorded command audio: {command_audio_data_np.shape} samples, max_amp={np.max(np.abs(command_audio_data_np)):.4f}"
                                )
                                if whisper_asr_instance:
                                    logger.info(
                                        f"Using Whisper ASR instance: {type(whisper_asr_instance).__name__}"
                                    )
                                    logger.info(
                                        f"Whisper available: {getattr(whisper_asr_instance, 'available', 'unknown')}"
                                    )
                                    logger.info(
                                        f"Whisper model: {getattr(whisper_asr_instance, 'model_id', 'unknown')}"
                                    )
                                    logger.info(
                                        "Transcribing command with Whisper after wake word..."
                                    )
                                    audio_to_transcribe = (
                                        command_audio_data_np.flatten()
                                    )
                                    logger.debug(
                                        f"Audio to Whisper: Max amp={np.max(np.abs(audio_to_transcribe)):.4f}, Mean amp={np.mean(np.abs(audio_to_transcribe)):.4f}, dtype={audio_to_transcribe.dtype}, shape={audio_to_transcribe.shape}"
                                    )

                                    # Check if audio has reasonable amplitude
                                    max_amplitude = np.max(np.abs(audio_to_transcribe))
                                    if max_amplitude < 0.001:
                                        logger.warning(
                                            f"Audio amplitude very low: {max_amplitude:.6f} - may be silent"
                                        )

                                    transcribed_text = whisper_asr_instance.transcribe(
                                        audio_to_transcribe, sample_rate=SAMPLE_RATE
                                    )
                                    logger.info(
                                        f"Whisper transcription: '{transcribed_text}'"
                                    )
                                    if transcribed_text:
                                        if async_event_loop and hasattr(
                                            async_event_loop, "call_soon_threadsafe"
                                        ):
                                            # Use event loop if available
                                            asyncio.run_coroutine_threadsafe(
                                                process_query_callback_async(
                                                    transcribed_text
                                                ),
                                                async_event_loop,
                                            )
                                        else:
                                            # Fallback: run in thread pool or directly
                                            logger.warning(
                                                "No event loop available, running callback directly"
                                            )
                                            try:
                                                # Try to run async callback in new loop
                                                import asyncio

                                                loop = asyncio.new_event_loop()
                                                asyncio.set_event_loop(loop)
                                                loop.run_until_complete(
                                                    process_query_callback_async(
                                                        transcribed_text
                                                    )
                                                )
                                                loop.close()
                                            except Exception as e:
                                                logger.error(
                                                    f"Error running async callback: {e}"
                                                )
                                    else:
                                        logger.warning(
                                            "Whisper returned empty transcription after wake word."
                                        )
                                        play_beep("error", loop=False)
                                else:
                                    logger.error("Whisper ASR instance missing.")
                            else:
                                logger.warning(
                                    "No audio recorded for command after wake word."
                                )

                            # Reset shared state after wake word processing
                            update_listening_state(listening=False)
                            update_wake_word_state(detected=False)

                            if assistant:
                                assistant.is_listening = False
                            if wakeword_model_instance:
                                wakeword_model_instance.reset()
                            break

                except Exception as e:
                    logger.error(
                        f"Error in wake word detection loop: {e}", exc_info=True
                    )
                    if stop_detector_event.is_set():
                        break
                    time.sleep(0.1)

    except Exception as e:
        if hasattr(sd, "PortAudioError") and isinstance(e, sd.PortAudioError):
            logger.error(
                f"PortAudioError during audio stream setup: {e}", exc_info=True
            )
        else:
            logger.error(
                f"Fatal error in run_wakeword_detection setup or stream: {e}",
                exc_info=True,
            )
        play_beep("error", loop=False)
    finally:
        logger.info("Wake word detection thread finishing.")
        if audio_stream is not None and not audio_stream.closed:
            try:
                audio_stream.stop()
                audio_stream.close()
                logger.info("Audio stream closed.")
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}", exc_info=True)
    return None
