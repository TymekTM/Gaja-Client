import asyncio
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from .sounddevice_loader import get_sounddevice, is_sounddevice_available
from .beep_sounds import play_beep

# Make shared_state importable when frozen / dev
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_state import update_listening_state, update_wake_word_state  # type: ignore

try:  # BASE_DIR for frozen mode
    from client.config import BASE_DIR  # type: ignore
except Exception:  # pragma: no cover - fallback
    if getattr(sys, "frozen", False):
        BASE_DIR = os.path.dirname(sys.executable)
    else:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sd = get_sounddevice()
SOUNDDEVICE_AVAILABLE = is_sounddevice_available()

# Audio / detection constants
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 50
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
COMMAND_RECORD_TIMEOUT_SECONDS = 7
MIN_COMMAND_AUDIO_CHUNKS = 40  # ~2s
VAD_SILENCE_AMPLITUDE_THRESHOLD = 0.002

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_base_path() -> str:
    if getattr(sys, "frozen", False):
        return BASE_DIR
    current = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current)  # Gaja-Client root


def _import_openwakeword_model():
    try:
        from openwakeword.model import Model as _M  # type: ignore
        return _M
    except ImportError:
        try:
            from openwakeword import Model as _M  # type: ignore
            return _M
        except ImportError as e:
            logger.warning(f"openwakeword unavailable: {e}")
            return None


def _select_microphone(explicit_id: Optional[int]) -> Optional[int]:
    if not SOUNDDEVICE_AVAILABLE:
        return None
    if explicit_id is not None:
        return explicit_id
    try:
        devices = sd.query_devices()  # type: ignore[attr-defined]
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                return i
    except Exception as e:
        logger.error(f"Device enumeration failed: {e}")
    return None

# ---------------------------------------------------------------------------
# Command recording (blocking, used inside detector thread)
# ---------------------------------------------------------------------------


def record_command_audio(mic_device_id: int, vad_silence_duration_ms: int, stop_event: threading.Event) -> Optional[np.ndarray]:
    if not SOUNDDEVICE_AVAILABLE:
        logger.error("sounddevice missing - cannot record command audio")
        play_beep("error", loop=False)
        return None
    vad_limit = max(1, vad_silence_duration_ms // CHUNK_DURATION_MS)
    silent_chunks = 0
    audio_chunks: list[np.ndarray] = []
    max_chunks = COMMAND_RECORD_TIMEOUT_SECONDS * (1000 // CHUNK_DURATION_MS)
    try:
        with sd.InputStream(  # type: ignore[attr-defined]
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=mic_device_id,
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            for _ in range(int(SAMPLE_RATE / CHUNK_SAMPLES * COMMAND_RECORD_TIMEOUT_SECONDS)):
                if stop_event.is_set():
                    break
                chunk, overflow = stream.read(CHUNK_SAMPLES)  # type: ignore[attr-defined]
                if overflow:
                    logger.debug("Input overflow during command capture")
                audio_chunks.append(chunk)
                if len(audio_chunks) > vad_limit:
                    seg = np.concatenate(audio_chunks[-vad_limit:])
                    rms = float(np.sqrt(np.mean(seg**2)))
                    if rms < VAD_SILENCE_AMPLITUDE_THRESHOLD:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    if silent_chunks >= vad_limit and len(audio_chunks) >= MIN_COMMAND_AUDIO_CHUNKS:
                        logger.info("Silence detected -> stop recording")
                        break
                if len(audio_chunks) >= max_chunks:
                    logger.info("Max command duration reached")
                    break
    except Exception as e:
        logger.error(f"Command recording error: {e}")
        play_beep("error", loop=False)
        return None
    if not audio_chunks or len(audio_chunks) < MIN_COMMAND_AUDIO_CHUNKS / 2:
        logger.info("Recorded command too short / empty")
        return None
    data = np.concatenate(audio_chunks, axis=0)
    logger.info(f"Recorded command audio: {len(data)/SAMPLE_RATE:.2f}s")
    return data

# ---------------------------------------------------------------------------
# Unified Detector
# ---------------------------------------------------------------------------


class UnifiedWakewordDetector:
    """Single wakeword detector implementation (replaces optimized / advanced / simple).

    Features:
    - openWakeWord ONNX inference (if available)
    - Energy fallback when model missing
    - Command audio capture with simple VAD
    - Whisper integration
    - Threaded detection + async-safe callbacks
    - Configuration update at runtime
    """

    def __init__(self, config: dict, callback: Callable[[str], Any]):
        self.enabled: bool = config.get("enabled", True)
        self.keyword: str = config.get("keyword", "gaja").lower()
        self.sensitivity: float = float(config.get("sensitivity", 0.6))
        self.device_id: Optional[int] = config.get("device_id")
        self.stt_silence_threshold_ms: int = int(config.get("stt_silence_threshold_ms", 2000))
        self._callback = callback

        self._stop_event = threading.Event()
        self._manual_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._model = None
        self._chunk_size = 1280  # default
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.whisper_asr = None

    # ---- Public API used elsewhere ----
    def start_detection(self):
        if not self.enabled:
            logger.info("Wakeword disabled in config")
            return
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("sounddevice unavailable - detection disabled")
            return
        if self._thread and self._thread.is_alive():
            logger.debug("Wakeword already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Unified wakeword detection started")

    def stop_detection(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("Unified wakeword detection stopped")

    def trigger_manual_detection(self):
        self._manual_event.set()

    def set_whisper_asr(self, whisper_asr):
        self.whisper_asr = whisper_asr

    def set_device(self, device_id: int | None):  # for settings_manager
        self.device_id = device_id
        logger.info(f"Wakeword input device set to {device_id}")

    def update_config(self, new_conf: dict):  # for settings_manager
        if "sensitivity" in new_conf:
            self.sensitivity = float(new_conf["sensitivity"])
        if "keyword" in new_conf:
            self.keyword = str(new_conf["keyword"]).lower()
        if "device_id" in new_conf:
            self.device_id = new_conf["device_id"]
        if "stt_silence_threshold_ms" in new_conf:
            self.stt_silence_threshold_ms = int(new_conf["stt_silence_threshold_ms"])
        logger.info(f"Wakeword config updated: keyword={self.keyword} sens={self.sensitivity}")

    # ---- Internal ----
    def _init_model(self):
        model_cls = _import_openwakeword_model()
        if not model_cls:
            return
        try:
            base_dir = get_base_path()
            model_dir = os.path.join(base_dir, "resources", "openWakeWord")
            if not os.path.isdir(model_dir):
                logger.warning("openWakeWord model directory missing -> energy fallback")
                return
            model_files = [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".onnx") and not any(x in f.lower() for x in ["preprocessor", "embedding", "melspectrogram"])
            ]
            if not model_files:
                logger.warning("No ONNX wakeword models found -> energy fallback")
                return
            kwargs = {"wakeword_models": model_files, "inference_framework": "onnx"}
            melspec = os.path.join(model_dir, "melspectrogram.onnx")
            if os.path.exists(melspec):
                kwargs["melspec_model_path"] = melspec
            self._model = model_cls(**kwargs)
            self._chunk_size = getattr(self._model, "expected_frame_length", 1280)
            logger.info(f"openWakeWord initialized ({len(model_files)} models, chunk {self._chunk_size})")
        except Exception as e:
            logger.error(f"openWakeWord init failed: {e}")
            self._model = None

    def _run_loop(self):
        # Capture running loop (if any) for async callbacks
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        self.device_id = _select_microphone(self.device_id)
        if self.device_id is None:
            logger.error("No microphone device available")
            return
        self._init_model()

        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        def sd_callback(indata, frames, time_info, status):  # pragma: no cover (I/O)
            if status:
                logger.debug(f"Input status: {status}")
            audio_q.put(indata.copy())

        try:
            with sd.InputStream(  # type: ignore[attr-defined]
                samplerate=SAMPLE_RATE,
                device=self.device_id,
                channels=1,
                dtype="int16",
                blocksize=self._chunk_size,
                callback=sd_callback,
            ):
                logger.info(f"Wakeword loop active (device {self.device_id})")
                while not self._stop_event.is_set():
                    # Manual trigger
                    if self._manual_event.is_set():
                        self._manual_event.clear()
                        self._handle_detection(manual=True)
                        continue
                    try:
                        frame = audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    # Model path - fail if model not available
                    if self._model is not None:
                        try:
                            scores = self._model.predict(frame.flatten())
                            for name, score in scores.items():
                                if score >= self.sensitivity:
                                    logger.info(f"Wakeword '{name}' detected (score {score:.2f})")
                                    self._handle_detection()
                                    try:
                                        self._model.reset()
                                    except Exception:
                                        pass
                                    break
                        except Exception as e:
                            logger.error(f"Model prediction error: {e}")
                    else:
                        # No fallback - fail if wakeword model not available
                        logger.error("Wakeword model not available and no fallback configured")
                        raise RuntimeError("Wakeword model not initialized - cannot detect wake words")
        except Exception as e:
            if hasattr(sd, "PortAudioError") and isinstance(e, getattr(sd, "PortAudioError")):  # type: ignore[attr-defined]
                logger.error(f"PortAudioError: {e}")
            else:
                logger.error(f"Wakeword loop error: {e}")
        finally:
            logger.info("Wakeword loop finished")

    # Core detection handling
    def _handle_detection(self, manual: bool = False):  # pragma: no cover (I/O heavy)
        play_beep("listening_start", loop=False)
        update_wake_word_state(detected=True)
        update_listening_state(listening=True)
        audio = record_command_audio(self.device_id or 0, self.stt_silence_threshold_ms, self._stop_event)
        update_listening_state(listening=False)
        update_wake_word_state(detected=False)
        if audio is None or audio.size == 0:
            logger.warning("No command audio captured")
            return
        transcription = None
        if self.whisper_asr:
            try:
                transcription = self.whisper_asr.transcribe(audio.flatten(), sample_rate=SAMPLE_RATE)
                logger.info(f"Transcription: {transcription}")
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")
        
        # Only use transcription if it's valid and not empty
        if transcription and transcription.strip():
            phrase = transcription.strip()
        else:
            # If no valid transcription, use just the keyword (wake word only)
            phrase = self.keyword
            logger.info(f"No valid transcription, using wake word: {phrase}")
            
        self._dispatch_callback(phrase)

    def _dispatch_callback(self, text: str):
        if not self._callback:
            return
        try:
            if asyncio.iscoroutinefunction(self._callback):
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(self._callback(text), self._loop)
                else:
                    asyncio.run(self._callback(text))
            else:
                self._callback(text)
        except Exception as e:
            logger.error(f"Wakeword callback error: {e}")

# ---------------------------------------------------------------------------
# Factory (legacy compatibility)
# ---------------------------------------------------------------------------


def create_wakeword_detector(config: dict, callback: Callable[[str], Any]) -> UnifiedWakewordDetector:
    return UnifiedWakewordDetector(config, callback)

# Simple self-test when executed directly
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    det = create_wakeword_detector({"enabled": False}, lambda q: print("DETECTED", q))
    det.start_detection()
    time.sleep(1)
    det.stop_detection()
