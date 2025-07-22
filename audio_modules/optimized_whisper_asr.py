#!/usr/bin/env python3
"""Optimized Whisper ASR module for GAJA Assistant.

This module provides high-performance speech recognition with optimized parameters,
efficient resource management, and enhanced accuracy for Polish language processing.

Follows AGENTS.md guidelines: async, testable, modular architecture.
"""

import asyncio
import ctypes
import logging
import os
import pathlib
import sys
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Performance and quality optimizations
OPTIMAL_BEAM_SIZE = 5  # Balanced speed/quality
OPTIMAL_TEMPERATURE = 0.0  # Deterministic results
OPTIMAL_COMPRESSION_THRESHOLD = 2.4  # Better for Polish
OPTIMAL_LOG_PROB_THRESHOLD = -1.0  # More lenient for non-native pronunciation
OPTIMAL_NO_SPEECH_THRESHOLD = 0.6  # Higher threshold for better speech detection
OPTIMAL_PATIENCE = 1  # Reduced for faster processing

# VAD parameters optimized for command recognition
VAD_MIN_SILENCE_DURATION = 500  # ms
VAD_SPEECH_PAD = 400  # ms
VAD_THRESHOLD = 0.5  # More sensitive

# Memory and performance settings
MAX_CACHE_SIZE = 3  # Keep max 3 models in memory
CPU_THREAD_LIMIT = 4  # Optimal for most systems
GPU_MEMORY_FRACTION = 0.7  # Leave some GPU memory for other tasks

# Language settings optimized for Polish
LANGUAGE_PRIORITY = ["pl", "en"]  # Polish first, English fallback
POLISH_SPECIFIC_PARAMS = {
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "suppress_tokens": [-1],  # Remove some common noise tokens
    "condition_on_previous_text": False,  # Better for short commands
}


class OptimizedWhisperASR:
    """High-performance Whisper ASR with optimizations for Polish language."""

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str | None = None,
        language: str = "pl",
    ):
        """Initialize optimized Whisper ASR.

        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cpu", "cuda", or None for auto-detection)
            compute_type: Computation type ("int8", "float16", "float32")
            language: Primary language for recognition
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self.available = False
        self.model_id = None

        # Performance tracking
        self.transcription_count = 0
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

        # Initialize device and compute type
        self.device, self.compute_type = self._optimize_device_settings(
            device, compute_type
        )

        # Load model
        self._load_optimized_model()

    def _optimize_device_settings(
        self, device: str | None, compute_type: str | None
    ) -> tuple[str, str]:
        """Optimize device and compute type settings.

        Args:
            device: Requested device
            compute_type: Requested compute type

        Returns:
            Optimized device and compute type
        """
        # Auto-detect optimal device
        if device is None:
            if self._is_gpu_available():
                device = "cuda"
                logger.info("GPU detected, using CUDA acceleration")
            else:
                device = "cpu"
                logger.info("Using CPU for processing")

        # Optimize compute type based on device
        if compute_type is None:
            if device == "cuda":
                compute_type = "float16"  # Optimal for GPU
            else:
                compute_type = "int8"  # Optimal for CPU

        # Set CPU thread limits for optimal performance
        if device == "cpu":
            self._optimize_cpu_settings()

        return device, compute_type

    def _is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available.

        Returns:
            True if GPU can be used
        """
        try:
            # Check if CUDA libraries are available
            if not self._ensure_cuda_libraries():
                return False

            # Check torch CUDA availability
            try:
                import torch

                if not torch.cuda.is_available():
                    return False

                # Check if it's not an AMD GPU (which doesn't support CUDA)
                device_name = torch.cuda.get_device_name(0)
                if any(keyword in device_name.lower() for keyword in ("amd", "radeon")):
                    logger.warning(
                        f"AMD GPU detected ({device_name}) - CUDA not supported"
                    )
                    return False

                logger.info(f"CUDA GPU available: {device_name}")
                return True

            except ImportError:
                logger.warning("PyTorch not available - cannot use GPU")
                return False

        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            return False

    def _ensure_cuda_libraries(self) -> bool:
        """Ensure CUDA libraries are available.

        Returns:
            True if CUDA libraries are accessible
        """
        try:
            # Try to load cuBLAS
            ctypes.CDLL("cublas64_12.dll")
            return True
        except OSError:
            # Check for bundled CUDA libraries
            if getattr(sys, "frozen", False):
                # Running as PyInstaller bundle
                cuda_paths = [
                    os.path.join(sys._MEIPASS, "cublas64_12.dll"),
                    os.path.join(
                        os.path.dirname(sys.executable), "_internal", "cublas64_12.dll"
                    ),
                ]

                for cuda_path in cuda_paths:
                    if os.path.exists(cuda_path):
                        try:
                            ctypes.CDLL(cuda_path)
                            # Add directory to PATH for other dependencies
                            cuda_dir = os.path.dirname(cuda_path)
                            if cuda_dir not in os.environ.get("PATH", ""):
                                os.environ["PATH"] = (
                                    cuda_dir + os.pathsep + os.environ.get("PATH", "")
                                )
                            return True
                        except OSError:
                            continue

            logger.warning("CUDA libraries not found - using CPU")
            return False

    def _optimize_cpu_settings(self) -> None:
        """Optimize CPU settings for performance."""
        try:
            # Set CPU thread limit
            import torch

            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(CPU_THREAD_LIMIT)

            # Set environment variables for optimal CPU performance
            os.environ["OMP_NUM_THREADS"] = str(CPU_THREAD_LIMIT)
            os.environ["MKL_NUM_THREADS"] = str(CPU_THREAD_LIMIT)

            logger.info(f"CPU optimized with {CPU_THREAD_LIMIT} threads")

        except Exception as e:
            logger.warning(f"Could not optimize CPU settings: {e}")

    def _load_optimized_model(self) -> None:
        """Load the Whisper model with optimizations."""
        try:
            from faster_whisper import WhisperModel

            # Determine model path/name
            model_candidates = self._get_model_candidates()

            for candidate in model_candidates:
                try:
                    logger.info(f"Loading Whisper model: {candidate}")

                    # Create model with optimized settings
                    self.model = WhisperModel(
                        candidate,
                        device=self.device,
                        compute_type=self.compute_type,
                        cpu_threads=CPU_THREAD_LIMIT if self.device == "cpu" else 0,
                        num_workers=1,  # Single worker for lower latency
                        download_root=self._get_cache_directory(),
                    )

                    self.model_id = candidate
                    self.available = True

                    logger.info(f"Successfully loaded Whisper model: {candidate}")
                    break

                except Exception as e:
                    logger.warning(f"Failed to load model {candidate}: {e}")
                    continue

            if not self.available:
                logger.error("Could not load any Whisper model")

        except ImportError:
            logger.error("faster-whisper not available - speech recognition disabled")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")

    def _get_model_candidates(self) -> list[str]:
        """Get list of model candidates in order of preference.

        Returns:
            List of model names/paths to try
        """
        base_name = self.model_size.lower()

        # Remove common prefixes if present
        if base_name.startswith("openai/whisper-"):
            base_name = base_name.replace("openai/whisper-", "")
        elif base_name.startswith("whisper-"):
            base_name = base_name.replace("whisper-", "")

        # Prefer Systran faster-whisper models
        candidates = [
            f"Systran/faster-whisper-{base_name}",
            f"openai/whisper-{base_name}",
            base_name,
        ]

        return candidates

    def _get_cache_directory(self) -> str:
        """Get optimized cache directory for models.

        Returns:
            Path to cache directory
        """
        if getattr(sys, "frozen", False):
            # Running as bundle - use persistent cache
            exe_dir = pathlib.Path(sys.executable).parent
            cache_dir = exe_dir / "user_data" / "whisper_cache"
        else:
            # Development mode
            cache_dir = pathlib.Path(__file__).parent.parent.parent / ".whisper_cache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    async def transcribe_async(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Transcribe audio asynchronously.

        Args:
            audio: Audio data or path to audio file
            language: Language for transcription (None for auto-detect)
            beam_size: Beam size for decoding
            temperature: Temperature for sampling

        Returns:
            Transcribed text
        """
        if not self.available:
            logger.error("Whisper ASR not available")
            return ""

        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_blocking, audio, language, beam_size, temperature
        )

    def transcribe(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
        sample_rate: int | None = None,
    ) -> str:
        """Transcribe audio (synchronous version).

        Args:
            audio: Audio data or path to audio file
            language: Language for transcription (None for auto-detect)
            beam_size: Beam size for decoding
            temperature: Temperature for sampling
            sample_rate: Sample rate (for compatibility, not used)

        Returns:
            Transcribed text
        """
        return self._transcribe_blocking(audio, language, beam_size, temperature)

    def _transcribe_blocking(
        self,
        audio: str | np.ndarray,
        language: str | None = None,
        beam_size: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Blocking transcription implementation.

        Args:
            audio: Audio data or path to audio file
            language: Language for transcription
            beam_size: Beam size for decoding
            temperature: Temperature for sampling

        Returns:
            Transcribed text
        """
        if not self.available:
            return ""

        try:
            import time

            start_time = time.time()
        except Exception:
            start_time = 0

        try:
            # Use provided parameters or optimized defaults
            language = language or self.language
            beam_size = beam_size or OPTIMAL_BEAM_SIZE
            temperature = (
                temperature if temperature is not None else OPTIMAL_TEMPERATURE
            )

            # Calculate audio duration for performance tracking
            if isinstance(audio, np.ndarray):
                audio_duration = len(audio) / 16000  # Assume 16kHz
            else:
                audio_duration = 0  # Unknown for file paths

            # Transcribe with optimized parameters
            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                temperature=temperature,
                # VAD parameters optimized for commands
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION,
                    "speech_pad_ms": VAD_SPEECH_PAD,
                    "threshold": VAD_THRESHOLD,
                },
                # Quality parameters
                no_speech_threshold=OPTIMAL_NO_SPEECH_THRESHOLD,
                log_prob_threshold=OPTIMAL_LOG_PROB_THRESHOLD,
                compression_ratio_threshold=OPTIMAL_COMPRESSION_THRESHOLD,
                patience=OPTIMAL_PATIENCE,
                # Polish-specific optimizations
                **POLISH_SPECIFIC_PARAMS,
            )

            # Extract and clean text
            text_segments = []
            for segment in segments:
                if segment.text.strip():
                    # Clean and normalize text
                    clean_text = self._clean_transcription(segment.text)
                    if clean_text:
                        text_segments.append(clean_text)

            result = " ".join(text_segments).strip()

            # Performance tracking
            self.transcription_count += 1
            self.total_audio_duration += audio_duration

            try:
                import time

                processing_time = time.time() - start_time
                self.total_processing_time += processing_time

                # Log performance metrics
                if self.transcription_count % 10 == 0:  # Every 10 transcriptions
                    avg_processing_time = (
                        self.total_processing_time / self.transcription_count
                    )
                    logger.info(
                        f"Whisper performance: {avg_processing_time:.3f}s avg processing time"
                    )
            except Exception:
                pass  # Ignore timing errors

            # Log result with confidence info
            confidence = getattr(info, "language_probability", 0.0)
            logger.info(f"Transcribed: '{result}' (confidence: {confidence:.2f})")

            return result

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            # Try fallback transcription with minimal parameters
            try:
                segments, _ = self.model.transcribe(
                    audio, language=language, beam_size=1, temperature=0.0
                )
                result = " ".join(s.text.strip() for s in segments if s.text.strip())
                logger.info(f"Fallback transcription: '{result}'")
                return result
            except Exception as fallback_e:
                logger.error(f"Fallback transcription failed: {fallback_e}")
                return ""

    def _clean_transcription(self, text: str) -> str:
        """Clean and normalize transcription text.

        Args:
            text: Raw transcription text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        cleaned = " ".join(text.strip().split())

        # Remove common transcription artifacts
        artifacts = ["[BLANK_AUDIO]", "[MUSIC]", "[NOISE]", "♪", "♫"]
        for artifact in artifacts:
            cleaned = cleaned.replace(artifact, "").strip()

        # Polish-specific cleaning
        # Remove common filler words that might be misrecognized
        filler_patterns = ["eee", "mmm", "aaa", "yyy"]
        for pattern in filler_patterns:
            if cleaned.lower() == pattern:
                return ""

        return cleaned

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if self.transcription_count == 0:
            return {
                "transcriptions": 0,
                "avg_processing_time": 0.0,
                "total_audio_duration": 0.0,
                "real_time_factor": 0.0,
            }

        avg_processing_time = self.total_processing_time / self.transcription_count
        real_time_factor = (
            self.total_processing_time / self.total_audio_duration
            if self.total_audio_duration > 0
            else 0.0
        )

        return {
            "transcriptions": self.transcription_count,
            "avg_processing_time": avg_processing_time,
            "total_audio_duration": self.total_audio_duration,
            "real_time_factor": real_time_factor,
            "device": self.device,
            "compute_type": self.compute_type,
            "model_id": self.model_id,
        }

    def unload(self) -> None:
        """Unload the model and free resources."""
        if hasattr(self, "model") and self.model is not None:
            logger.info(f"Unloading Whisper model: {self.model_id}")
            del self.model
            self.model = None
            self.available = False

            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                try:
                    import torch

                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared")
                except ImportError:
                    pass


# Factory functions for compatibility and convenience
async def create_optimized_whisper_async(
    model_size: str = "base", device: str | None = None, language: str = "pl"
) -> OptimizedWhisperASR:
    """Create an optimized Whisper ASR instance asynchronously.

    Args:
        model_size: Model size to use
        device: Device preference
        language: Primary language

    Returns:
        Configured Whisper ASR instance
    """
    # Run model loading in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def create_whisper():
        return OptimizedWhisperASR(
            model_size=model_size, device=device, language=language
        )

    return await loop.run_in_executor(None, create_whisper)


def create_optimized_whisper(
    model_size: str = "base", device: str | None = None, language: str = "pl"
) -> OptimizedWhisperASR:
    """Create an optimized Whisper ASR instance.

    Args:
        model_size: Model size to use
        device: Device preference
        language: Primary language

    Returns:
        Configured Whisper ASR instance
    """
    return OptimizedWhisperASR(model_size=model_size, device=device, language=language)


# Compatibility class that matches the interface of the original WhisperASR
class WhisperASR(OptimizedWhisperASR):
    """Compatibility wrapper for existing code."""

    def __init__(self, model_size: str = "base", compute_type: str = "int8"):
        """Initialize with legacy interface.

        Args:
            model_size: Model size
            compute_type: Compute type
        """
        super().__init__(
            model_size=model_size, compute_type=compute_type, language="pl"
        )


# Additional compatibility functions
def create_whisper_asr(config: dict | None = None) -> OptimizedWhisperASR:
    """Create a Whisper ASR instance with configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured Whisper ASR instance
    """
    if config is None:
        config = {}

    model_size = config.get("model", "base")
    device = config.get("device")
    language = config.get("language", "pl")

    return OptimizedWhisperASR(model_size=model_size, device=device, language=language)


def create_audio_recorder(sample_rate: int = 16000, duration: float = 5.0):
    """Legacy-compatible factory function for audio recorder.

    Args:
        sample_rate: Audio sample rate (not used by OptimizedSpeechRecorder)
        duration: Recording duration (not used by OptimizedSpeechRecorder)

    Returns:
        Configured OptimizedSpeechRecorder instance
    """
    from .optimized_wakeword_detector import OptimizedSpeechRecorder

    # Create recorder with only supported parameters
    recorder = OptimizedSpeechRecorder(device_id=None)

    return recorder


# Maintain optimized factory function names as well
create_optimized_recorder = create_audio_recorder
create_optimized_whisper_async = create_whisper_asr
