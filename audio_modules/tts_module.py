import asyncio
import logging
import os
import subprocess
import threading

try:
    from openai import OpenAI  # type: ignore
except Exception:  # openai not installed
    OpenAI = None

# Try to import performance_monitor, fallback if not available
try:
    from performance_monitor import measure_performance
except ImportError:

    def measure_performance(func):
        """Fallback decorator when performance_monitor is not available."""
        return func


# Handle relative imports
try:
    from .ffmpeg_installer import ensure_ffmpeg_installed
except ImportError:
    try:
        from ffmpeg_installer import ensure_ffmpeg_installed
    except ImportError:

        def ensure_ffmpeg_installed():
            """Fallback function when ffmpeg_installer is not available."""
            pass


# TTS prompt (fallback if prompts module not available)
try:
    from prompts import get_tts_voice_prompt
except ImportError:

    def get_tts_voice_prompt() -> str:
        return "Speak naturally and conversationally."


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


import glob
import time


class TTSModule:
    CLEANUP_INTERVAL = 10  # seconds
    INACTIVITY_THRESHOLD = 30  # seconds

    def __init__(self):
        # Mute flag to disable TTS in text/chat mode
        self.mute = False
        self.current_process = None
        self._last_activity = time.time()
        self._cleanup_task_started = False  # TTS configuration
        self.volume = 200  # ffplay volume (200% for louder audio)
        self.voice = "sage"  # OpenAI voice
        self.model = "gpt-4o-mini-tts"  # OpenAI TTS model (correct model name)
        # Defer cleanup task start to first use to avoid blocking initialization
        # self._start_cleanup_task()

    def _adjust_settings(self) -> None:
        """Adjust volume based on time of day and holidays."""
        from datetime import datetime

        hour = datetime.now().hour
        if 6 <= hour < 12:
            self.volume = 200
        elif hour >= 22 or hour < 6:
            self.volume = 120
        else:
            self.volume = 180
        try:
            from prompts import _holiday_hint

            if _holiday_hint():
                self.volume = min(self.volume + 20, 250)
        except Exception:
            pass

    def _start_cleanup_task(self):
        if not self._cleanup_task_started:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._cleanup_temp_files_loop())
                except RuntimeError:
                    # No running loop, start in a background thread
                    threading.Thread(
                        target=lambda: asyncio.run(self._cleanup_temp_files_loop()),
                        daemon=True,
                    ).start()
                self._cleanup_task_started = True
            except Exception as e:
                logger.error(f"Failed to start TTS cleanup task: {e}")

    async def _cleanup_temp_files_loop(self):
        while True:
            try:
                now = time.time()
                # Only clean up if no TTS activity for INACTIVITY_THRESHOLD
                if now - self._last_activity > self.INACTIVITY_THRESHOLD:
                    pattern = os.path.join("resources", "sounds", "temp_tts_*.mp3")
                    for path in glob.glob(pattern):
                        try:
                            mtime = os.path.getmtime(path)
                            if now - mtime > self.INACTIVITY_THRESHOLD:
                                os.remove(path)
                                logger.info(
                                    f"[TTS Cleanup] Deleted old temp file: {path}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"[TTS Cleanup] Failed to delete {path}: {e}"
                            )
            except Exception as e:
                logger.error(f"[TTS Cleanup] Error in cleanup loop: {e}")
            await asyncio.sleep(self.CLEANUP_INTERVAL)

    def cancel(self):
        if self.current_process:
            try:
                self.current_process.terminate()
            except Exception as e:
                logger.error("Error stopping TTS: %s", e)
            self.current_process = None

    @measure_performance
    async def speak(self, text: str):
        # Start cleanup task on first use
        if not self._cleanup_task_started:
            self._start_cleanup_task()

        self._adjust_settings()

        # Skip speaking if muted (e.g., in text/chat mode)
        if getattr(self, "mute", False):
            return
        logger.info("TTS: %s", text)
        if OpenAI is None:
            logger.error("openai library is not available")
            return

        api_key = os.getenv("OPENAI_API_KEY")

        # Try to load from environment manager if available
        if not api_key:
            try:
                from server.config_manager import EnvironmentManager

                env_file_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), ".env"
                )
                env_manager = EnvironmentManager(env_file=env_file_path)
                api_key = env_manager.get_api_key("openai")
                logger.info(f"Loaded API key from environment manager: {bool(api_key)}")
            except ImportError as e:
                logger.warning(f"Environment manager not available: {e}")
            except Exception as e:
                logger.warning(f"Error loading API key from environment manager: {e}")

        if not api_key:
            try:
                import json

                # Load config from client config file
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "client_config.json"
                )
                if os.path.exists(config_path):
                    with open(config_path, encoding="utf-8") as f:
                        config = json.load(f)
                        api_key = config.get("api_keys", {}).get("openai")
            except Exception as e:
                logger.warning(f"Could not load API key from client config: {e}")

        if not api_key:
            try:
                from config import _config

                api_key = _config.get("API_KEYS", {}).get("OPENAI_API_KEY")
            except Exception:
                api_key = None
        if not api_key:
            logger.error("OpenAI API key not provided")
            return

        client = OpenAI(api_key=api_key)
        self._last_activity = time.time()
        voice_prompt = get_tts_voice_prompt()

        def _stream_and_play() -> None:
            ensure_ffmpeg_installed()
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    response_format="opus",
                ) as response:
                    self.cancel()
                    self.current_process = subprocess.Popen(
                        [
                            "ffplay",
                            "-nodisp",
                            "-autoexit",
                            "-loglevel",
                            "quiet",
                            "-volume",
                            str(self.volume),
                            "-i",
                            "-",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,  # Prevent signal propagation
                    )
                    for chunk in response.iter_bytes():
                        if self.current_process.stdin:
                            try:
                                self.current_process.stdin.write(chunk)
                                self.current_process.stdin.flush()
                            except BrokenPipeError:
                                break
                    if self.current_process.stdin:
                        self.current_process.stdin.close()
                    self.current_process.wait()
            except Exception as e:
                logger.error("TTS error: %s", e)
            finally:
                self.current_process = None

        await asyncio.to_thread(_stream_and_play)


# Create a global instance of TTSModule
_tts_module_instance = TTSModule()


# Define a module-level async speak function
async def speak(text: str):
    """Module-level function to handle text-to-speech."""
    await _tts_module_instance.speak(text)
