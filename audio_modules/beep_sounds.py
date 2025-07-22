import logging
import os
import shutil
import subprocess
import sys

# Global mute flag to disable beeps (e.g., in chat/text mode)
MUTE = False

# Set logger to DEBUG globally
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# Helper function to determine the base path for resources
def get_base_path():
    """Returns the base path for resources, whether running normally or bundled."""
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        # Use executable directory
        return os.path.dirname(sys.executable)
    # Development: 2 levels up from this file
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Słownik mapujący typ dźwięku do ścieżki do pliku
BEEP_SOUNDS = {
    # General-purpose beep (alias of keyword)
    "beep": "resources/sounds/beep.mp3",
    "keyword": "resources/sounds/beep.mp3",
    "search": "resources/sounds/search_beep.mp3",
    "screenshot": "resources/sounds/screenshot_beep.mp3",
    "deep": "resources/sounds/deepthink_beep.mp3",
    "alarm": "resources/sounds/alarm.wav",
    # Added extra beep sounds for wake word detection and VAD events
    "listening_start": "resources/sounds/beep.mp3",
    "listening_done": "resources/sounds/beep.mp3",
    "error": "resources/sounds/beep.mp3",
    "timeout": "resources/sounds/beep.mp3",
}


def play_beep(
    sound_type: str = "keyword", loop: bool = False
) -> subprocess.Popen | None:
    """Odtwarza dźwięk z odpowiedniego pliku na podstawie podanego typu. Zwraca obiekt
    Popen procesu odtwarzania lub None w przypadku błędu.

    :param sound_type: Typ dźwięku do odtworzenia ("keyword", "search", "screenshot",
        "deep", "api"). Domyślnie "keyword".
    :param loop: Czy dźwięk ma być odtwarzany w pętli (True) czy tylko raz (False).
        Domyślnie False (dla pojedynczego odtwarzania).
    :return: Obiekt subprocess.Popen lub None.
    """
    # Validate ffplay availability and safety
    if not _validate_ffplay_available():
        logger.error("ffplay validation failed - cannot play sound")
        return None

    # If muted, skip playing sounds
    if MUTE:
        logger.debug("Audio muted - skipping sound playback")
        return None

    base_dir = get_base_path()
    logger.debug(f"Base directory for beeps: {base_dir}")
    rel_path = BEEP_SOUNDS.get(sound_type)

    if not rel_path:  # If sound_type is not in BEEP_SOUNDS
        logger.debug(
            f"No specific beep sound found for type: {sound_type}. Attempting default."
        )
        rel_path = BEEP_SOUNDS.get("keyword")  # Fallback to default keyword beep

    if not rel_path:  # If still no rel_path (e.g. "keyword" also missing)
        logger.error(
            f"Default beep sound ('keyword') not found in BEEP_SOUNDS. Cannot play sound for type: {sound_type}"
        )
        return None

    # Build absolute path to the sound file
    beep_file = os.path.join(base_dir, rel_path)

    # Validate the audio file path for security
    if not _validate_audio_file_path(beep_file):
        logger.error(f"Audio file validation failed: {beep_file}")
        return None

    try:
        logger.info(
            f"Playing sound '{sound_type}' from file: {beep_file} (Loop: {loop})"
        )

        # Build command with security considerations
        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
        if loop:
            cmd.extend(["-loop", "0"])
        cmd.append(beep_file)

        # Execute with security measures
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,  # Prevent input injection
            start_new_session=True,  # Prevent signal propagation
        )

        logger.debug(f"Started audio playback process (PID: {process.pid})")
        return process

    except FileNotFoundError:
        logger.error(
            f"Error playing sound '{sound_type}': 'ffplay' command not found. Ensure FFmpeg is installed and in PATH."
        )
        return None
    except Exception as e:
        logger.error(f"Error playing sound '{sound_type}': {e}")
        return None


def stop_beep(process: subprocess.Popen):
    """Terminuje proces odtwarzania dźwięku."""
    if process and process.poll() is None:  # Check if process is still running
        try:
            logger.info(f"Zatrzymuję dźwięk (PID: {process.pid})")
            process.terminate()
            process.wait(timeout=1)  # Give it a moment to terminate gracefully
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Proces dźwięku (PID: {process.pid}) nie zakończył się na czas, wymuszam zamknięcie."
            )
            process.kill()
            process.wait(timeout=1)  # Wait after kill
        except Exception as e:
            # Catch potential errors if the process already terminated between poll() and terminate()
            logger.error(
                f"Błąd podczas zatrzymywania dźwięku (PID: {process.pid}): {e}"
            )
    elif process:
        logger.debug(f"Proces dźwięku (PID: {process.pid}) już zakończony.")


def _validate_ffplay_available() -> bool:
    """Validate that ffplay is available and safe to use.

    Returns:
        True if ffplay is available and safe to use, False otherwise
    """
    try:
        # Check if ffplay is available in PATH
        ffplay_path = shutil.which("ffplay")
        if not ffplay_path:
            logger.error("ffplay not found in PATH. Please install FFmpeg.")
            return False
        # Additional security: ensure it's actually ffplay
        try:
            result = subprocess.run(
                [ffplay_path, "-version"], capture_output=True, text=True, timeout=5
            )
            # Check both stdout and stderr for ffplay signature
            output_text = (result.stdout + result.stderr).lower()
            if "ffplay" not in output_text:
                logger.error("ffplay binary validation failed")
                return False
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.error(f"ffplay validation error: {e}")
            return False

        logger.debug(f"ffplay validated: {ffplay_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating ffplay: {e}")
        return False


def _validate_audio_file_path(file_path: str) -> bool:
    """Validate that the audio file path is safe and within expected directories.

    Args:
        file_path: Path to the audio file

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Convert to absolute path and resolve any symlinks
        abs_path = os.path.abspath(file_path)

        # Get the base directory (project root)
        base_dir = get_base_path()

        # Ensure the file is within the project directory
        if not abs_path.startswith(os.path.abspath(base_dir)):
            logger.error(f"Audio file outside project directory: {abs_path}")
            return False

        # Check file extension (only allow audio files)
        allowed_extensions = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
        file_ext = os.path.splitext(abs_path)[1].lower()
        if file_ext not in allowed_extensions:
            logger.error(f"Invalid audio file extension: {file_ext}")
            return False

        # Check file exists and is readable
        if not os.path.isfile(abs_path):
            logger.error(f"Audio file not found: {abs_path}")
            return False

        # Check file size (prevent DoS attacks with huge files)
        file_size = os.path.getsize(abs_path)
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            logger.error(f"Audio file too large: {file_size} bytes")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating audio file path: {e}")
        return False
