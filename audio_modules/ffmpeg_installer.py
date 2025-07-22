import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile

try:
    import winreg
except ImportError:
    winreg = None

logger = logging.getLogger(__name__)

FFMPEG_INSTALL_DIR = r"C:\ffmpeg"
FFMPEG_BIN_DIR = os.path.join(FFMPEG_INSTALL_DIR, "bin")
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def ensure_ffmpeg_installed():
    """Checks if ffmpeg is available, and if not, downloads a static build, extracts it
    to C:\ffmpeg, adds it to PATH and registers it in the Windows registry."""
    # Check if ffmpeg is on PATH
    if shutil.which("ffmpeg"):
        return
    # Check if already installed in target directory
    if os.path.exists(FFMPEG_BIN_DIR) and shutil.which(
        os.path.join(FFMPEG_BIN_DIR, "ffmpeg")
    ):
        os.environ["PATH"] += os.pathsep + FFMPEG_BIN_DIR
        return
    try:
        logger.info("Downloading FFmpeg...")
        tmp_zip = os.path.join(tempfile.gettempdir(), "ffmpeg.zip")
        urllib.request.urlretrieve(FFMPEG_URL, tmp_zip)
        logger.info(f"Extracting FFmpeg to {FFMPEG_INSTALL_DIR}...")
        with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
            extract_dir = tempfile.mkdtemp()
            zip_ref.extractall(extract_dir)
            # Move extracted folder to install directory
            for name in os.listdir(extract_dir):
                src = os.path.join(extract_dir, name)
                if os.path.isdir(src) and "ffmpeg" in name.lower():
                    if os.path.exists(FFMPEG_INSTALL_DIR):
                        shutil.rmtree(FFMPEG_INSTALL_DIR)
                    shutil.move(src, FFMPEG_INSTALL_DIR)
                    break
        os.remove(tmp_zip)
        # Add to PATH for this process
        os.environ["PATH"] += os.pathsep + FFMPEG_BIN_DIR
        # Add to system PATH in registry
        if winreg:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                    0,
                    winreg.KEY_ALL_ACCESS,
                ) as env_key:
                    path_value, reg_type = winreg.QueryValueEx(env_key, "Path")
                    if FFMPEG_BIN_DIR not in path_value:
                        new_path = path_value + os.pathsep + FFMPEG_BIN_DIR
                        winreg.SetValueEx(env_key, "Path", 0, reg_type, new_path)
                        logger.info("FFmpeg path added to system PATH in registry.")
            except Exception as e:
                logger.warning(f"Failed to update system PATH in registry: {e}")
    except Exception as e:
        logger.error(f"Could not install FFmpeg: {e}")
