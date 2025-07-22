import io
import logging
import os

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Fallback performance monitor
try:
    from performance_monitor import measure_performance
except ImportError:

    def measure_performance(func):
        return func


# Attempt to load API key from various sources (environment, env manager, config)


def _load_api_key() -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        import json
        import sys

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.insert(0, parent_dir)
        from server.config_manager import EnvironmentManager

        env_file_path = os.path.join(parent_dir, ".env")
        env_manager = EnvironmentManager(env_file=env_file_path)
        api_key = env_manager.get_api_key("openai")
        if api_key:
            return api_key
    except Exception:
        pass
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "client_config.json"
        )
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            api_key = config.get("api_keys", {}).get("openai")
            if api_key:
                return api_key
    except Exception:
        pass
    try:
        from config import _config

        api_key = _config.get("API_KEYS", {}).get("OPENAI_API_KEY")
        return api_key
    except Exception:
        pass
    return None


class OpenAIASR:
    def __init__(self, model: str = "whisper-1"):
        self.model = model
        self.api_key = _load_api_key()
        if OpenAI is None:
            logger.error("openai library is not available")

    @measure_performance
    def transcribe(self, audio: str | bytes, language: str = "pl") -> str:
        if OpenAI is None:
            logger.error("openai library is not available")
            return ""
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            return ""
        client = OpenAI(api_key=self.api_key)
        # Prepare file-like object
        if isinstance(audio, str):
            file = open(audio, "rb")
            close_file = True
        else:
            file = io.BytesIO(audio)
            close_file = False
        try:
            response = client.audio.transcriptions.create(
                model=self.model, file=file, language=language
            )
            return getattr(response, "text", "")
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return ""
        finally:
            if close_file:
                file.close()


# Factory function for compatibility


def create_openai_asr(config: dict | None = None) -> OpenAIASR:
    model = "whisper-1"
    if config:
        model = config.get("model", model)
    return OpenAIASR(model=model)
