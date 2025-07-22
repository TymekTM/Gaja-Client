"""Settings Manager for GAJA Client.

Manages client configuration including audio devices, voice settings, and overlay
options.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import sounddevice as sd

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("sounddevice not available - audio device detection disabled")


class SettingsManager:
    """Manages client settings and configuration."""

    def __init__(self, client_app=None):
        """Initialize settings manager.

        Args:
            client_app: Reference to main ClientApp instance
        """
        self.client_app = client_app
        self.settings_file = Path(__file__).parent.parent / "client_config.json"
        self.default_settings = {
            "server_url": "ws://localhost:8001/ws/client1",
            "user_id": "1",
            "audio": {
                "input_device": None,
                "output_device": None,
                "sample_rate": 16000,
                "record_duration": 5.0,
            },
            "wakeword": {
                "enabled": True,
                "keyword": "gaja",
                "sensitivity": 0.6,
                "device_id": None,
                "stt_silence_threshold_ms": 2000,
            },
            "whisper": {"model": "base", "language": "pl"},
            "overlay": {
                "enabled": True,
                "position": "top-right",
                "opacity": 0.9,
                "auto_hide_delay": 10,
            },
            "voice": {"wake_word": "gaja", "sensitivity": 0.6, "language": "pl-PL"},
            "daily_briefing": {
                "enabled": True,
                "startup_briefing": True,
                "briefing_time": "08:00",
                "location": "Sosnowiec,PL",
            },
        }

    def load_settings(self) -> dict[str, Any]:
        """Load settings from config file.

        Returns:
            Dictionary containing current settings
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, encoding="utf-8") as f:
                    settings = json.load(f)

                # Merge with defaults to ensure all keys exist
                merged_settings = self._merge_settings(self.default_settings, settings)
                return merged_settings
            else:
                logger.info("Settings file not found, using defaults")
                return self.default_settings.copy()

        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return self.default_settings.copy()

    def save_settings(self, settings: dict[str, Any]) -> bool:
        """Save settings to config file.

        Args:
            settings: Dictionary containing settings to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate settings structure
            validated_settings = self._validate_settings(settings)

            # Create backup of existing file
            if self.settings_file.exists():
                backup_path = self.settings_file.with_suffix(".json.backup")
                import shutil

                shutil.copy2(self.settings_file, backup_path)

            # Save new settings
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(validated_settings, f, indent=2, ensure_ascii=False)

            logger.info(f"Settings saved to {self.settings_file}")

            # Apply settings to client if available
            if self.client_app:
                asyncio.create_task(self._apply_settings_to_client(validated_settings))

            return True

        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False

    def get_audio_devices(self) -> dict[str, list[dict[str, Any]]]:
        """Get available audio devices.

        Returns:
            Dictionary with 'input_devices' and 'output_devices' lists
        """
        try:
            if not AUDIO_AVAILABLE:
                return {
                    "input_devices": [
                        {
                            "id": "default",
                            "name": "Domyślne urządzenie wejściowe",
                            "is_default": True,
                        }
                    ],
                    "output_devices": [
                        {
                            "id": "default",
                            "name": "Domyślne urządzenie wyjściowe",
                            "is_default": True,
                        }
                    ],
                }

            devices = sd.query_devices()
            input_devices = []
            output_devices = []

            default_input = sd.default.device[0] if sd.default.device else None
            default_output = sd.default.device[1] if sd.default.device else None

            for i, device in enumerate(devices):
                device_info = {
                    "id": str(i),
                    "name": device["name"],
                    "is_default": False,
                }

                # Check if device has input capabilities
                if device["max_input_channels"] > 0:
                    device_info_input = device_info.copy()
                    device_info_input["is_default"] = i == default_input
                    input_devices.append(device_info_input)

                # Check if device has output capabilities
                if device["max_output_channels"] > 0:
                    device_info_output = device_info.copy()
                    device_info_output["is_default"] = i == default_output
                    output_devices.append(device_info_output)

            return {"input_devices": input_devices, "output_devices": output_devices}

        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return {
                "input_devices": [
                    {
                        "id": "default",
                        "name": "Błąd pobierania urządzeń",
                        "is_default": True,
                    }
                ],
                "output_devices": [
                    {
                        "id": "default",
                        "name": "Błąd pobierania urządzeń",
                        "is_default": True,
                    }
                ],
            }

    def get_connection_status(self) -> dict[str, Any]:
        """Get current connection status.

        Returns:
            Dictionary with connection status information
        """
        try:
            if not self.client_app:
                return {"connected": False, "error": "Client app not available"}

            is_connected = (
                hasattr(self.client_app, "websocket")
                and self.client_app.websocket is not None
            )

            if is_connected:
                server_url = getattr(self.client_app, "server_url", "Unknown")
                port = "8001"  # Default port
                if ":" in server_url:
                    port = server_url.split(":")[-1].split("/")[0]

                return {
                    "connected": True,
                    "port": port,
                    "server_status": {
                        "status": self.client_app.current_status,
                        "monitoring": getattr(
                            self.client_app, "monitoring_wakeword", False
                        ),
                        "overlay_visible": getattr(
                            self.client_app, "overlay_visible", False
                        ),
                    },
                }
            else:
                return {"connected": False, "error": "Nie połączono z serwerem"}

        except Exception as e:
            logger.error(f"Error getting connection status: {e}")
            return {"connected": False, "error": f"Błąd: {str(e)}"}

    def _merge_settings(
        self, defaults: dict[str, Any], user_settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user settings with defaults.

        Args:
            defaults: Default settings dictionary
            user_settings: User's settings dictionary

        Returns:
            Merged settings dictionary
        """
        merged = defaults.copy()

        for key, value in user_settings.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._merge_settings(merged[key], value)
                else:
                    merged[key] = value
            else:
                merged[key] = value

        return merged

    def _validate_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Validate settings structure and values.

        Args:
            settings: Settings dictionary to validate

        Returns:
            Validated settings dictionary
        """
        validated = self.default_settings.copy()

        try:
            # Audio settings
            if "audio" in settings:
                audio = settings["audio"]
                if isinstance(audio, dict):
                    if "input_device" in audio:
                        validated["audio"]["input_device"] = audio["input_device"]
                    if "output_device" in audio:
                        validated["audio"]["output_device"] = audio["output_device"]
                    if "sample_rate" in audio and isinstance(
                        audio["sample_rate"], (int, float)
                    ):
                        validated["audio"]["sample_rate"] = max(
                            8000, min(48000, int(audio["sample_rate"]))
                        )
                    if "record_duration" in audio and isinstance(
                        audio["record_duration"], (int, float)
                    ):
                        validated["audio"]["record_duration"] = max(
                            1.0, min(30.0, float(audio["record_duration"]))
                        )

            # Voice settings
            if "voice" in settings:
                voice = settings["voice"]
                if isinstance(voice, dict):
                    if "wake_word" in voice and isinstance(voice["wake_word"], str):
                        validated["voice"]["wake_word"] = voice["wake_word"]
                        # Also update wakeword config
                        validated["wakeword"]["keyword"] = voice["wake_word"]
                    if "sensitivity" in voice and isinstance(
                        voice["sensitivity"], (int, float)
                    ):
                        sensitivity = max(0.1, min(1.0, float(voice["sensitivity"])))
                        validated["voice"]["sensitivity"] = sensitivity
                        validated["wakeword"]["sensitivity"] = sensitivity
                    if "language" in voice and isinstance(voice["language"], str):
                        validated["voice"]["language"] = voice["language"]
                        # Update whisper language too
                        lang_code = voice["language"].split("-")[0]
                        validated["whisper"]["language"] = lang_code

            # Overlay settings
            if "overlay" in settings:
                overlay = settings["overlay"]
                if isinstance(overlay, dict):
                    if "enabled" in overlay and isinstance(overlay["enabled"], bool):
                        validated["overlay"]["enabled"] = overlay["enabled"]
                    if "position" in overlay and isinstance(overlay["position"], str):
                        valid_positions = [
                            "top-right",
                            "top-left",
                            "bottom-right",
                            "bottom-left",
                            "center",
                        ]
                        if overlay["position"] in valid_positions:
                            validated["overlay"]["position"] = overlay["position"]
                    if "opacity" in overlay and isinstance(
                        overlay["opacity"], (int, float)
                    ):
                        validated["overlay"]["opacity"] = max(
                            0.1, min(1.0, float(overlay["opacity"]))
                        )

            # Daily briefing settings
            if "daily_briefing" in settings:
                briefing = settings["daily_briefing"]
                if isinstance(briefing, dict):
                    if "enabled" in briefing and isinstance(briefing["enabled"], bool):
                        validated["daily_briefing"]["enabled"] = briefing["enabled"]
                    if "startup_briefing" in briefing and isinstance(
                        briefing["startup_briefing"], bool
                    ):
                        validated["daily_briefing"]["startup_briefing"] = briefing[
                            "startup_briefing"
                        ]
                    if "briefing_time" in briefing and isinstance(
                        briefing["briefing_time"], str
                    ):
                        validated["daily_briefing"]["briefing_time"] = briefing[
                            "briefing_time"
                        ]
                    if "location" in briefing and isinstance(briefing["location"], str):
                        validated["daily_briefing"]["location"] = briefing["location"]

            return validated

        except Exception as e:
            logger.error(f"Error validating settings: {e}")
            return validated

    async def _apply_settings_to_client(self, settings: dict[str, Any]):
        """Apply settings to client components.

        Args:
            settings: Settings dictionary to apply
        """
        try:
            if not self.client_app:
                return

            # Update client config
            self.client_app.config = settings

            # Update audio devices if changed
            if "audio" in settings:
                audio_config = settings["audio"]

                # Update audio recorder if available
                if (
                    hasattr(self.client_app, "audio_recorder")
                    and self.client_app.audio_recorder
                ):
                    if "sample_rate" in audio_config:
                        self.client_app.audio_recorder.sample_rate = audio_config[
                            "sample_rate"
                        ]
                    if "record_duration" in audio_config:
                        self.client_app.audio_recorder.duration = audio_config[
                            "record_duration"
                        ]

                # Update wakeword detector device
                if (
                    hasattr(self.client_app, "wakeword_detector")
                    and self.client_app.wakeword_detector
                ):
                    if "input_device" in audio_config and audio_config["input_device"]:
                        # Apply device setting to wakeword detector
                        if hasattr(self.client_app.wakeword_detector, "set_device"):
                            self.client_app.wakeword_detector.set_device(
                                audio_config["input_device"]
                            )

            # Update wakeword settings
            if "wakeword" in settings:
                wakeword_config = settings["wakeword"]
                if (
                    hasattr(self.client_app, "wakeword_detector")
                    and self.client_app.wakeword_detector
                ):
                    if hasattr(self.client_app.wakeword_detector, "update_config"):
                        self.client_app.wakeword_detector.update_config(wakeword_config)

            # Update overlay settings
            if "overlay" in settings:
                overlay_config = settings["overlay"]
                if not overlay_config.get("enabled", True):
                    await self.client_app.hide_overlay()

            logger.info("Settings applied to client successfully")

        except Exception as e:
            logger.error(f"Error applying settings to client: {e}")

    def reset_to_defaults(self) -> bool:
        """Reset settings to default values.

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.save_settings(self.default_settings.copy())
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            return False

    def export_settings(self, export_path: Path) -> bool:
        """Export current settings to file.

        Args:
            export_path: Path to export settings to

        Returns:
            True if successful, False otherwise
        """
        try:
            current_settings = self.load_settings()

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(current_settings, f, indent=2, ensure_ascii=False)

            logger.info(f"Settings exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            return False

    def import_settings(self, import_path: Path) -> bool:
        """Import settings from file.

        Args:
            import_path: Path to import settings from

        Returns:
            True if successful, False otherwise
        """
        try:
            if not import_path.exists():
                logger.error(f"Settings file not found: {import_path}")
                return False

            with open(import_path, encoding="utf-8") as f:
                imported_settings = json.load(f)

            return self.save_settings(imported_settings)

        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            return False
