"""GAJA Assistant Client Klient obsługujący lokalne komponenty (wakeword, overlay,
Whisper ASR) i komunikujący się z serwerem przez WebSocket."""

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import websockets
from loguru import logger

# Dodaj ścieżkę główną projektu do PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from active_window_module import get_active_window_title

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")
except ImportError:
    logger.warning("python-dotenv not installed, skipping .env loading")
    # Try to load manually
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        logger.info(f"Loading .env manually from {env_path}")
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        logger.info("Environment variables loaded manually")
    else:
        logger.warning(f"No .env file found at {env_path}")

# Import local modules - lazy loading after dependency check
# Audio modules will be imported dynamically after dependency installation

# Import basic modules first
try:
    from active_window_module import get_active_window_info
except ImportError:
    logger.warning("Active window module not available")
    get_active_window_info = None

# Import enhanced modules if available, fallback to lazy loading
TTSModule = None
WhisperASR = None
create_whisper_asr = None
create_audio_recorder = None
create_wakeword_detector = None

# Import user mode system if available
try:
    from src.gaja_core.mode_integrator import user_integrator

    USER_MODE_AVAILABLE = True
    logger.info("User Mode System available")
except ImportError:
    USER_MODE_AVAILABLE = False
    user_integrator = None
    logger.info("User Mode System not available - using legacy mode")

# Import system tray manager
try:
    from modules.tray_manager import TrayManager

    TRAY_AVAILABLE = True
except ImportError:
    logger.warning("Tray manager not available")
    TrayManager = None
    TRAY_AVAILABLE = False

# Import settings manager
try:
    from modules.settings_manager import SettingsManager

    SETTINGS_AVAILABLE = True
except ImportError:
    logger.warning("Settings manager not available")
    SettingsManager = None
    SETTINGS_AVAILABLE = False


class ClientApp:
    """Główna klasa klienta GAJA."""

    def __init__(self):
        self.config = self.load_client_config()
        self.websocket = None
        self.user_id = self.config.get("user_id", "1")
        # server_url should be set during load_client_config
        if not hasattr(self, "server_url") or self.server_url is None:
            self.server_url = self.config.get(
                "server_url", "ws://localhost:8001/ws/client1"
            )
            logger.info(f"🔗 Server URL fallback set to: {self.server_url}")
        else:
            logger.info(f"🔗 Server URL already set to: {self.server_url}")
        self.running = False

        # Initialize settings manager
        self.settings_manager = None
        if SETTINGS_AVAILABLE:
            self.settings_manager = SettingsManager(client_app=self)

        # Audio components
        self.wakeword_detector = None
        self.whisper_asr = None
        self.audio_recorder = None
        self.tts = None
        self.overlay_process = None  # External Tauri overlay process
        # State management
        self.monitoring_wakeword = False  # Whether wakeword monitoring is active
        self.wake_word_detected = False  # Whether wakeword was just detected
        self.recording_command = False
        self.tts_playing = False
        self.last_tts_text = ""
        self.current_status = "Starting..."
        # HTTP server for overlay
        self.http_server = None
        self.http_thread = None
        # WebSocket server for overlay (lepsze niż HTTP polling)
        self.websocket_server = None
        self.websocket_clients = set()  # Połączeni klienci overlay
        self.command_queue = queue.Queue()  # Queue for HTTP commands
        # Update overlay status data for showing/hiding
        self.overlay_visible = False
        self.sse_clients = []

        # Message tracking for system tray
        self.message_limit = None
        self.daily_message_count = 0

        # System tray manager
        self.tray_manager = None
        if TRAY_AVAILABLE:
            self.tray_manager = TrayManager(client_app=self)

    def load_client_config(self) -> dict:
        """Załaduj konfigurację klienta."""
        # Config file should be in the same directory as this script
        config_path = Path(__file__).parent / "client_config.json"

        logger.info(f"🔧 Looking for config at: {config_path.absolute()}")

        if config_path.exists():
            logger.info("📁 Config file found, loading...")
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                self.server_url = config.get("server_url")  # Store server URL
                logger.info(f"✅ Config loaded: server_url={config.get('server_url')}")
                logger.info(f"🔗 Server URL set to: {self.server_url}")
                return config

        logger.warning("⚠️ Config file not found, using defaults")
        # Default config
        default_config = {
            "server_url": "ws://localhost:8001/ws/client1",
            "user_id": "1",
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
                "auto_hide_delay": 10,
            },
            "audio": {"sample_rate": 16000, "record_duration": 5.0},
        }
        self.server_url = default_config["server_url"]  # Store default server URL
        return default_config

    async def initialize_components(self):
        """Inicjalizuj komponenty audio i overlay."""
        try:
            # Start HTTP server for overlay first
            self.start_http_server()
            # Start WebSocket server for overlay (better than HTTP polling)
            await self.start_websocket_server()
            self.update_status("Initializing...")

            # Initialize external Tauri overlay only
            if self.config.get("overlay", {}).get("enabled", True):
                await self.start_overlay()
            # Initialize wakeword detector
            wakeword_config = self.config.get("wakeword", {})

            # First, load audio modules lazily (after dependencies are installed)
            audio_available = await self._load_audio_modules()
            if not audio_available:
                logger.warning("Audio modules not available - running in limited mode")

            if wakeword_config.get("enabled", True) and create_wakeword_detector:
                self.wakeword_detector = create_wakeword_detector(
                    config=wakeword_config, callback=self.on_wakeword_detected
                )
                logger.info("Wakeword detector initialized")

            # Initialize Whisper ASR - Enhanced or Legacy
            whisper_config = self.config.get("whisper", {})
            if USER_MODE_AVAILABLE and hasattr(user_integrator, "asr_module"):
                # ASR module provided by user_integrator
                self.whisper_asr = user_integrator.asr_module
                logger.info("ASR module initialized via User Mode Integrator")
            elif create_whisper_asr:
                self.whisper_asr = create_whisper_asr(whisper_config)
                logger.info("Legacy Whisper ASR initialized")

            # Always create audio recorder for wakeword detection
            if create_audio_recorder:
                self.audio_recorder = create_audio_recorder(
                    sample_rate=self.config.get("audio", {}).get("sample_rate", 16000),
                    duration=self.config.get("audio", {}).get("record_duration", 5.0),
                )

            # Initialize TTS - Enhanced or Legacy with error handling
            try:
                if USER_MODE_AVAILABLE and hasattr(user_integrator, "tts_module"):
                    # Enhanced TTS will be managed by user_integrator
                    self.tts = user_integrator.tts_module
                    logger.info(
                        "Enhanced TTS module initialized via User Mode Integrator"
                    )
                else:
                    # Try to import and use TTS module
                    try:
                        from audio_modules.tts_module import TTSModule

                        self.tts = TTSModule()
                        logger.info("Legacy TTS module initialized")

                        # Test TTS initialization
                        if hasattr(self.tts, "_adjust_settings"):
                            self.tts._adjust_settings()
                            logger.info("TTS settings adjusted")
                    except ImportError as e:
                        logger.warning(f"TTS module import failed: {e}")
                        self.tts = None
                    except Exception as e:
                        logger.error(f"TTS module creation failed: {e}")
                        self.tts = None

            except Exception as e:
                logger.error(f"Error initializing TTS module: {e}")
                # Try to create a fallback TTS module
                try:
                    from audio_modules.tts_module import TTSModule

                    self.tts = TTSModule()
                    logger.info("Fallback TTS module initialized")
                except Exception as e2:
                    logger.error(f"Fallback TTS initialization failed: {e2}")
                    self.tts = None
            # Set whisper ASR for wakeword detector
            if self.wakeword_detector:
                self.wakeword_detector.set_whisper_asr(self.whisper_asr)
                logger.info("Whisper ASR set for wakeword detector")

            logger.info("All client components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    async def start_overlay(self):
        """Uruchom zewnętrzny overlay Tauri."""
        try:
            # Możliwe ścieżki do overlay exe - najpierw sprawdzamy nowy naprawiony overlay
            overlay_paths = [
                # Nowy naprawiony overlay z katalogu głównego (release) - NAJWYZSZY PRIORYTET
                Path(__file__).parent.parent
                / "overlay"
                / "target"
                / "release"
                / "gaja-overlay.exe",
                # Nowy naprawiony overlay z katalogu głównego (debug)
                Path(__file__).parent.parent
                / "overlay"
                / "target"
                / "debug"
                / "gaja-overlay.exe",
                # Fallback do starych lokalizacji
                Path(__file__).parent
                / "overlay"
                / "target"
                / "release"
                / "gaja-overlay.exe",
                Path(__file__).parent / "overlay" / "gaja-overlay.exe",
            ]

            overlay_path = None
            for path in overlay_paths:
                if path.exists():
                    overlay_path = path
                    break

            if overlay_path:
                # Start overlay process
                self.overlay_process = subprocess.Popen(
                    [str(overlay_path)],
                    cwd=overlay_path.parent.parent,
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
                        else 0
                    ),
                )
                logger.info(
                    f"Started Tauri overlay process: {self.overlay_process.pid} from {overlay_path}"
                )

                # Wait a moment for overlay to start
                await asyncio.sleep(1)

            else:
                logger.warning("Overlay executable not found in any expected location")
                logger.info("Checked paths:")
                for path in overlay_paths:
                    logger.info(f"  - {path}")

        except Exception as e:
            logger.error(f"Error starting overlay: {e}")

    async def connect_to_server(self):
        """Nawiąż połączenie z serwerem."""
        try:
            logger.info(f"Attempting to connect to: {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"Connected to server: {self.server_url}")

            # Sprawdź czy to pierwszy start dnia i poproś o briefing
            await self.request_startup_briefing()

        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            logger.error(f"Server URL was: {self.server_url}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def request_startup_briefing(self):
        """Poproś serwer o briefing startowy."""
        try:
            logger.info("Requesting startup briefing...")
            await self.send_message({"type": "startup_briefing"})
        except Exception as e:
            logger.error(f"Error requesting startup briefing: {e}")

    async def handle_summary_response(self, data: dict):
        """Obsłuż odpowiedź z podsumowaniem."""
        try:
            message_type = data.get("type")
            summary_data = data.get("data", {})

            logger.info(f"Summary response received: {message_type}")

            # Update overlay with summary type
            self.update_status(f"Summary: {message_type}")
            await self.show_overlay()

            # Generate summary text for TTS
            summary_text = self._format_summary_for_speech(message_type, summary_data)

            if summary_text and self.tts:
                try:
                    self.tts_playing = True
                    await self.tts.speak(summary_text)
                    logger.info(f"Summary spoken: {message_type}")
                except Exception as e:
                    logger.error(f"Error speaking summary: {e}")
                finally:
                    self.tts_playing = False
                    await self.hide_overlay()
                    self.update_status("Listening...")
            else:
                logger.info("No summary text to speak or TTS not available")
                await self.hide_overlay()
                self.update_status("Listening...")

        except Exception as e:
            logger.error(f"Error handling summary response: {e}")

    def _format_summary_for_speech(self, summary_type: str, summary_data: dict) -> str:
        """Sformatuj podsumowanie do mowy."""
        try:
            if summary_type == "day_summary":
                if summary_data.get("success"):
                    stats = summary_data.get("statistics", {})
                    active_time = stats.get("total_active_time_hours", 0)
                    interactions = stats.get("total_interactions", 0)
                    productivity = stats.get("productivity_score", 0)

                    return (
                        f"Podsumowanie dnia: pracowałeś {active_time:.1f} godzin, "
                        f"miałeś {interactions} interakcji, "
                        f"produktywność {productivity:.0%}."
                    )

            elif summary_type == "week_summary":
                if summary_data.get("success"):
                    total_stats = summary_data.get("total_statistics", {})
                    total_time = total_stats.get("total_active_time", 0)
                    avg_productivity = total_stats.get("average_productivity", 0)

                    return (
                        f"Podsumowanie tygodnia: łącznie {total_time:.1f} godzin pracy, "
                        f"średnia produktywność {avg_productivity:.0%}."
                    )

            elif summary_type == "day_narrative":
                if summary_data.get("success"):
                    narrative = summary_data.get("narrative", "")
                    return narrative

            elif summary_type == "behavior_insights":
                if summary_data.get("success"):
                    recommendations = summary_data.get("insights", {}).get(
                        "recommendations", []
                    )
                    if recommendations:
                        return f"Wglądy w zachowania: {recommendations[0]}"
                    else:
                        return "Analiza zachowań została zakończona."

            elif summary_type == "routine_insights":
                if summary_data.get("success"):
                    recommendations = summary_data.get("insights", {}).get(
                        "recommendations", []
                    )
                    if recommendations:
                        return f"Analiza rutyn: {recommendations[0]}"
                    else:
                        return "Analiza rutyn została zakończona."

            return f"Otrzymano {summary_type}, ale nie mogę go odczytać."

        except Exception as e:
            logger.error(f"Error formatting summary for speech: {e}")
            return "Wystąpił błąd podczas formatowania podsumowania."

    async def request_day_summary(
        self, summary_type: str = "day", date: str = None, style: str = "friendly"
    ):
        """Poproś serwer o podsumowanie dnia."""
        try:
            logger.info(f"Requesting day summary: type={summary_type}")
            await self.send_message(
                {
                    "type": "day_summary",
                    "summary_type": summary_type,
                    "date": date,
                    "style": style,
                }
            )
        except Exception as e:
            logger.error(f"Error requesting day summary: {e}")

    async def send_message(self, message: dict):
        """Wyślij wiadomość do serwera."""
        if self.websocket:
            try:
                logger.info(f"Sending message to server: {message}")
                await self.websocket.send(json.dumps(message))
                logger.debug(f"Sent message: {message['type']}")
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    async def listen_for_messages(self):
        """Nasłuchuj wiadomości od serwera."""
        try:
            while self.running:
                if self.websocket:
                    try:
                        message = await self.websocket.recv()
                        data = json.loads(message)
                        await self.handle_server_message(data)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection to server lost")
                        break
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from server: {e}")
                    except asyncio.CancelledError:
                        logger.info("WebSocket listener cancelled")
                        break
                else:
                    try:
                        await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        logger.info("WebSocket listener cancelled")
                        break
        except asyncio.CancelledError:
            logger.info("WebSocket listener cancelled")
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")

    async def handle_server_message(self, data: dict):
        """Obsłuż wiadomość od serwera."""
        message_type = data.get("type")

        # Debug: log the entire message structure
        logger.debug(
            f"Received message from server: type={message_type}, data keys={list(data.keys())}"
        )
        if message_type == "ai_response":
            logger.info(
                f"AI response data structure: {data}"
            )  # Change to INFO for visibility

        # Track message limits and counts if provided
        if "message_limit" in data:
            self.message_limit = data["message_limit"]
        if "daily_message_count" in data:
            self.daily_message_count = data["daily_message_count"]

        if message_type == "daily_briefing":
            # Obsłuż daily briefing
            briefing_text = data.get("text", "")
            logger.info(f"Daily briefing received: {briefing_text[:100]}...")

            if briefing_text:
                # Update overlay with briefing
                self.update_status("Daily Briefing")
                await self.show_overlay()

                # Speak the briefing - ensure TTS is properly initialized
                if self.tts:
                    try:
                        self.tts_playing = True
                        logger.info("Starting TTS for daily briefing...")
                        await self.tts.speak(briefing_text)
                        logger.info("Daily briefing spoken successfully")
                    except Exception as e:
                        logger.error(f"Error speaking daily briefing: {e}")
                        # Try to reinitialize TTS if it failed
                        try:
                            if TTSModule:
                                self.tts = TTSModule()
                                await self.tts.speak(briefing_text)
                                logger.info("Daily briefing spoken after TTS reinit")
                        except Exception as e2:
                            logger.error(f"Failed to reinitialize TTS: {e2}")
                    finally:
                        self.tts_playing = False
                        await self.hide_overlay()
                        self.update_status("Listening...")
                else:
                    logger.warning("TTS not available for daily briefing")
                    # Show briefing text in overlay for longer if TTS not available
                    await asyncio.sleep(5)
                    await self.hide_overlay()
                    self.update_status("Listening...")

        elif message_type == "briefing_skipped":
            logger.info("Daily briefing skipped - already delivered today")
            self.update_status("Ready - Briefing skipped")

        elif message_type == "proactive_notifications":
            # Obsłuż proaktywne powiadomienia
            await self.handle_proactive_notifications(data)

        elif message_type == "startup_briefing":
            # Obsłuż briefing startowy
            briefing = data.get("briefing", {})
            await self.handle_startup_briefing(briefing)

        elif message_type == "day_summary":
            # Obsłuż podsumowanie dnia
            summary = data.get("summary", {})
            await self.handle_day_summary(summary)

        elif message_type == "ai_response":
            # Extract response from the data structure
            message_data = data.get("data", {})
            response = message_data.get("response", "")
            logger.info(f"AI Response received: {response}")

            # Check if response is empty or None
            if not response:
                logger.warning("Received empty AI response")
                self.wake_word_detected = False
                self.recording_command = False
                await self.hide_overlay()
                self.update_status("słucham")
                return

            # Response is already a JSON string from the server
            try:
                if isinstance(response, str):
                    response_data = json.loads(response)
                else:
                    response_data = response

                text = response_data.get("text", "")
                if text:
                    logger.info(f"AI text response: {text}")

                    # Update overlay with AI response - set text BEFORE showing overlay
                    self.last_tts_text = text
                    self.update_status("mówię")

                    # Show overlay immediately when starting TTS
                    await self.show_overlay()

                    # Play TTS response
                    if self.tts:
                        try:
                            self.tts_playing = True
                            self.update_status("mówię")
                            # Ensure text is set during TTS playback
                            self.last_tts_text = text
                            await self.tts.speak(text)
                            logger.info("TTS response played")
                        except Exception as tts_e:
                            logger.error(f"TTS error: {tts_e}")
                        finally:
                            self.tts_playing = False
                            self.wake_word_detected = (
                                False  # Reset wakeword flag after speaking
                            )
                            self.recording_command = False  # Reset recording flag
                            # Hide overlay after speaking
                            await self.hide_overlay()
                            self.update_status(
                                "słucham"
                            )  # Return to listening immediately
                    else:
                        logger.warning("TTS not available")
                        self.wake_word_detected = False
                        self.update_status(
                            "słucham"
                        )  # Return to listening even without TTS
                else:
                    logger.warning("No text in AI response")

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(
                    f"Error parsing AI response: {e}"
                )  # Fallback: treat response as plain text
                if isinstance(response, str) and response.strip():
                    text = response.strip()
                    logger.info(f"Using response as plain text: {text}")

                    # Update overlay - set text BEFORE showing overlay
                    self.last_tts_text = text
                    self.update_status("mówię")

                    # Show overlay immediately when starting TTS
                    await self.show_overlay()

                    # Play TTS
                    if self.tts:
                        try:
                            self.tts_playing = True
                            self.update_status("mówię")
                            # Ensure text is set during TTS playback
                            self.last_tts_text = text
                            await self.tts.speak(text)
                            logger.info("TTS response played (plain text)")
                        except Exception as tts_e:
                            logger.error(f"TTS error: {tts_e}")
                        finally:
                            self.tts_playing = False
                            self.wake_word_detected = (
                                False  # Reset wakeword flag after speaking (fallback)
                            )
                            self.recording_command = False  # Reset recording flag
                            # Hide overlay after speaking
                            await self.hide_overlay()
                            self.update_status(
                                "słucham"
                            )  # Return to listening immediately
                    else:
                        logger.warning("TTS not available")
                        self.wake_word_detected = False
                        self.update_status(
                            "słucham"
                        )  # Return to listening even without TTS

        elif message_type == "function_result":
            function_name = data.get("function")
            result = data.get("result")
            logger.info(f"Function {function_name} result: {result}")

        elif message_type == "plugin_toggled":
            plugin = data.get("plugin")
            status = data.get("status")
            logger.info(f"Plugin {plugin} {status}")

        elif message_type == "plugin_status_updated":
            plugins = data.get("plugins", {})
            logger.info(f"Plugin status update: {plugins}")

        elif message_type == "error":
            error = data.get("error", "Unknown error")
            logger.error(f"Server error: {error}")
            # DON'T update status to "Error" as it causes overlay to show
            # Just log the error silently

        elif message_type == "handshake_response":
            # Handle handshake confirmation from server
            handshake_data = data.get("data", {})
            if handshake_data.get("success", False):
                logger.info("Handshake successful with server")
            else:
                logger.warning("Handshake failed with server")

        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def on_wakeword_detected(self, query: str = None):
        """Callback wywoływany po wykryciu słowa aktywującego i transkrypcji."""
        if query:
            # We already have transcribed text from wakeword detector
            logger.info(f"Wakeword detected with query: {query}")

            # IMMEDIATE RESPONSE - Set wake word detection flag and show overlay INSTANTLY
            self.wake_word_detected = True
            self.update_status("myślę")

            # Show overlay IMMEDIATELY when wake word is detected - BEFORE any processing
            await self.show_overlay()

            if self.recording_command:
                logger.warning("Already processing command")
                return

            try:
                self.recording_command = True
                self.update_status("Przetwarzam zapytanie...")

                # If server is available, send query
                if self.websocket:
                    active_title = get_active_window_title()
                    message = {
                        "type": "query",
                        "query": query,
                        "context": {
                            "source": "voice",
                            "user_name": "Voice User",
                            "active_window_title": active_title,
                            "track_active_window_setting": True,
                        },
                    }
                    await self.send_message(message)
                else:
                    # Standalone mode - simulate processing and response
                    logger.info("Standalone mode - simulating processing")
                    await asyncio.sleep(1)  # Simulate thinking time

                    # Show that we heard the command but can't process it
                    response_text = (
                        f"Wykryto polecenie: '{query}'. Serwer AI niedostępny."
                    )
                    self.last_tts_text = response_text
                    self.update_status("mówię")

                    # Simulate TTS
                    self.tts_playing = True
                    self.update_status("mówię")
                    await asyncio.sleep(2)  # Simulate speech time
                    self.tts_playing = False
                    self.update_status("Ready")

                    # Reset flags and hide overlay
                    self.wake_word_detected = False
                    self.recording_command = False
                    await self.hide_overlay()

            except Exception as e:
                logger.error(f"Error processing voice command: {e}")
                self.update_status("Error")
                # Reset flags on error
                self.wake_word_detected = False
                self.recording_command = False
                # Return to listening state
                await asyncio.sleep(1)
                self.update_status("słucham")
            finally:
                # Note: recording_command and wake_word_detected are reset in TTS completion
                pass
        else:
            # Legacy support - wakeword detected without transcription
            logger.info(
                "Wakeword detected! Recording and transcription handled by wakeword detector."
            )
            self.wake_word_detected = True
            self.update_status("słucham")

    async def update_overlay_status(self, status: str):
        """Zaktualizuj status w overlay."""
        logger.debug(f"Overlay status: {status}")
        # Overlay is external process - status updates would need IPC

    def get_current_status(self) -> str:
        """Zwróć aktualny status klienta."""
        return self.current_status

    def update_status(self, status: str):
        """Zaktualizuj status i powiadom overlay."""
        self.current_status = status
        self.notify_sse_clients()

        # Wyślij status do overlay przez WebSocket (asynchronicznie)
        if self.websocket_clients:
            message = {
                "type": "status",
                "data": {
                    "status": self.current_status,
                    "text": self.last_tts_text,
                    "is_listening": self.recording_command,
                    "is_speaking": self.tts_playing,
                    "wake_word_detected": self.wake_word_detected,
                    "overlay_visible": self.overlay_visible,
                    "monitoring": self.monitoring_wakeword,
                },
            }
            # Uruchom broadcast w tle (nie blokuj UI)
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.broadcast_to_overlay(message))
                else:
                    asyncio.run(self.broadcast_to_overlay(message))
            except Exception as e:
                logger.debug(f"Could not broadcast to overlay: {e}")

        # Update system tray
        if self.tray_manager:
            self.tray_manager.update_status(status)

    def add_sse_client(self, client):
        """Dodaj klienta SSE."""
        self.sse_clients.append(client)

    def remove_sse_client(self, client):
        """Usuń klienta SSE."""
        if client in self.sse_clients:
            self.sse_clients.remove(client)

    def notify_sse_clients(self):
        """Powiadom wszystkich klientów SSE o zmianie statusu."""
        # Determine if we should show content based on current state
        has_meaningful_text = self.last_tts_text and self.last_tts_text not in [
            "",
            "Listening...",
            "Ready",
            "Offline",
        ]

        should_show_content = (
            self.wake_word_detected
            or self.tts_playing
            or self.recording_command
            or has_meaningful_text
        )

        # When just listening (no wake word detected), show only the orb
        show_just_orb = (
            not self.wake_word_detected
            and not self.tts_playing
            and not self.recording_command
            and not has_meaningful_text
        )

        # IMMEDIATE STATUS UPDATES - no debouncing for critical states
        is_critical_state = (
            self.wake_word_detected
            or self.current_status
            in ["Przetwarzam...", "Mówię...", "Przetwarzam zapytanie..."]
            or self.tts_playing
            or self.recording_command
        )

        status_data = {
            "status": self.current_status,
            "text": self.last_tts_text if self.last_tts_text else self.current_status,
            "is_listening": not self.recording_command
            and not self.tts_playing
            and not self.wake_word_detected,
            "is_speaking": self.tts_playing,
            "wake_word_detected": self.wake_word_detected,
            "overlay_visible": self.overlay_visible,
            "show_content": should_show_content,
            "show_just_orb": show_just_orb,
            "overlay_enabled": True,  # Always enabled unless explicitly disabled
            # Add timing information to help with overlay display
            "timestamp": time.time(),
            "critical": is_critical_state,  # Flag for immediate processing
        }

        message = f"data: {json.dumps(status_data)}\n\n"

        # Send to all connected SSE clients with IMMEDIATE delivery for critical states
        for client in self.sse_clients[
            :
        ]:  # Copy list to avoid modification during iteration
            try:
                client.wfile.write(message.encode())
                client.wfile.flush()
                # Force immediate flush for critical states
                if is_critical_state:
                    import socket

                    try:
                        client.wfile._sock.setsockopt(
                            socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
                        )
                    except:
                        pass  # Ignore socket option errors
            except Exception as e:
                logger.debug(f"SSE client disconnected: {e}")
                self.remove_sse_client(client)

    def start_http_server(self):
        """Uruchom HTTP serwer dla overlay."""
        try:
            # Try port 5001 first (debug), then 5000 (release)
            ports = [5001, 5000]

            for port in ports:
                try:
                    # Create handler with reference to client app
                    def handler(*args, **kwargs):
                        return StatusHTTPHandler(self, *args, **kwargs)

                    self.http_server = HTTPServer(("127.0.0.1", port), handler)

                    # Start server in separate thread
                    self.http_thread = threading.Thread(
                        target=self.http_server.serve_forever, daemon=True
                    )
                    self.http_thread.start()

                    logger.info(f"HTTP server started on port {port} for overlay")
                    break

                except OSError as e:
                    logger.warning(f"Port {port} unavailable: {e}")
                    continue
            else:
                logger.error("Could not start HTTP server on any port")

        except Exception as e:
            logger.error(f"Error starting HTTP server: {e}")

    async def start_websocket_server(self):
        """Uruchom WebSocket serwer dla overlay."""
        try:
            import websockets

            async def handle_overlay_connection(websocket, path):
                """Obsługa połączenia WebSocket z overlay."""
                logger.info(
                    f"Overlay connected via WebSocket: {websocket.remote_address}"
                )
                self.websocket_clients.add(websocket)

                try:
                    # Wyślij początkowy status
                    initial_status = {
                        "type": "status",
                        "data": {
                            "status": self.current_status,
                            "text": self.last_tts_text,
                            "is_listening": self.recording_command,
                            "is_speaking": self.tts_playing,
                            "wake_word_detected": self.wake_word_detected,
                            "overlay_visible": self.overlay_visible,
                            "monitoring": self.monitoring_wakeword,
                        },
                    }
                    await websocket.send(json.dumps(initial_status))

                    # Słuchaj wiadomości od overlay
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            logger.debug(f"Received from overlay: {data}")

                            # Obsługuj komendy od overlay
                            if data.get("type") == "command":
                                command = data.get("command")
                                if command == "toggle_monitoring":
                                    await self.toggle_wakeword_monitoring()
                                elif command == "stop_tts":
                                    await self.stop_tts()

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from overlay: {message}")

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Overlay WebSocket disconnected")
                except Exception as e:
                    logger.error(f"Error in overlay WebSocket handler: {e}")
                finally:
                    self.websocket_clients.discard(websocket)

            # Próbuj porty WebSocket (6001, 6000)
            ports = [6001, 6000]

            for port in ports:
                try:
                    self.websocket_server = await websockets.serve(
                        handle_overlay_connection, "127.0.0.1", port
                    )
                    logger.info(f"WebSocket server started on port {port} for overlay")
                    logger.info(f"WebSocket server object: {self.websocket_server}")
                    logger.info(
                        f"WebSocket server bound to: {self.websocket_server.sockets}"
                    )
                    break
                except OSError as e:
                    logger.warning(f"WebSocket port {port} unavailable: {e}")
                    continue
            else:
                logger.error("Could not start WebSocket server on any port")

        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}")

    async def stop_websocket_server(self):
        """Zatrzymaj WebSocket serwer."""
        if self.websocket_server:
            try:
                self.websocket_server.close()
                await self.websocket_server.wait_closed()
                logger.info("WebSocket server stopped")
            except Exception as e:
                logger.error(f"Error stopping WebSocket server: {e}")

    async def broadcast_to_overlay(self, message):
        """Wyślij wiadomość do wszystkich połączonych overlay."""
        if not self.websocket_clients:
            return

        disconnected = set()
        for client in self.websocket_clients.copy():
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to overlay: {e}")
                disconnected.add(client)

        # Usuń rozłączonych klientów
        self.websocket_clients -= disconnected

    def stop_http_server(self):
        """Zatrzymaj HTTP serwer."""
        if self.http_server:
            try:
                self.http_server.shutdown()
                self.http_server.server_close()
                logger.info("HTTP server stopped")
            except Exception as e:
                logger.error(f"Error stopping HTTP server: {e}")

    async def run(self):
        """Uruchom główną pętlę klienta."""
        try:
            self.running = True

            # Store the current event loop for use by tray manager
            self.loop = asyncio.get_event_loop()

            # Initialize components (includes HTTP server)
            await self.initialize_components()

            # Start system tray
            if self.tray_manager:
                if self.tray_manager.start():
                    logger.info("System tray started successfully")
                    self.tray_manager.update_status("Starting...")
                else:
                    logger.warning("Failed to start system tray")

            # Try to connect to server (but don't fail if can't connect)
            try:
                await self.connect_to_server()
                logger.info("Connected to server successfully")
                self.update_status("Connected")
            except Exception as e:
                logger.warning(f"Could not connect to server: {e}")
                logger.warning(
                    "Running in standalone mode - overlay and local features will work"
                )
                self.update_status("Offline Mode")

            # Start wakeword monitoring
            await self.start_wakeword_monitoring()
            # Set status to ready with correct Polish status
            self.update_status("słucham")

            # Listen for server messages (or just keep running if no server)
            if self.websocket:
                # Start command queue processor alongside websocket listener and proactive check
                try:
                    await asyncio.gather(
                        self.listen_for_messages(),
                        self.process_command_queue(),
                        # self.periodic_proactive_check(),  # DISABLED: Temporarily disabled for release
                    )
                except asyncio.CancelledError:
                    logger.info("Main tasks cancelled")
            else:
                # Standalone mode - just keep running with wakeword detection
                logger.info("Running in standalone mode - use Ctrl+C to stop")
                try:
                    await self.process_command_queue()  # Just process commands
                except asyncio.CancelledError:
                    logger.info("Command queue processor cancelled")

        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        except Exception as e:
            logger.error(f"Error in client main loop: {e}")
        finally:
            await self.cleanup()

    async def process_command_queue(self):
        """Process commands from HTTP requests in the main async loop."""
        try:
            while self.running:
                try:
                    # Check for commands from HTTP handlers
                    while not self.command_queue.empty():
                        command = self.command_queue.get_nowait()
                        try:
                            await self.execute_command(command)
                        except Exception as e:
                            logger.error(f"Error executing command {command}: {e}")

                    await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                except asyncio.CancelledError:
                    # Task was cancelled, exit gracefully
                    logger.info("Command queue processor cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in command queue processor: {e}")
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Command queue processor cancelled")
        except Exception as e:
            logger.error(f"Fatal error in command queue processor: {e}")

    async def execute_command(self, command: dict):
        """Execute a command from the HTTP interface."""
        command_type = command.get("type")

        if command_type == "show_overlay":
            await self.show_overlay()
        elif command_type == "hide_overlay":
            await self.hide_overlay()
        elif command_type == "test_wakeword":
            query = command.get("query", "Powiedz mi coś o sobie.")
            await self.on_wakeword_detected(query)
        elif command_type == "test_tts":
            # Test TTS functionality
            test_text = command.get("text", "Test syntezatora mowy działa poprawnie.")
            if self.tts:
                try:
                    self.tts_playing = True
                    self.update_status("Testing TTS...")
                    await self.tts.speak(test_text)
                    logger.info("TTS test completed successfully")
                except Exception as e:
                    logger.error(f"TTS test failed: {e}")
                finally:
                    self.tts_playing = False
                    self.update_status("Listening...")
            else:
                logger.warning("TTS not available for testing")

        elif command_type == "open_settings":
            # Handle settings request from tray
            await self.handle_settings_request()

        elif command_type == "toggle_overlay":
            # Toggle overlay visibility
            try:
                await self.send_overlay_update(
                    {"action": "toggle_visibility", "toggle_visibility": True}
                )
                logger.info("Overlay visibility toggled")
            except Exception as e:
                logger.error(f"Failed to toggle overlay: {e}")

        elif command_type == "status_update":
            # Update overlay status
            try:
                status_data = command.get("data", {})
                await self.send_overlay_update(
                    {"action": "status_update", "status": status_data}
                )
                logger.debug("Status update sent to overlay")
            except Exception as e:
                logger.error(f"Failed to update overlay status: {e}")

        elif command_type == "show_notification":
            # Show notification via overlay
            try:
                notification_data = command.get("data", {})
                await self.send_overlay_update(
                    {"action": "show_notification", "notification": notification_data}
                )
                logger.info("Notification sent to overlay")
            except Exception as e:
                logger.error(f"Failed to show notification: {e}")

        else:
            logger.warning(f"Unknown command type: {command_type}")

    async def handle_settings_request(self):
        """Handle settings request from system tray."""
        try:
            # First try to open via overlay as native window
            await self.send_overlay_update(
                {"action": "open_settings", "open_settings": True}
            )
            logger.info("Settings window requested via overlay")
        except Exception as e:
            logger.error(f"Error requesting settings window: {e}")
            # Fallback: Try to open settings as app-like window
            try:
                import subprocess

                # Try Edge in app mode first
                subprocess.run(
                    [
                        "msedge",
                        "--app=http://localhost:5001/settings.html",
                        "--window-size=800,600",
                        "--disable-web-security",
                        "--disable-features=TranslateUI",
                        "--no-default-browser-check",
                    ],
                    check=True,
                )
                logger.info("Opened settings in Edge app mode as fallback")
            except Exception as e2:
                logger.error(f"Edge app mode failed: {e2}")
                # Final fallback - basic browser
                try:
                    import webbrowser

                    settings_url = "http://localhost:5001/settings.html"
                    webbrowser.open(settings_url)
                    logger.info("Opened settings via browser as last fallback")
                except Exception as e3:
                    logger.error(f"All settings opening methods failed: {e3}")

    async def start_wakeword_monitoring(self):
        """Uruchom monitoring słowa aktywującego w tle."""
        if self.wakeword_detector:
            self.monitoring_wakeword = True
            # Start wakeword detection in separate thread
            wakeword_thread = threading.Thread(
                target=self.wakeword_detector.start_detection, daemon=True
            )
            wakeword_thread.start()
            logger.info("Wakeword monitoring started")

    async def periodic_proactive_check(self):
        """Okresowo sprawdzaj proaktywne powiadomienia."""
        try:
            # Wait a bit after startup before starting proactive checks
            await asyncio.sleep(30)  # Initial delay

            while self.running:
                try:
                    # Request proactive notifications every 5 minutes
                    if self.websocket:
                        await self.request_proactive_notifications()

                        # Also update user context with current activity
                        from active_window_module import get_active_window_title

                        context_data = {
                            "timestamp": time.time(),
                            "active_window": get_active_window_title(),
                            "client_status": "active",
                        }
                        await self.update_user_context(context_data)

                    # Wait 5 minutes before next check
                    await asyncio.sleep(300)  # 5 minutes

                except asyncio.CancelledError:
                    logger.info("Periodic proactive check cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in periodic proactive check: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error

        except asyncio.CancelledError:
            logger.info("Periodic proactive check cancelled")
        except Exception as e:
            logger.error(f"Fatal error in periodic proactive check: {e}")

    async def cleanup(self):
        """Wyczyść zasoby przed zamknięciem."""
        try:
            self.running = False

            # Stop system tray first
            if self.tray_manager:
                self.tray_manager.stop()
                logger.info("System tray stopped")

            # Import required modules at function level
            import os
            import signal
            import subprocess

            # Stop HTTP server
            self.stop_http_server()
            # Stop WebSocket server
            await self.stop_websocket_server()
            # Stop wakeword detection
            if self.wakeword_detector:
                self.wakeword_detector.stop_detection()

            # Close overlay process with improved termination
            if self.overlay_process and self.overlay_process.poll() is None:
                logger.info(
                    f"Terminating overlay process (PID: {self.overlay_process.pid})..."
                )

                try:
                    # On Windows, try direct termination first

                    # Send CTRL_C_EVENT to gracefully close the overlay
                    try:
                        os.kill(self.overlay_process.pid, signal.CTRL_C_EVENT)
                        # Wait for graceful shutdown
                        self.overlay_process.wait(timeout=2)
                        logger.info("Overlay process terminated gracefully with CTRL_C")
                    except (subprocess.TimeoutExpired, OSError):
                        # If CTRL_C doesn't work, try SIGTERM
                        try:
                            self.overlay_process.terminate()
                            self.overlay_process.wait(timeout=3)
                            logger.info(
                                "Overlay process terminated gracefully with SIGTERM"
                            )
                        except subprocess.TimeoutExpired:
                            logger.warning(
                                "Overlay process didn't terminate gracefully, trying forceful kill..."
                            )
                            try:
                                self.overlay_process.kill()
                                self.overlay_process.wait(timeout=2)
                                logger.info("Overlay process killed forcefully")
                            except subprocess.TimeoutExpired:
                                logger.error(
                                    "Failed to kill overlay process with standard methods"
                                )
                                # Last resort: try Windows taskkill for overlay exe specifically
                                try:
                                    result = subprocess.run(
                                        ["taskkill", "/IM", "gaja-overlay.exe", "/F"],
                                        capture_output=True,
                                        text=True,
                                        timeout=5,
                                    )
                                    if result.returncode == 0:
                                        logger.info(
                                            "Successfully killed overlay process with taskkill by name"
                                        )
                                    else:
                                        logger.warning(
                                            f"Taskkill by name returned non-zero exit code: {result.returncode}"
                                        )

                                        # Try by PID as final fallback
                                        result = subprocess.run(
                                            [
                                                "taskkill",
                                                "/PID",
                                                str(self.overlay_process.pid),
                                                "/F",
                                            ],
                                            capture_output=True,
                                            text=True,
                                            timeout=5,
                                        )
                                        if result.returncode == 0:
                                            logger.info(
                                                "Successfully killed process with taskkill by PID"
                                            )
                                        else:
                                            logger.warning(
                                                f"Taskkill by PID returned non-zero exit code: {result.returncode}"
                                            )
                                except subprocess.TimeoutExpired:
                                    logger.error("Taskkill command timed out")
                                except Exception as taskkill_e:
                                    logger.error(f"Taskkill failed: {taskkill_e}")

                except Exception as term_e:
                    logger.error(f"Error terminating overlay process: {term_e}")

                # Set process to None to indicate it's been handled
                self.overlay_process = None

            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                logger.info("WebSocket connection closed")

            logger.info("Client cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _load_audio_modules(self):
        """Lazy load audio modules after dependencies are installed."""
        global TTSModule, WhisperASR, create_whisper_asr, create_audio_recorder, create_wakeword_detector

        try:
            # Load audio modules only after dependencies are ready
            # Note: These imports will fail if heavy dependencies are not available
            # That's expected - the dependency_manager will install them first
            # Try to load optimized audio components first
            try:
                from audio_modules.optimized_wakeword_detector import (
                    create_wakeword_detector,
                )

                logger.info("✅ Optimized wakeword detector loaded")
            except ImportError as e:
                logger.warning(f"Optimized wakeword detector not available: {e}")
                # Fallback to advanced wakeword detector
                try:
                    from audio_modules.advanced_wakeword_detector import (
                        create_wakeword_detector,
                    )

                    logger.info("✅ Advanced wakeword detector loaded as fallback")
                except ImportError as e2:
                    logger.warning(f"Advanced wakeword detector not available: {e2}")
                    # Fallback to simple wakeword detector
                    try:
                        from audio_modules.simple_wakeword_detector import (
                            create_wakeword_detector,
                        )

                        logger.info(
                            "✅ Simple wakeword detector loaded as final fallback"
                        )
                    except ImportError as e3:
                        logger.warning(
                            f"Simple wakeword detector also not available: {e3}"
                        )
                        create_wakeword_detector = None

            try:
                from audio_modules.optimized_whisper_asr import (
                    create_optimized_recorder,
                    create_optimized_whisper_async,
                )

                create_audio_recorder = create_optimized_recorder
                create_whisper_asr = create_optimized_whisper_async
                logger.info("✅ Optimized Whisper ASR loaded")
            except ImportError as e:
                logger.warning(f"Optimized Whisper ASR not available: {e}")
                # Fallback to legacy whisper
                try:
                    from audio_modules.whisper_asr import (
                        create_audio_recorder,
                        create_whisper_asr,
                    )

                    logger.info("✅ Legacy Whisper ASR loaded as fallback")
                except ImportError as e2:
                    logger.warning(f"Legacy Whisper ASR not available: {e2}")
                    create_whisper_asr = None
                    create_audio_recorder = None

            # Try enhanced modules if available, fallback to legacy
            try:
                # Use optimized TTS module when available
                from audio_modules.tts_module import TTSModule

                logger.info("✅ TTS Module loaded")
            except ImportError as e:
                logger.warning(f"TTS Module not available: {e}")
                TTSModule = None

            # Optimized Whisper ASR is already loaded above via create_optimized_whisper_async
            # No need for separate WhisperASR import as we use factory functions

            # Return True if at least some modules loaded
            basic_modules_available = any(
                [
                    create_wakeword_detector is not None,
                    create_whisper_asr is not None,
                    TTSModule is not None,
                ]
            )

            if basic_modules_available:
                logger.info("✅ Some audio modules loaded successfully")
            else:
                logger.warning("❌ No audio modules available")

            return basic_modules_available

        except Exception as e:
            logger.error(f"❌ Failed to load audio modules: {e}")
            logger.info("Audio features will not be available")
            return False

    async def request_proactive_notifications(self):
        """Request proactive notifications from server."""
        try:
            logger.debug("Requesting proactive notifications...")
            await self.send_message({"type": "proactive_check"})
        except Exception as e:
            logger.error(f"Error requesting proactive notifications: {e}")

    async def update_user_context(self, context_data: dict):
        """Send user context data to server for proactive notifications."""
        try:
            logger.debug(f"Updating user context: {context_data}")
            await self.send_message(
                {"type": "user_context_update", "data": context_data}
            )
        except Exception as e:
            logger.error(f"Error updating user context: {e}")

    async def handle_startup_briefing(self, briefing) -> None:
        """Handle startup briefing from server.

        Args:
            briefing: Dictionary or string containing briefing data from server
        """
        try:
            logger.info("Processing startup briefing...")

            # Handle both string and dict formats
            if isinstance(briefing, str):
                briefing_content = briefing
            elif isinstance(briefing, dict):
                # Extract briefing content
                text = briefing.get("text", "")
                summary = briefing.get("summary", "")
                briefing_content = text if text else summary
            else:
                logger.warning(f"Unexpected briefing format: {type(briefing)}")
                return

            if not briefing_content:
                logger.warning("Empty startup briefing received")
                return

            logger.info(f"Startup briefing received: {briefing_content[:100]}...")

            # Check if briefing was already delivered today
            if (
                "już został dzisiaj" in briefing_content.lower()
                or "dostarczony" in briefing_content.lower()
            ):
                logger.info("Startup briefing already delivered today - skipping TTS")
                self.update_status("Ready")
                return

            # Update overlay status
            self.update_status("Startup Briefing")
            await self.show_overlay()

            # Speak the briefing if TTS is available
            if self.tts and briefing_content:
                try:
                    self.tts_playing = True
                    await self.tts.speak(briefing_content)
                    logger.info("Startup briefing spoken successfully")
                except Exception as e:
                    logger.error(f"Error speaking startup briefing: {e}")
                finally:
                    self.tts_playing = False
                    await self.hide_overlay()
                    self.update_status("Ready")
            else:
                # No TTS available, just show briefly
                await asyncio.sleep(3)
                await self.hide_overlay()
                self.update_status("Ready")

        except Exception as e:
            logger.error(f"Error handling startup briefing: {e}")
            self.update_status("Error")

    async def handle_day_summary(self, summary: dict) -> None:
        """Handle day summary from server.

        Args:
            summary: Dictionary containing day summary data from server
        """
        try:
            logger.info("Processing day summary...")

            # Extract summary content
            summary_type = summary.get("type", "day_summary")
            content = summary.get("content", "")
            statistics = summary.get("statistics", {})
            narrative = summary.get("narrative", "")

            if not content and not narrative:
                logger.warning("Empty day summary received")
                return

            # Use narrative if available, otherwise use content
            summary_text = narrative if narrative else content

            logger.info(f"Day summary received: {summary_type}")

            # Update overlay status
            self.update_status(f"Day Summary: {summary_type}")
            await self.show_overlay()

            # Format summary for speech
            formatted_summary = self._format_day_summary(
                summary_type, summary_text, statistics
            )

            # Speak the summary if TTS is available
            if self.tts and formatted_summary:
                try:
                    self.tts_playing = True
                    await self.tts.speak(formatted_summary)
                    logger.info("Day summary spoken successfully")
                except Exception as e:
                    logger.error(f"Error speaking day summary: {e}")
                finally:
                    self.tts_playing = False
                    await self.hide_overlay()
                    self.update_status("Listening...")
            else:
                # No TTS available, just show briefly
                await asyncio.sleep(3)
                await self.hide_overlay()
                self.update_status("Listening...")

        except Exception as e:
            logger.error(f"Error handling day summary: {e}")
            self.update_status("Error")

    async def handle_proactive_notifications(self, data: dict) -> None:
        """Handle proactive notifications from server.

        Args:
            data: Dictionary containing notification data from server
        """
        try:
            logger.info("Processing proactive notifications...")

            # Extract notification data
            notifications = data.get("notifications", [])
            priority = data.get("priority", "normal")

            if not notifications:
                logger.info("No proactive notifications to process")
                return

            logger.info(
                f"Received {len(notifications)} proactive notifications (priority: {priority})"
            )

            # Process each notification
            for notification in notifications:
                await self._process_single_notification(notification, priority)

                # Add delay between notifications to avoid overwhelming user
                if len(notifications) > 1:
                    await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Error handling proactive notifications: {e}")
            self.update_status("Error")

    async def _process_single_notification(
        self, notification: dict, priority: str = "normal"
    ) -> None:
        """Process a single proactive notification.

        Args:
            notification: Single notification data
            priority: Notification priority level
        """
        try:
            # Extract notification content
            message = notification.get("message", "")
            title = notification.get("title", "")
            notification_type = notification.get("type", "info")

            if not message:
                logger.warning("Empty notification message")
                return

            # Format notification for display and speech
            display_text = f"{title}: {message}" if title else message

            logger.info(
                f"Processing notification: {notification_type} - {display_text[:50]}..."
            )

            # Update overlay with notification
            self.update_status(f"Notification: {display_text[:30]}...")
            await self.show_overlay()

            # Speak notification based on priority
            should_speak = priority in ["high", "urgent"] or notification_type in [
                "urgent",
                "important",
            ]

            if self.tts and should_speak:
                try:
                    self.tts_playing = True
                    await self.tts.speak(display_text)
                    logger.info("Notification spoken successfully")
                except Exception as e:
                    logger.error(f"Error speaking notification: {e}")
                finally:
                    self.tts_playing = False

            # Show notification for appropriate time based on priority
            display_time = 5 if priority in ["high", "urgent"] else 3
            await asyncio.sleep(display_time)

            await self.hide_overlay()
            self.update_status("Listening...")

        except Exception as e:
            logger.error(f"Error processing single notification: {e}")

    def _format_day_summary(
        self, summary_type: str, content: str, statistics: dict
    ) -> str:
        """Format day summary for speech output.

        Args:
            summary_type: Type of summary
            content: Summary content text
            statistics: Summary statistics

        Returns:
            Formatted summary text for speech
        """
        try:
            if not content and not statistics:
                return "Podsumowanie dnia jest puste."

            # If we have content, use it directly
            if content:
                return content

            # Otherwise, format statistics
            if statistics:
                active_time = statistics.get("active_time_hours", 0)
                interactions = statistics.get("interactions_count", 0)
                productivity = statistics.get("productivity_score", 0)

                return (
                    f"Statystyki dnia: {active_time:.1f} godzin aktywności, "
                    f"{interactions} interakcji, "
                    f"produktywność: {productivity:.0%}."
                )

            return f"Otrzymano {summary_type}, ale brak szczegółów."

        except Exception as e:
            logger.error(f"Error formatting day summary: {e}")
            return "Wystąpił błąd podczas formatowania podsumowania."

    async def send_to_server(self, message: dict):
        """Send message to server - alias for send_message."""
        await self.send_message(message)

    async def process_wakeword_detection(self):
        """Process wakeword detection - simulate the workflow."""
        if not self.wakeword_detector:
            logger.warning("Wakeword detector not available")
            return

        # This simulates the actual wakeword detection process
        await self.on_wakeword_detected()

    async def speak_text(self, text: str):
        """Speak text using TTS."""
        if not self.tts:
            logger.warning("TTS module not available")
            return

        try:
            self.tts_playing = True
            self.last_tts_text = text
            self.update_status("Odpowiadam...")  # ← Dodane
            await self.tts.speak(text)
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            self.tts_playing = False
            self.update_status("Ready")  # ← Dodane

    def list_available_audio_devices(self):
        """List available audio devices."""
        try:
            # Try to import sounddevice for device listing
            import sounddevice as sd

            devices = sd.query_devices()
            input_devices = []
            output_devices = []

            default_input = sd.default.device[0] if sd.default.device else None
            default_output = sd.default.device[1] if sd.default.device else None

            for i, device in enumerate(devices):
                device_info = {
                    "id": i,
                    "name": device["name"],
                    "channels": device.get("max_input_channels", 0),
                    "sample_rate": device.get("default_samplerate", 44100),
                    "is_default": False,
                }

                # Check if device has input capabilities
                if device["max_input_channels"] > 0:
                    device_info_input = device_info.copy()
                    device_info_input["is_default"] = i == default_input
                    device_info_input["type"] = "input"
                    input_devices.append(device_info_input)

                # Check if device has output capabilities
                if device["max_output_channels"] > 0:
                    device_info_output = device_info.copy()
                    device_info_output["is_default"] = i == default_output
                    device_info_output["type"] = "output"
                    output_devices.append(device_info_output)

            return {
                "input_devices": input_devices,
                "output_devices": output_devices,
                "default_input": default_input,
                "default_output": default_output,
            }
        except ImportError:
            logger.warning("sounddevice not available for device listing")
            return {
                "input_devices": [
                    {"id": 0, "name": "Default Device", "is_default": True}
                ],
                "output_devices": [
                    {"id": 0, "name": "Default Device", "is_default": True}
                ],
                "default_input": 0,
                "default_output": 0,
            }
        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")
            return {
                "input_devices": [
                    {"id": 0, "name": "Error listing devices", "is_default": True}
                ],
                "output_devices": [
                    {"id": 0, "name": "Error listing devices", "is_default": True}
                ],
                "default_input": None,
                "default_output": None,
            }

    async def send_overlay_update(self, data: dict):
        """Send update to overlay."""
        try:
            # Update internal status
            if "status" in data:
                self.current_status = data["status"]

            # Notify SSE clients using the existing method
            message = f"data: {json.dumps(data)}\n\n"

            # Send to all connected SSE clients
            for client in self.sse_clients[
                :
            ]:  # Copy list to avoid modification during iteration
                try:
                    client.wfile.write(message.encode())
                    client.wfile.flush()
                except Exception as e:
                    logger.debug(f"SSE client disconnected: {e}")
                    self.remove_sse_client(client)

        except Exception as e:
            logger.error(f"Error sending overlay update: {e}")

    async def record_and_transcribe(self):
        """Record audio and transcribe it."""
        if not self.audio_recorder or not self.whisper_asr:
            raise Exception("Recording components not available")

        try:
            # Record audio
            audio_data = await self.audio_recorder.record()

            # Transcribe
            transcription = await self.whisper_asr.transcribe(audio_data)
            return transcription

        except Exception as e:
            logger.error(f"Record and transcribe error: {e}")
            raise Exception("Recording failed") from e

    async def stop_current_audio(self):
        """Stop current audio playback."""
        if self.tts and self.tts_playing:
            try:
                self.tts.stop()
                self.tts_playing = False
            except Exception as e:
                logger.error(f"Error stopping audio: {e}")

    def format_briefing_text(self, briefing_data: dict) -> str:
        """Format briefing data for display."""
        if not briefing_data:
            return "No briefing available"

        sections = []
        if "weather" in briefing_data:
            sections.append(f"Weather: {briefing_data['weather']}")
        if "calendar" in briefing_data:
            sections.append(f"Calendar: {briefing_data['calendar']}")
        if "tasks" in briefing_data:
            sections.append(f"Tasks: {briefing_data['tasks']}")

        return "\n".join(sections) if sections else "No briefing data"

    def format_summary_text(self, summary_data: dict) -> str:
        """Format daily summary for display."""
        if not summary_data:
            return "No summary available"

        sections = []
        if "completed_tasks" in summary_data:
            sections.append(f"Completed: {summary_data['completed_tasks']}")
        if "pending_tasks" in summary_data:
            sections.append(f"Pending: {summary_data['pending_tasks']}")
        if "highlights" in summary_data:
            sections.append(f"Highlights: {summary_data['highlights']}")

        return "\n".join(sections) if sections else "No summary data"

    async def show_overlay(self):
        """Show the overlay (async version)."""
        try:
            self.overlay_visible = True
            self.update_status("Overlay Shown")
            logger.info("Overlay shown")

            # Send update to overlay via SSE
            await self.send_overlay_update(
                {
                    "status": "Overlay Shown",
                    "overlay_visible": True,
                    "show_overlay": True,
                }
            )
        except Exception as e:
            logger.error(f"Error showing overlay: {e}")

    async def hide_overlay(self):
        """Hide the overlay."""
        try:
            self.overlay_visible = False
            self.update_status("Overlay Hidden")
            logger.info("Overlay hidden")

            # Send update to overlay via SSE
            await self.send_overlay_update(
                {
                    "status": "Overlay Hidden",
                    "overlay_visible": False,
                    "hide_overlay": True,
                }
            )
        except Exception as e:
            logger.error(f"Error hiding overlay: {e}")

    def show_overlay_sync(self):
        """Show the overlay (sync version for tray)."""
        try:
            self.overlay_visible = True
            self.update_status("Overlay Shown")
            logger.info("Overlay shown from system tray")

            # Send update to overlay via SSE
            asyncio.create_task(
                self.send_overlay_update(
                    {
                        "status": "Overlay Shown",
                        "overlay_visible": True,
                        "show_overlay": True,
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error showing overlay: {e}")

    def stop_wakeword_monitoring(self):
        """Stop wakeword monitoring (called from tray)."""
        try:
            if self.wakeword_detector and self.monitoring_wakeword:
                self.wakeword_detector.stop_detection()
                self.monitoring_wakeword = False
                self.update_status("Monitoring Stopped")
                logger.info("Wakeword monitoring stopped from system tray")
        except Exception as e:
            logger.error(f"Error stopping wakeword monitoring: {e}")

    def start_wakeword_monitoring_sync(self):
        """Start wakeword monitoring synchronously (called from tray)."""
        try:
            if self.wakeword_detector and not self.monitoring_wakeword:
                # Since this is called from tray thread, we need to schedule it
                if self.running:
                    asyncio.create_task(self.start_wakeword_monitoring())
                    logger.info("Wakeword monitoring start scheduled from system tray")
        except Exception as e:
            logger.error(f"Error starting wakeword monitoring: {e}")

    # ...existing code...


class StatusHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler dla overlay status API."""

    def __init__(self, client_app, *args, **kwargs):
        self.client_app = client_app
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Obsługa GET requests."""
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            status = {
                "status": self.client_app.get_current_status(),
                "text": self.client_app.last_tts_text,
                "is_listening": self.client_app.recording_command,
                "is_speaking": self.client_app.tts_playing,
                "wake_word_detected": self.client_app.wake_word_detected,
                "overlay_visible": self.client_app.overlay_visible,
                "monitoring": self.client_app.monitoring_wakeword,
                "tts_playing": self.client_app.tts_playing,
                "last_tts": self.client_app.last_tts_text,
            }
            self.wfile.write(json.dumps(status).encode())

        elif self.path == "/status/stream":
            # SSE endpoint for real-time status updates
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Add client to SSE list
            self.client_app.add_sse_client(self)

            # Send initial status
            initial_status = {
                "status": self.client_app.current_status,
                "text": self.client_app.last_tts_text,
                "is_listening": self.client_app.recording_command,
                "is_speaking": self.client_app.tts_playing,
                "wake_word_detected": self.client_app.wake_word_detected,
                "overlay_visible": self.client_app.overlay_visible,
            }

            try:
                message = f"data: {json.dumps(initial_status)}\n\n"
                self.wfile.write(message.encode())
                self.wfile.flush()

                logger.info("SSE client connected")

                # Keep connection alive by monitoring for client disconnect
                import threading

                def monitor_connection():
                    try:
                        # Send heartbeat every 30 seconds to detect disconnection
                        import time

                        while True:
                            time.sleep(30)
                            try:
                                heartbeat = 'data: {"heartbeat": true}\n\n'
                                self.wfile.write(heartbeat.encode())
                                self.wfile.flush()
                            except:
                                logger.info("SSE client disconnected")
                                self.client_app.remove_sse_client(self)
                                break
                    except:
                        self.client_app.remove_sse_client(self)

                # Start heartbeat thread to keep connection alive
                heartbeat_thread = threading.Thread(
                    target=monitor_connection, daemon=True
                )
                heartbeat_thread.start()

                return  # Don't close connection immediately

            except Exception as e:
                logger.error(f"SSE connection error: {e}")
                self.client_app.remove_sse_client(self)

        elif self.path == "/api/trigger_wakeword":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Add command to queue
            command = {"type": "test_wakeword"}
            self.client_app.command_queue.put(command)

            self.wfile.write(json.dumps({"success": True}).encode())

        elif self.path == "/api/audio_devices":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if self.client_app.settings_manager:
                    devices = self.client_app.settings_manager.get_audio_devices()
                    self.wfile.write(json.dumps(devices).encode())
                else:
                    # Fallback if settings manager not available
                    devices = self.client_app.list_available_audio_devices()
                    fallback_response = {
                        "input_devices": devices,
                        "output_devices": devices,
                    }
                    self.wfile.write(json.dumps(fallback_response).encode())
            except Exception as e:
                logger.error(f"Error getting audio devices: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/api/connection_status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if self.client_app.settings_manager:
                    status = self.client_app.settings_manager.get_connection_status()
                    self.wfile.write(json.dumps(status).encode())
                else:
                    # Fallback status
                    status = {
                        "connected": self.client_app.websocket is not None,
                        "error": "Settings manager not available",
                    }
                    self.wfile.write(json.dumps(status).encode())
            except Exception as e:
                logger.error(f"Error getting connection status: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/api/current_settings":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                if self.client_app.settings_manager:
                    settings = self.client_app.settings_manager.load_settings()
                    self.wfile.write(json.dumps(settings).encode())
                else:
                    # Fallback to client config
                    self.wfile.write(json.dumps(self.client_app.config).encode())
            except Exception as e:
                logger.error(f"Error getting current settings: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/api/test_microphone":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                # Test microphone by recording a short sample
                if self.client_app.audio_recorder:
                    # This is a simplified test - in reality you'd want to record and analyze
                    test_result = {
                        "success": True,
                        "message": "Mikrofon jest dostępny",
                        "device_info": self.client_app.list_available_audio_devices(),
                    }
                else:
                    test_result = {
                        "success": False,
                        "message": "Rejestrator audio nie jest dostępny",
                        "device_info": self.client_app.list_available_audio_devices(),
                    }

                self.wfile.write(json.dumps(test_result).encode())
            except Exception as e:
                logger.error(f"Error testing microphone: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": str(e), "success": False}).encode()
                )

        elif self.path == "/api/test_tts":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                # Test TTS by speaking a short phrase
                if self.client_app.tts:
                    command = {
                        "type": "test_tts",
                        "text": "Test syntezatora mowy. Jeśli słyszysz tę wiadomość, TTS działa poprawnie.",
                    }
                    self.client_app.command_queue.put(command)

                    test_result = {
                        "success": True,
                        "message": "Test TTS został uruchomiony",
                    }
                else:
                    test_result = {
                        "success": False,
                        "message": "Moduł TTS nie jest dostępny",
                    }

                self.wfile.write(json.dumps(test_result).encode())
            except Exception as e:
                logger.error(f"Error testing TTS: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": str(e), "success": False}).encode()
                )

        elif self.path == "/settings.html":
            # Serve settings.html file
            try:
                settings_path = Path(__file__).parent / "resources" / "settings.html"
                if settings_path.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header(
                        "Cache-Control", "no-cache, no-store, must-revalidate"
                    )
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.end_headers()

                    with open(settings_path, encoding="utf-8") as f:
                        content = f.read()
                    self.wfile.write(content.encode("utf-8"))
                    logger.info(f"Served settings.html from {settings_path}")
                else:
                    logger.error(f"Settings file not found at {settings_path}")
                    self.send_response(404)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b"<h1>Settings file not found</h1>")
            except Exception as e:
                logger.error(f"Error serving settings.html: {e}")
                self.send_response(500)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(f"<h1>Error: {e}</h1>".encode())

        elif self.path == "/gaja.ico":
            # Serve gaja.ico file
            try:
                icon_path = Path(__file__).parent.parent / "gaja.ico"
                if icon_path.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "image/x-icon")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.end_headers()
                    with open(icon_path, "rb") as f:
                        self.wfile.write(f.read())
                    logger.info(f"Served gaja.ico from {icon_path}")
                else:
                    logger.error(f"Icon file not found at {icon_path}")
                    self.send_response(404)
                    self.end_headers()
            except Exception as e:
                logger.error(f"Error serving gaja.ico: {e}")
                self.send_response(500)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Obsługa POST requests."""
        if self.path == "/api/update_status":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                new_status = data.get("status", "")

                if new_status:
                    command = {
                        "type": "status_update",
                        "status": new_status,
                    }
                    self.client_app.command_queue.put(command)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode())

            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/api/save_settings":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                settings = data.get("settings", {})

                if self.client_app.settings_manager:
                    success = self.client_app.settings_manager.save_settings(settings)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": success}).encode())
                else:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "Settings manager not available"}).encode()
                    )

            except Exception as e:
                logger.error(f"Error saving settings: {e}")
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Obsługa CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Override to use custom logging."""
        logger.debug(f"HTTP: {format % args}")


async def main():
    """Główna funkcja klienta."""
    print("Starting GAJA Client...")

    app = ClientApp()

    try:
        await app.run()

    except asyncio.CancelledError:
        logger.info("Client cancelled")
    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
    finally:
        logger.info("Client shutdown")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
