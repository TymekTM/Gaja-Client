"""Main GAJA Client application module - simplified and modular."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from modules.networking import WebSocketManager, create_voice_command_message
from modules.handlers import MessageHandlers
from modules.overlay_comm import OverlayManager
from modules.tts_playback import TTSPlayback

logger = logging.getLogger(__name__)


class GajaClient:
    """Main GAJA client application - simplified and modular."""
    
    def __init__(self):
        # Load configuration
        self.config = self._load_config()
        
        # Core state
        self.running = False
        self.current_status = "Initializing..."
        self.last_tts_text = ""
        self.tts_playing = False
        self.wake_word_detected = False
        self.recording_command = False
        self.monitoring_wakeword = True
        
        # Component references
        self.tts: Optional[object] = None
        self.whisper_asr: Optional[object] = None
        self.wakeword_detector: Optional[object] = None
        
        # Modular components
        self.networking = WebSocketManager(
            server_url=self.config.get("server_url", "ws://localhost:8001/ws/client1"),
            user_id=self.config.get("user_id", "1")
        )
        self.handlers = MessageHandlers(self)
        self.overlay = OverlayManager(self)
        self.tts_player = TTSPlayback()
        
        # Set up message routing
        self.networking.set_message_handler(self._route_message)
        
        # Initialize system tray and settings manager if available
        self.tray_manager = None
        self.settings_manager = None
        self._init_optional_components()
    
    def _load_config(self) -> dict:
        """Load client configuration."""
        config_path = Path(__file__).parent / "client_config.json"
        
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        
        # Default configuration
        return {
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
    
    def _init_optional_components(self):
        """Initialize optional components that may not be available."""
        # System tray
        try:
            from modules.tray_manager import TrayManager
            self.tray_manager = TrayManager(self)
        except ImportError:
            logger.info("Tray manager not available")
        
        # Settings manager
        try:
            from modules.settings_manager import SettingsManager
            self.settings_manager = SettingsManager(self)
        except ImportError:
            logger.info("Settings manager not available")
    
    async def initialize(self):
        """Initialize all client components."""
        try:
            logger.info("Initializing GAJA client components...")
            
            # Initialize audio components
            await self._init_audio_components()
            
            # Initialize overlay
            self.overlay.start_http_server()
            await self.overlay.start_websocket_server()
            
            # Start external overlay
            await self.overlay.start_overlay()
            
            self.update_status("Ready")
            logger.info("✅ All client components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    async def _init_audio_components(self):
        """Initialize audio-related components."""
        # Import audio modules dynamically
        try:
            from audio_modules.tts_module import TTSModule
            self.tts = TTSModule()
            self.tts_player = TTSPlayback(self.tts)
            logger.info("TTS module initialized")
        except ImportError as e:
            logger.warning(f"TTS module not available: {e}")
        
        try:
            from audio_modules.optimized_whisper_asr import create_optimized_whisper_async
            whisper_config = self.config.get("whisper", {})
            self.whisper_asr = await create_optimized_whisper_async(whisper_config)
            logger.info("Whisper ASR initialized")
        except ImportError as e:
            logger.warning(f"Whisper ASR not available: {e}")
        
        try:
            from audio_modules.wakeword_detector import create_wakeword_detector
            wakeword_config = self.config.get("wakeword", {})
            if wakeword_config.get("enabled", True):
                self.wakeword_detector = create_wakeword_detector(
                    wakeword_config, self.on_wakeword_detected
                )
                if self.whisper_asr:
                    self.wakeword_detector.set_whisper_asr(self.whisper_asr)
                logger.info("Wakeword detector initialized")
        except ImportError as e:
            logger.warning(f"Wakeword detector not available: {e}")
    
    async def start(self):
        """Start the GAJA client."""
        logger.info("🚀 Starting GAJA client...")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Start networking
            if not await self.networking.start():
                logger.error("Failed to connect to server")
                return False
            
            # Start wakeword detection
            if self.wakeword_detector:
                self.wakeword_detector.start_detection()
            
            self.running = True
            self.update_status("Listening...")
            
            logger.info("✅ GAJA client started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting client: {e}")
            return False
    
    async def stop(self):
        """Stop the GAJA client."""
        logger.info("🛑 Stopping GAJA client...")
        
        self.running = False
        
        # Stop wakeword detection
        if self.wakeword_detector:
            self.wakeword_detector.stop_detection()
        
        # Stop TTS
        if self.tts_player:
            self.tts_player.stop()
        
        # Stop networking
        await self.networking.stop()
        
        # Cleanup overlay
        await self.overlay.cleanup()
        
        logger.info("✅ GAJA client stopped")
    
    async def run(self):
        """Run the client until stopped."""
        if not await self.start():
            return
        
        try:
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.stop()
    
    # ==================== MESSAGE ROUTING ====================
    
    async def _route_message(self, data: dict):
        """Route incoming messages to appropriate handlers."""
        message_type = data.get("type")
        
        # Track message limits
        if "message_limit" in data:
            # Handle message limits if needed
            pass
        if "daily_message_count" in data:
            # Handle daily counts if needed  
            pass
        
        # Route to appropriate handler
        if message_type == "ai_response":
            await self.handlers.handle_ai_response(data)
        elif message_type == "clarification_request":
            await self.handlers.handle_clarification_request(data)
        elif message_type == "daily_briefing":
            await self.handlers.handle_daily_briefing(data)
        elif message_type == "handshake":
            await self.handlers.handle_handshake(data)
        elif message_type == "startup_briefing":
            briefing = data.get("briefing", {})
            await self.handlers.handle_startup_briefing(briefing)
        elif message_type == "day_summary":
            summary = data.get("summary", {})
            await self.handlers.handle_day_summary(summary)
        elif message_type == "proactive_notifications":
            await self.handlers.handle_proactive_notifications(data)
        elif message_type in ["day_summary", "week_summary", "day_narrative", "behavior_insights", "routine_insights"]:
            await self.handlers.handle_summary_response(data)
        else:
            logger.warning(f"Unknown message type: {message_type}")
    
    # ==================== VOICE COMMAND HANDLING ====================
    
    async def on_wakeword_detected(self, query: str):
        """Handle wakeword detection with voice command."""
        if not query or not query.strip():
            logger.warning("Empty query received from wakeword detector")
            return
        
        logger.info(f"Voice command detected: '{query}'")
        
        try:
            self.wake_word_detected = True
            self.recording_command = False  # Recording already done by wakeword detector
            self.update_status("Przetwarzam zapytanie...")
            
            # Send command to server
            message = create_voice_command_message(query.strip())
            await self.networking.send_message(message)
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            self.update_status("Error")
            await asyncio.sleep(1)
            self.update_status("Listening...")
            # Reset state
            self.wake_word_detected = False
            self.recording_command = False
    
    # ==================== TTS HELPERS (from original helpers) ====================
    
    async def play_server_tts(self, tts_audio_b64: str, volume: float = 1.0):
        """Play server-provided TTS audio."""
        return await self.tts_player.play_server_tts(tts_audio_b64, volume)
    
    async def speak_text_local(self, text: str):
        """Speak text using local TTS."""
        return await self.tts_player.speak_text_local(text)
    
    # ==================== OVERLAY MANAGEMENT ====================
    
    async def show_overlay(self):
        """Show overlay."""
        await self.overlay.show_overlay()
    
    async def hide_overlay(self):
        """Hide overlay."""
        await self.overlay.hide_overlay()
    
    async def toggle_overlay(self):
        """Toggle overlay visibility."""
        await self.overlay.toggle_overlay()
    
    def update_status(self, status: str):
        """Update status and notify overlay."""
        self.current_status = status
        self.overlay.update_status(status)
        
        # Update system tray if available
        if self.tray_manager:
            self.tray_manager.update_status(status)
    
    # ==================== ADDITIONAL HELPERS ====================
    
    async def restart_recording_after_clarification(self):
        """Restart recording after clarification (if needed)."""
        # Simple beep to indicate ready for clarification response
        try:
            import winsound
            winsound.Beep(800, 200)
        except Exception:
            pass
        
        # Ensure wakeword detection is active
        if self.wakeword_detector and hasattr(self.wakeword_detector, 'start_detection'):
            # Wakeword detector should already be running
            pass


# Entry point for the application
async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from some loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.INFO)
    
    # Create and run client
    client = GajaClient()
    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
