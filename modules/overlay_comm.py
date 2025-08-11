"""Overlay communication module for GAJA client."""

import asyncio
import json
import logging
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import websockets

logger = logging.getLogger(__name__)


class OverlayManager:
    """Manages overlay communication via HTTP SSE and WebSocket."""
    
    def __init__(self, client_app):
        self.client = client_app
        self.overlay_visible = False
        self.overlay_process: Optional[subprocess.Popen] = None
        
        # HTTP SSE server
        self.http_server: Optional[HTTPServer] = None
        self.http_port = 8765
        self.sse_clients: List[Any] = []
        
        # WebSocket server
        self.websocket_server: Optional[Any] = None
        self.websocket_port = 8766
        self.websocket_clients: Set[Any] = set()
        
        # Debouncing for status updates
        self.last_status_update = 0
        self.status_update_interval = 0.1  # 100ms minimum between updates
    
    async def start_overlay(self):
        """Start the external Tauri overlay."""
        try:
            # Possible paths to overlay executable
            overlay_paths = [
                Path(__file__).parent.parent.parent / "Gaja-Overlay" / "target" / "release" / "gaja-overlay.exe",
                Path(__file__).parent.parent.parent / "Gaja-Overlay" / "target" / "debug" / "gaja-overlay.exe",
                Path(__file__).parent.parent / "overlay" / "target" / "release" / "gaja-overlay.exe",
                Path(__file__).parent.parent / "overlay" / "target" / "debug" / "gaja-overlay.exe",
            ]
            
            overlay_exe = None
            for path in overlay_paths:
                if path.exists():
                    overlay_exe = path
                    break
            
            if not overlay_exe:
                logger.error("Overlay executable not found in any expected location")
                return False
            
            logger.info(f"Starting overlay from: {overlay_exe}")
            self.overlay_process = subprocess.Popen(
                [str(overlay_exe)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
            )
            
            # Give overlay time to start
            await asyncio.sleep(2)
            
            if self.overlay_process.poll() is None:
                logger.info("✅ Overlay started successfully")
                return True
            else:
                logger.error("❌ Overlay process exited immediately")
                return False
                
        except Exception as e:
            logger.error(f"Error starting overlay: {e}")
            return False
    
    def start_http_server(self):
        """Start HTTP server for SSE communication."""
        try:
            def create_handler(*args, **kwargs):
                return StatusHTTPHandler(self, *args, **kwargs)
            
            self.http_server = HTTPServer(("127.0.0.1", self.http_port), create_handler)
            
            # Start server in background thread
            server_thread = threading.Thread(
                target=self.http_server.serve_forever, daemon=True
            )
            server_thread.start()
            
            logger.info(f"✅ HTTP SSE server started on port {self.http_port}")
            
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
    
    async def start_websocket_server(self):
        """Start WebSocket server for overlay communication."""
        try:
            async def handle_overlay_connection(websocket, path):
                """Handle new overlay WebSocket connections."""
                logger.info("Overlay WebSocket client connected")
                self.websocket_clients.add(websocket)
                
                try:
                    # Send initial status
                    await self._send_websocket_status(websocket)
                    
                    # Keep connection alive and handle messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self._handle_overlay_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from overlay: {message}")
                        except Exception as e:
                            logger.error(f"Error handling overlay message: {e}")
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Overlay WebSocket client disconnected")
                finally:
                    self.websocket_clients.discard(websocket)
            
            # Try different ports if default is taken
            for port in range(self.websocket_port, self.websocket_port + 10):
                try:
                    self.websocket_server = await websockets.serve(
                        handle_overlay_connection, "127.0.0.1", port
                    )
                    self.websocket_port = port
                    logger.info(f"✅ WebSocket server started on port {port}")
                    break
                except OSError as e:
                    if port == self.websocket_port + 9:  # Last attempt
                        raise e
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def show_overlay(self):
        """Show the overlay."""
        if not self.overlay_visible:
            self.overlay_visible = True
            await self.broadcast_to_overlay({"type": "show"})
    
    async def hide_overlay(self):
        """Hide the overlay."""
        if self.overlay_visible:
            self.overlay_visible = False
            await self.broadcast_to_overlay({"type": "hide"})
    
    async def toggle_overlay(self):
        """Toggle overlay visibility."""
        if self.overlay_visible:
            await self.hide_overlay()
        else:
            await self.show_overlay()
    
    def update_status(self, status: str):
        """Update status and notify overlay clients with debouncing."""
        import time
        
        # Debounce status updates for non-critical states
        current_time = time.time()
        is_critical = status in ["Przetwarzam...", "Mówię...", "Przetwarzam zapytanie..."]
        
        if not is_critical and (current_time - self.last_status_update) < self.status_update_interval:
            return  # Skip this update
        
        self.last_status_update = current_time
        
        # Notify SSE clients
        self.notify_sse_clients()
        
        # Notify WebSocket clients
        if self.websocket_clients:
            message = {
                "type": "status",
                "data": {
                    "status": status,
                    "text": self.client.last_tts_text,
                    "is_listening": self.client.recording_command,
                    "is_speaking": self.client.tts_playing,
                    "wake_word_detected": self.client.wake_word_detected,
                    "overlay_visible": self.overlay_visible,
                    "monitoring": getattr(self.client, 'monitoring_wakeword', True),
                },
            }
            
            # Broadcast in background to avoid blocking
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.broadcast_to_overlay(message))
                else:
                    asyncio.run(self.broadcast_to_overlay(message))
            except Exception as e:
                logger.debug(f"Could not broadcast to overlay: {e}")
    
    def notify_sse_clients(self):
        """Notify SSE clients of status change."""
        # Determine content visibility
        has_meaningful_text = self.client.last_tts_text and self.client.last_tts_text not in [
            "", "Listening...", "Ready", "Offline"
        ]
        
        should_show_content = (
            self.client.wake_word_detected
            or self.client.tts_playing
            or self.client.recording_command
            or has_meaningful_text
        )
        
        status_data = {
            "status": self.client.current_status,
            "text": self.client.last_tts_text if self.client.last_tts_text else self.client.current_status,
            "is_listening": not self.client.recording_command and not self.client.tts_playing and not self.client.wake_word_detected,
            "is_speaking": self.client.tts_playing,
            "wake_word_detected": self.client.wake_word_detected,
            "show_content": should_show_content,
            "monitoring": getattr(self.client, 'monitoring_wakeword', True),
        }
        
        # Send to all SSE clients
        data_json = json.dumps(status_data)
        for client in self.sse_clients[:]:  # Copy list to avoid modification during iteration
            try:
                client.wfile.write(f"data: {data_json}\n\n".encode())
                client.wfile.flush()
            except Exception as e:
                logger.debug(f"SSE client write error: {e}")
                self.remove_sse_client(client)
    
    async def broadcast_to_overlay(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket overlay clients."""
        if not self.websocket_clients:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.debug(f"Error sending to overlay client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def _send_websocket_status(self, websocket):
        """Send current status to a specific WebSocket client."""
        message = {
            "type": "status",
            "data": {
                "status": self.client.current_status,
                "text": self.client.last_tts_text,
                "is_listening": self.client.recording_command,
                "is_speaking": self.client.tts_playing,
                "wake_word_detected": self.client.wake_word_detected,
                "overlay_visible": self.overlay_visible,
                "monitoring": getattr(self.client, 'monitoring_wakeword', True),
            },
        }
        
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.debug(f"Error sending status to WebSocket client: {e}")
    
    async def _handle_overlay_message(self, data: Dict[str, Any]):
        """Handle messages from overlay."""
        message_type = data.get("type")
        
        if message_type == "toggle_visibility":
            await self.toggle_overlay()
        elif message_type == "manual_trigger":
            if hasattr(self.client, 'wakeword_detector'):
                self.client.wakeword_detector.trigger_manual_detection()
        else:
            logger.debug(f"Unknown overlay message type: {message_type}")
    
    def add_sse_client(self, client):
        """Add SSE client."""
        self.sse_clients.append(client)
    
    def remove_sse_client(self, client):
        """Remove SSE client."""
        if client in self.sse_clients:
            self.sse_clients.remove(client)
    
    async def cleanup(self):
        """Clean up overlay resources."""
        # Close WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        # Close HTTP server
        if self.http_server:
            self.http_server.shutdown()
        
        # Terminate overlay process
        if self.overlay_process:
            try:
                self.overlay_process.terminate()
                self.overlay_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.overlay_process.kill()
            except Exception as e:
                logger.debug(f"Error terminating overlay: {e}")


class StatusHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for SSE communication."""
    
    def __init__(self, overlay_manager: OverlayManager, *args, **kwargs):
        self.overlay_manager = overlay_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for SSE."""
        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            # Add this client to SSE clients
            self.overlay_manager.add_sse_client(self)
            
            # Send initial status
            try:
                self.overlay_manager.notify_sse_clients()
            except Exception as e:
                logger.debug(f"Error sending initial SSE status: {e}")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass  # Disable default logging
