"""WebSocket networking module for GAJA client."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Handles WebSocket connection, reconnection, and message routing."""
    
    def __init__(self, server_url: str, user_id: str):
        self.server_url = server_url
        self.user_id = user_id
        self.websocket: Optional[Any] = None  # Use Any to avoid websockets type issues
        self.running = False
        self.connected = False
        
        # Reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 0  # 0 = infinite
        self.reconnect_delay = 5.0
        self.max_reconnect_delay = 60.0
        self.reconnect_task: Optional[asyncio.Task] = None
        
        # Message handler
        self.message_handler: Optional[Callable] = None
        
    def set_message_handler(self, handler: Callable):
        """Set the callback function for handling received messages."""
        self.message_handler = handler
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to server."""
        try:
            logger.info(f"Connecting to server: {self.server_url}")
            self.websocket = await websockets.connect(self.server_url)
            
            # Send handshake
            handshake = {
                "type": "handshake",
                "user_id": self.user_id,
                "client_type": "desktop"
            }
            await self.send_message(handshake)
            
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("✅ Connected to server successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Cleanly disconnect from server."""
        self.running = False
        self.connected = False
        
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        logger.info("Disconnected from server")
    
    async def send_message(self, message: dict):
        """Send a message to the server."""
        if not self.websocket or not self.connected:
            logger.warning("Cannot send message - not connected to server")
            return
            
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message.get('type', 'unknown')}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.connected = False
    
    async def listen_for_messages(self):
        """Listen for messages from server and handle them."""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    logger.debug(f"Received message type: {data.get('type', 'unknown')}")
                    
                    if self.message_handler:
                        await self.message_handler(data)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except ConnectionClosed:
            logger.warning("WebSocket connection closed by server")
            self.connected = False
        except Exception as e:
            logger.error(f"Error listening for messages: {e}")
            self.connected = False
    
    async def start_reconnect_task(self):
        """Start automatic reconnection task."""
        if self.reconnect_task and not self.reconnect_task.done():
            return  # Already running
            
        self.reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """Background task for automatic reconnection."""
        while self.running and not self.connected:
            if self.max_reconnect_attempts > 0 and self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.error("Max reconnection attempts reached")
                break
                
            self.reconnect_attempts += 1
            logger.debug(f"Attempting to reconnect (attempt {self.reconnect_attempts})")
            
            if await self.connect():
                logger.info("Reconnected successfully")
                # Start listening again
                asyncio.create_task(self.listen_for_messages())
                break
            
            # Exponential backoff
            delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 
                       self.max_reconnect_delay)
            logger.debug(f"Waiting {delay}s before next reconnect attempt")
            await asyncio.sleep(delay)
    
    async def start(self):
        """Start the WebSocket manager."""
        self.running = True
        
        if not await self.connect():
            await self.start_reconnect_task()
            return False
        
        # Start listening for messages
        asyncio.create_task(self.listen_for_messages())
        return True
    
    async def stop(self):
        """Stop the WebSocket manager."""
        await self.disconnect()


# Helper functions for common message types
def create_voice_command_message(command: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a voice command message."""
    return {
        "type": "voice_command",
        "command": command,
        "metadata": metadata or {}
    }

def create_clarification_response_message(response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a clarification response message."""
    return {
        "type": "clarification_response", 
        "response": response,
        "context": context or {}
    }

def create_summary_request_message(summary_type: str = "day", date: Optional[str] = None, style: str = "friendly") -> Dict[str, Any]:
    """Create a summary request message."""
    return {
        "type": "request_summary",
        "summary_type": summary_type,
        "date": date,
        "style": style
    }
