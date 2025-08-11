"""Unified TTS playback module for both server and local TTS."""

import asyncio
import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TTSPlayback:
    """Unified TTS playback handler supporting both server-provided audio and local TTS."""
    
    def __init__(self, local_tts_module=None):
        """Initialize with optional local TTS module."""
        self.local_tts = local_tts_module
        self.is_playing = False
    
    async def play_server_tts(self, tts_audio_b64: str, volume: float = 1.0) -> bool:
        """Play server-provided TTS audio using ffplay.
        
        Returns True if successful, False otherwise.
        """
        if not tts_audio_b64:
            return False
            
        try:
            # Decode audio from base64
            audio_data = base64.b64decode(tts_audio_b64)
            logger.debug(f"Playing server TTS audio: {len(audio_data)} bytes")
            
            # Use ffplay to play audio directly from stdin
            ffplay_volume = int(volume * 200)  # Convert to ffplay volume scale
            process = await asyncio.create_subprocess_exec(
                "ffplay",
                "-nodisp",
                "-autoexit", 
                "-loglevel", "quiet",
                "-volume", str(ffplay_volume),
                "-i", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            # Write audio data to ffplay stdin
            stdout, stderr = await process.communicate(input=audio_data)
            
            if process.returncode == 0:
                logger.debug("Server TTS audio played successfully via ffplay")
                return True
            else:
                logger.warning(f"ffplay returned code {process.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"Error playing server TTS audio: {e}")
            return False

    async def speak_text_local(self, text: str) -> bool:
        """Speak text using local TTS module.
        
        Returns True if successful, False otherwise.
        """
        if not self.local_tts:
            logger.warning("Local TTS not available")
            return False
            
        try:
            await self.local_tts.speak(text)
            logger.debug("Local TTS completed successfully")
            return True
        except Exception as e:
            logger.error(f"Local TTS error: {e}")
            return False

    async def play_audio(self, text: str, tts_audio_b64: Optional[str] = None, volume: float = 1.0) -> bool:
        """Play audio with automatic fallback: server TTS -> local TTS.
        
        Args:
            text: Text to speak (fallback if server audio fails)
            tts_audio_b64: Base64 encoded server audio (optional)
            volume: Volume level (0.0 to 1.0+)
            
        Returns True if any playback method succeeded.
        """
        self.is_playing = True
        try:
            # Try server TTS first if available
            if tts_audio_b64:
                if await self.play_server_tts(tts_audio_b64, volume):
                    return True
                logger.info("Server TTS failed, falling back to local TTS")
            
            # Fallback to local TTS
            return await self.speak_text_local(text)
            
        finally:
            self.is_playing = False

    def stop(self):
        """Stop any ongoing TTS playback."""
        # TODO: Implement process termination for ffplay if needed
        self.is_playing = False
        if hasattr(self.local_tts, 'cancel'):
            try:
                self.local_tts.cancel()
            except Exception as e:
                logger.debug(f"Could not cancel local TTS: {e}")
