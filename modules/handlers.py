"""Message handlers for different server message types."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MessageHandlers:
    """Handles different types of messages from the server."""
    
    def __init__(self, client_app):
        """Initialize with reference to main client app."""
        self.client = client_app
    
    async def handle_ai_response(self, data: Dict[str, Any]):
        """Handle AI response from server."""
        message_data = data.get("data", {})
        response = message_data.get("response", "")
        tts_audio_b64 = message_data.get("tts_audio")
        
        logger.info(f"AI Response received: {response}")
        
        if not response:
            logger.warning("Received empty AI response")
            await self._reset_client_state()
            return
        
        try:
            # Parse response (may be JSON string)
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response
            
            text = response_data.get("text", "")
            if not text:
                logger.warning("No text in AI response")
                await self._reset_client_state()
                return
            
            logger.info(f"AI text response: {text}")
            
            # Update overlay
            self.client.last_tts_text = text
            self.client.update_status("mówię")
            await self.client.show_overlay()
            
            # Play TTS response
            self.client.tts_playing = True
            try:
                if tts_audio_b64:
                    # Get volume from server TTS config
                    tts_config = data.get("tts_config", {})
                    volume = tts_config.get("volume", 1.0)
                    await self.client.play_server_tts(tts_audio_b64, volume)
                else:
                    # Fallback to local TTS
                    await self.client.speak_text_local(text)
            finally:
                await self._reset_client_state()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {e}")
            await self._reset_client_state()
        except Exception as e:
            logger.error(f"Error handling AI response: {e}")
            await self._reset_client_state()
    
    async def handle_clarification_request(self, data: Dict[str, Any]):
        """Handle clarification request from server."""
        logger.info("Received clarification request from server")
        
        clarification_data = data.get("data", {})
        question = clarification_data.get("question", "")
        tts_audio_b64 = clarification_data.get("tts_audio")
        
        if not question:
            logger.warning("Empty clarification question received")
            return
        
        logger.info(f"Clarification question: {question}")
        
        # Update overlay
        self.client.last_tts_text = question
        self.client.update_status("wyjaśnienie")
        await self.client.show_overlay()
        
        # Play clarification TTS
        self.client.tts_playing = True
        try:
            if tts_audio_b64:
                tts_config = data.get("tts_config", {})
                volume = tts_config.get("volume", 1.0)
                await self.client.play_server_tts(tts_audio_b64, volume)
            else:
                await self.client.speak_text_local(question)
        finally:
            self.client.tts_playing = False
            await self.client.restart_recording_after_clarification()
        
        # Update status for user response
        self.client.update_status("czekam na odpowiedź")
    
    async def handle_daily_briefing(self, data: Dict[str, Any]):
        """Handle daily briefing from server."""
        briefing_text = data.get("text", "")
        logger.info(f"Daily briefing received: {briefing_text[:100]}...")
        
        if not briefing_text:
            logger.warning("Empty daily briefing received")
            return
        
        # Update overlay and speak briefing
        self.client.update_status("Daily Briefing")
        await self.client.show_overlay()
        
        if self.client.tts:
            try:
                self.client.tts_playing = True
                logger.info("Starting TTS for daily briefing...")
                await self.client.tts.speak(briefing_text)
                logger.info("Daily briefing spoken successfully")
            except Exception as e:
                logger.error(f"Error speaking daily briefing: {e}")
                # Try to reinitialize TTS if it failed
                try:
                    from audio_modules.tts_module import TTSModule
                    self.client.tts = TTSModule()
                    await self.client.tts.speak(briefing_text)
                    logger.info("Daily briefing spoken after TTS reinit")
                except Exception as e2:
                    logger.error(f"Failed to reinitialize TTS: {e2}")
            finally:
                self.client.tts_playing = False
                await self.client.hide_overlay()
                self.client.update_status("Listening...")
    
    async def handle_summary_response(self, data: Dict[str, Any]):
        """Handle summary response from server."""
        message_type = data.get("type")
        summary_data = data.get("data", {})
        
        logger.debug(f"Summary response received: {message_type}")
        
        summary_text = self._format_summary_for_speech(message_type or "unknown", summary_data)
        
        if summary_text:
            # Update overlay and speak summary
            self.client.last_tts_text = summary_text
            self.client.update_status("podsumowanie")
            await self.client.show_overlay()
            
            if self.client.tts:
                try:
                    self.client.tts_playing = True
                    await self.client.tts.speak(summary_text)
                    logger.info(f"Summary spoken: {message_type}")
                except Exception as e:
                    logger.error(f"Error speaking summary: {e}")
                finally:
                    self.client.tts_playing = False
                    await self.client.hide_overlay()
                    self.client.update_status("słucham")
    
    async def handle_handshake(self, data: Dict[str, Any]):
        """Handle handshake response from server."""
        handshake_data = data.get("data", {})
        if handshake_data.get("success", False):
            logger.info("Handshake successful with server")
        else:
            logger.warning("Handshake failed with server")
    
    async def handle_startup_briefing(self, briefing: Dict[str, Any]):
        """Handle startup briefing."""
        briefing_content = briefing.get("briefing", "")
        if briefing_content:
            logger.info(f"Startup briefing received: {briefing_content[:100]}...")
            # Similar to daily briefing handling
            await self.handle_daily_briefing({"text": briefing_content})
    
    async def handle_day_summary(self, summary: Dict[str, Any]):
        """Handle day summary."""
        summary_type = summary.get("type", "day_summary")
        logger.info(f"Day summary received: {summary_type}")
        # Process like other summaries
        await self.handle_summary_response({"type": summary_type, "data": summary})
    
    async def handle_proactive_notifications(self, data: Dict[str, Any]):
        """Handle proactive notifications from server."""
        notifications = data.get("notifications", [])
        logger.info(f"Received {len(notifications)} proactive notifications")
        
        # For now, just log them - could be extended to show in overlay
        for notification in notifications:
            logger.info(f"Notification: {notification.get('message', 'No message')}")
    
    def _format_summary_for_speech(self, summary_type: str, summary_data: Dict[str, Any]) -> str:
        """Format summary data into speech-friendly text."""
        try:
            if summary_type == "day_summary":
                if summary_data.get("success"):
                    apps = summary_data.get("top_applications", [])
                    total_time = summary_data.get("total_active_time", 0)
                    productivity_score = summary_data.get("productivity_score", 0)
                    
                    if apps:
                        top_app = apps[0]
                        app_name = top_app.get("name", "Unknown")
                        app_time = top_app.get("total_time", 0)
                        
                        return (
                            f"Podsumowanie dnia: pracowałeś łącznie {total_time:.1f} godzin. "
                            f"Najczęściej używana aplikacja to {app_name} - {app_time:.1f} godzin. "
                            f"Produktywność: {productivity_score:.0%}."
                        )
                    else:
                        return f"Podsumowanie dnia: łącznie {total_time:.1f} godzin pracy."
            
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
                    recommendations = summary_data.get("insights", {}).get("recommendations", [])
                    if recommendations:
                        return f"Wglądy w zachowania: {recommendations[0]}"
                    else:
                        return "Analiza zachowań została zakończona."
            
            elif summary_type == "routine_insights":
                if summary_data.get("success"):
                    recommendations = summary_data.get("insights", {}).get("recommendations", [])
                    if recommendations:
                        return f"Analiza rutyn: {recommendations[0]}"
                    else:
                        return "Analiza rutyn została zakończona."
            
            return f"Otrzymano {summary_type}, ale nie mogę go odczytać."
            
        except Exception as e:
            logger.error(f"Error formatting summary for speech: {e}")
            return "Wystąpił błąd podczas formatowania podsumowania."
    
    async def _reset_client_state(self):
        """Reset client state after completing response."""
        self.client.tts_playing = False
        self.client.wake_word_detected = False
        self.client.recording_command = False
        await self.client.hide_overlay()
        self.client.update_status("słucham")
