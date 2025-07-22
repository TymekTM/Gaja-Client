"""System Tray Manager for GAJA Client Provides system tray icon with status updates and
context menu actions."""

import asyncio
import threading
import time
from pathlib import Path

try:
    import pystray
    from PIL import Image, ImageDraw

    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    pystray = None
    Image = None
    ImageDraw = None

from loguru import logger


class TrayManager:
    """Manages system tray icon for GAJA client."""

    def __init__(self, client_app=None):
        """Initialize tray manager.

        Args:
            client_app: Reference to main ClientApp instance for callbacks
        """
        self.client_app = client_app
        self.icon = None
        self.tray_thread = None
        self.running = False
        self.status = "Starting..."

        if not TRAY_AVAILABLE:
            logger.warning("Pystray not available - system tray will not be shown")

    def create_icon_image(self, color: str = "blue"):
        """Create a simple icon image for the tray.

        Args:
            color: Color of the icon (blue, green, red, orange, gray)

        Returns:
            PIL Image object or None if PIL not available
        """
        if not TRAY_AVAILABLE:
            return None

        try:
            # First try to load existing GAJA icon
            icon_path = Path(__file__).parent.parent.parent / "gaja.ico"
            if icon_path.exists():
                try:
                    image = Image.open(icon_path)
                    # Resize to 32x32 if needed
                    if image.size != (32, 32):
                        image = image.resize((32, 32), Image.Resampling.LANCZOS)

                    # Convert to RGBA for color modification
                    if image.mode != "RGBA":
                        image = image.convert("RGBA")

                    # Apply color tint based on status
                    if color != "blue":  # Only modify if not default
                        pixels = image.load()
                        width, height = image.size

                        # Color mapping for tinting
                        color_multipliers = {
                            "green": (0.7, 1.3, 0.7, 1.0),  # More green
                            "red": (1.3, 0.7, 0.7, 1.0),  # More red
                            "orange": (1.3, 1.1, 0.7, 1.0),  # Orange tint
                            "gray": (0.8, 0.8, 0.8, 1.0),  # Desaturated
                        }

                        multiplier = color_multipliers.get(color, (1.0, 1.0, 1.0, 1.0))

                        for x in range(width):
                            for y in range(height):
                                r, g, b, a = pixels[x, y]
                                if a > 0:  # Only modify non-transparent pixels
                                    new_r = min(255, int(r * multiplier[0]))
                                    new_g = min(255, int(g * multiplier[1]))
                                    new_b = min(255, int(b * multiplier[2]))
                                    pixels[x, y] = (new_r, new_g, new_b, a)

                    return image

                except Exception as e:
                    logger.debug(
                        f"Could not load existing icon, creating simple one: {e}"
                    )

            # Fallback: Create a 32x32 icon
            width, height = 32, 32
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

            # Color mapping
            colors = {
                "blue": (0, 100, 255, 255),  # Connected/Normal
                "green": (0, 200, 0, 255),  # Active/Processing
                "red": (255, 50, 50, 255),  # Error/Disconnected
                "orange": (255, 150, 0, 255),  # Warning/Connecting
                "gray": (128, 128, 128, 255),  # Inactive/Disabled
            }

            fill_color = colors.get(color, colors["blue"])

            # Draw a simple circular icon with "G" for GAJA
            # Circle background
            draw.ellipse([2, 2, width - 2, height - 2], fill=fill_color)

            # White "G" letter
            try:
                # Try to draw text (font might not be available)
                draw.text(
                    (width // 2 - 6, height // 2 - 8), "G", fill=(255, 255, 255, 255)
                )
            except:
                # Fallback: draw a simple geometric shape
                draw.rectangle(
                    [width // 2 - 4, height // 2 - 6, width // 2 + 4, height // 2 + 6],
                    fill=(255, 255, 255, 255),
                )

            return image

        except Exception as e:
            logger.error(f"Error creating tray icon: {e}")
            return None

    def get_status_color(self, status: str) -> str:
        """Get appropriate color for current status.

        Args:
            status: Current status string

        Returns:
            Color name for icon
        """
        status_lower = status.lower()

        if any(word in status_lower for word in ["connected", "ready", "listening"]):
            return "green"  # Active and ready (changed from blue)
        elif any(
            word in status_lower
            for word in ["processing", "recording", "active", "speaking"]
        ):
            return "blue"  # Processing/Active
        elif any(word in status_lower for word in ["error", "failed", "disconnected"]):
            return "red"
        elif any(word in status_lower for word in ["connecting", "starting"]):
            return "orange"
        else:
            return "gray"

    def get_detailed_status_info(self) -> str:
        """Get detailed status information for tooltip."""
        if not self.client_app:
            return "GAJA Assistant - Status Unknown"

        status_parts = []

        # Basic status
        status_parts.append(f"Status: {self.status}")

        # Connection status with more details
        if hasattr(self.client_app, "websocket") and self.client_app.websocket:
            server_url = getattr(self.client_app, "server_url", "Unknown")
            if "localhost" in server_url:
                status_parts.append("🔗 Connected (Local)")
            else:
                status_parts.append("🌐 Connected (Remote)")
        else:
            status_parts.append("🔴 Disconnected")

        # Overlay status
        if hasattr(self.client_app, "overlay_visible"):
            overlay_status = (
                "👁️ Overlay ON" if self.client_app.overlay_visible else "👁️ Overlay OFF"
            )
            status_parts.append(overlay_status)

        # Listening status
        if hasattr(self.client_app, "monitoring_wakeword"):
            if self.client_app.monitoring_wakeword:
                status_parts.append("🎤 Listening")
            else:
                status_parts.append("🎤 Not listening")

        # Keep it under 128 characters total
        tooltip_text = "GAJA Assistant\n" + "\n".join(status_parts)
        if len(tooltip_text) > 125:  # Leave some margin
            # Truncate if too long
            status_parts = status_parts[:3]  # Keep only first 3 items
            tooltip_text = "GAJA Assistant\n" + "\n".join(status_parts)

        return tooltip_text

    def update_status(self, status: str):
        """Update tray icon status and tooltip.

        Args:
            status: New status text
        """
        self.status = status

        if self.icon and self.running:
            try:
                # Update icon color based on status
                color = self.get_status_color(status)
                new_image = self.create_icon_image(color)

                if new_image:
                    self.icon.icon = new_image
                    # Use detailed status info for tooltip
                    detailed_info = self.get_detailed_status_info()
                    self.icon.title = detailed_info

                logger.debug(f"Tray icon updated: {status}")

            except Exception as e:
                logger.error(f"Error updating tray icon: {e}")

        # Also update menu items to reflect current state
        try:
            if self.icon and self.running:
                # Recreate menu with updated overlay toggle text
                new_menu = self.create_menu()
                if new_menu:
                    self.icon.menu = new_menu
        except Exception as e:
            logger.error(f"Error updating tray menu: {e}")

        # Update the client app status as well if available
        if self.client_app and hasattr(self.client_app, "current_status"):
            self.client_app.current_status = status

    def on_quit(self, icon, item):
        """Handle quit action from tray menu."""
        logger.info("Quit requested from system tray")

        # Signal the main app to quit if available
        if self.client_app:
            self.client_app.running = False

            # Send quit signal to overlay process
            try:
                # Get the main event loop from the client app
                loop = None
                if hasattr(self.client_app, "loop") and self.client_app.loop:
                    loop = self.client_app.loop
                else:
                    # Try to get the running loop
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning("No running event loop found for quit signal")
                        loop = None

                if loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self.client_app.send_overlay_update(
                            {"action": "quit", "quit": True}
                        ),
                        loop,
                    )

                    # Wait for the coroutine to complete with timeout
                    try:
                        future.result(timeout=2.0)
                        logger.info("Quit signal sent to overlay")
                    except Exception as e:
                        logger.error(f"Error sending quit signal to overlay: {e}")
                else:
                    logger.warning(
                        "Could not send quit signal to overlay - no event loop"
                    )
            except Exception as e:
                logger.error(f"Error sending quit signal to overlay: {e}")

            # Force quit the overlay process directly
            try:
                if (
                    hasattr(self.client_app, "overlay_process")
                    and self.client_app.overlay_process
                ):
                    import os
                    import signal
                    import subprocess

                    overlay_process = self.client_app.overlay_process
                    if overlay_process.poll() is None:  # Process still running
                        logger.info(
                            f"Force terminating overlay process (PID: {overlay_process.pid})"
                        )
                        try:
                            # Try graceful termination first
                            overlay_process.terminate()
                            overlay_process.wait(timeout=2)
                            logger.info("Overlay process terminated gracefully")
                        except subprocess.TimeoutExpired:
                            # Force kill if graceful termination fails
                            overlay_process.kill()
                            overlay_process.wait(timeout=2)
                            logger.info("Overlay process killed forcefully")
                        except Exception as e:
                            logger.error(f"Error terminating overlay process: {e}")
                            # Last resort: Windows taskkill
                            try:
                                subprocess.run(
                                    ["taskkill", "/IM", "gaja-overlay.exe", "/F"],
                                    capture_output=True,
                                    timeout=5,
                                )
                                logger.info("Overlay process killed with taskkill")
                            except Exception as e2:
                                logger.error(
                                    f"Failed to kill overlay with taskkill: {e2}"
                                )
            except Exception as e:
                logger.error(f"Error handling overlay process termination: {e}")

            # Use a more robust approach to signal shutdown
            import os
            import signal
            import sys

            try:
                # Send interrupt signal to main process
                os.kill(os.getpid(), signal.SIGINT)
                logger.info("Sent interrupt signal to main process")
            except Exception as e:
                logger.error(f"Error sending interrupt signal: {e}")
                # Fallback to system exit
                sys.exit(0)

        # Stop the tray icon last
        self.stop()

    def on_show_status(self, icon, item):
        """Handle show status action from tray menu."""
        logger.info(f"Current status: {self.status}")

        # You could open a status window here or trigger overlay
        if self.client_app and hasattr(self.client_app, "show_overlay"):
            try:
                self.client_app.show_overlay()
            except Exception as e:
                logger.error(f"Error showing overlay: {e}")

    def on_open_settings(self, icon, item):
        """Handle open settings action from tray menu."""
        logger.info("Settings requested from system tray")

        # Direct app-like settings opening instead of overlay
        try:
            # Try to open settings HTML as application window (not browser)
            import os
            import subprocess

            if os.name == "nt":  # Windows
                # Try Edge in app mode first (most app-like)
                try:
                    edge_path = (
                        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                    )
                    if not os.path.exists(edge_path):
                        edge_path = (
                            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
                        )

                    if os.path.exists(edge_path):
                        subprocess.run(
                            [
                                edge_path,
                                "--app=http://localhost:5001/settings.html",
                                "--window-size=800,600",
                                "--disable-web-security",
                                "--disable-features=TranslateUI",
                                "--no-default-browser-check",
                                "--no-first-run",
                                "--disable-background-mode",
                            ],
                            check=True,
                        )
                        logger.info("Opened settings in Edge app mode")
                        return
                except Exception as e:
                    logger.debug(f"Edge app mode failed: {e}")

                # Try Chrome app mode as fallback
                try:
                    chrome_path = (
                        r"C:\Program Files\Google\Chrome\Application\chrome.exe"
                    )
                    if not os.path.exists(chrome_path):
                        chrome_path = r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"

                    if os.path.exists(chrome_path):
                        subprocess.run(
                            [
                                chrome_path,
                                "--app=http://localhost:5001/settings.html",
                                "--window-size=800,600",
                                "--disable-web-security",
                                "--no-first-run",
                                "--disable-background-mode",
                            ],
                            check=True,
                        )
                        logger.info("Opened settings in Chrome app mode")
                        return
                except Exception as e:
                    logger.debug(f"Chrome app mode failed: {e}")

                # Try Firefox app mode
                try:
                    firefox_path = r"C:\Program Files\Mozilla Firefox\firefox.exe"
                    if not os.path.exists(firefox_path):
                        firefox_path = (
                            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
                        )

                    if os.path.exists(firefox_path):
                        subprocess.run(
                            [
                                firefox_path,
                                "--new-window",
                                "--width=800",
                                "--height=600",
                                "http://localhost:5001/settings.html",
                            ],
                            check=True,
                        )
                        logger.info("Opened settings in Firefox")
                        return
                except Exception as e:
                    logger.debug(f"Firefox failed: {e}")

            # Final fallback - regular browser
            import webbrowser

            webbrowser.open("http://localhost:5001/settings.html")
            logger.info("Opened settings via default browser")

        except Exception as e:
            logger.error(f"Could not open settings interface: {e}")

        # Also try to send overlay update for settings if client app is available
        if self.client_app:
            try:
                # Add settings request to command queue
                command = {"type": "open_settings"}
                self.client_app.command_queue.put(command)
                logger.info("Added settings request to command queue")
            except Exception as e:
                logger.error(f"Failed to add settings request to queue: {e}")

    def on_connect_to_server(self, icon, item):
        """Handle connect to server action from tray menu."""
        logger.info("Connect to server requested from system tray")

        if self.client_app:
            try:
                # Schedule reconnection attempt
                if (
                    hasattr(self.client_app, "connect_to_server")
                    and self.client_app.running
                ):
                    import asyncio
                    import threading

                    def reconnect():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self.client_app.connect_to_server())
                            self.client_app.update_status("Connected")
                            logger.info("Reconnection successful")
                        except Exception as e:
                            logger.error(f"Reconnection failed: {e}")
                            self.client_app.update_status("Connection Failed")

                    # Run reconnection in separate thread
                    threading.Thread(target=reconnect, daemon=True).start()
                    self.client_app.update_status("Connecting...")

            except Exception as e:
                logger.error(f"Error during reconnection attempt: {e}")

    def on_restart_client(self, icon, item):
        """Handle restart client action from tray menu."""
        logger.info("Client restart requested from system tray")

        if self.client_app:
            try:
                # Graceful restart - signal the main app
                self.client_app.update_status("Restarting...")

                # Stop current client and restart
                import os
                import subprocess
                import sys

                # Get current script path
                script_path = sys.argv[0]

                # Schedule restart after cleanup
                def restart_after_cleanup():
                    import time

                    time.sleep(2)  # Wait for cleanup

                    try:
                        # Start new instance
                        subprocess.Popen(
                            [sys.executable, script_path],
                            cwd=os.getcwd(),
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                            if os.name == "nt"
                            else 0,
                        )
                        logger.info("New client instance started")
                    except Exception as e:
                        logger.error(f"Failed to restart client: {e}")

                import threading

                threading.Thread(target=restart_after_cleanup, daemon=True).start()

                # Stop current instance
                self.client_app.running = False

            except Exception as e:
                logger.error(f"Error during restart: {e}")

    def on_about(self, icon, item):
        """Handle about action from tray menu."""
        logger.info("About dialog requested from system tray")

        try:
            # Show info about GAJA
            about_text = (
                """
GAJA Assistant Client v1.0
AI-powered voice assistant

Features:
- Voice wake word detection
- Speech recognition (Whisper)
- AI conversation (OpenAI/Claude)
- Text-to-speech
- Overlay interface
- System tray integration

Status: """
                + self.status
                + """

© 2024 GAJA Assistant
            """.strip()
            )

            logger.info(f"GAJA Assistant Info: {about_text}")

            # Try to show in overlay if available
            if self.client_app and hasattr(self.client_app, "show_overlay"):
                self.client_app.show_overlay()

        except Exception as e:
            logger.error(f"Error showing about info: {e}")

    def on_toggle_overlay(self, icon, item):
        """Handle toggle overlay action from tray menu."""
        if self.client_app:
            try:
                # Get the main event loop from the client app
                loop = None
                if hasattr(self.client_app, "loop") and self.client_app.loop:
                    loop = self.client_app.loop
                else:
                    # Try to get the running loop
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        logger.warning("No running event loop found for overlay toggle")
                        return

                # Toggle overlay visibility
                if (
                    hasattr(self.client_app, "overlay_visible")
                    and self.client_app.overlay_visible
                ):
                    logger.info("Hiding overlay from tray")
                    # Call hide_overlay method to properly hide overlay
                    future = asyncio.run_coroutine_threadsafe(
                        self.client_app.hide_overlay(), loop
                    )

                    # Wait for the coroutine to complete with timeout
                    try:
                        future.result(timeout=2.0)
                        if hasattr(self.client_app, "update_status"):
                            self.client_app.update_status("Overlay Hidden")
                    except Exception as e:
                        logger.error(f"Error hiding overlay: {e}")
                else:
                    logger.info("Showing overlay from tray")
                    # Call show_overlay method to properly show overlay
                    future = asyncio.run_coroutine_threadsafe(
                        self.client_app.show_overlay(), loop
                    )

                    # Wait for the coroutine to complete with timeout
                    try:
                        future.result(timeout=2.0)
                        if hasattr(self.client_app, "update_status"):
                            self.client_app.update_status("Overlay Shown")
                    except Exception as e:
                        logger.error(f"Error showing overlay: {e}")
            except Exception as e:
                logger.error(f"Error toggling overlay: {e}")

    def create_menu(self):
        """Create context menu for tray icon."""
        if not TRAY_AVAILABLE:
            return None

        try:
            # Dynamic overlay toggle text based on current state
            overlay_text = (
                "Disable Overlay"
                if (
                    self.client_app
                    and hasattr(self.client_app, "overlay_visible")
                    and self.client_app.overlay_visible
                )
                else "Enable Overlay"
            )

            # Simplified menu with only essential options
            menu_items = [
                pystray.MenuItem("Settings", self.on_open_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(overlay_text, self.on_toggle_overlay),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.on_quit),
            ]

            return pystray.Menu(*menu_items)

        except Exception as e:
            logger.error(f"Error creating tray menu: {e}")
            return None

    def start(self):
        """Start the system tray icon."""
        if not TRAY_AVAILABLE:
            logger.warning("Cannot start system tray - pystray not available")
            return False

        if self.running:
            logger.warning("Tray manager already running")
            return True

        try:
            # Create initial icon
            icon_image = self.create_icon_image("orange")  # Starting state
            if not icon_image:
                logger.error("Failed to create tray icon image")
                return False

            # Create menu
            menu = self.create_menu()
            if not menu:
                logger.error("Failed to create tray menu")
                return False

            # Create tray icon
            self.icon = pystray.Icon(
                "gaja_assistant", icon_image, "GAJA Assistant - Starting...", menu
            )

            # Start tray in separate thread
            self.running = True
            self.tray_thread = threading.Thread(target=self._run_tray, daemon=True)
            self.tray_thread.start()

            logger.info("System tray icon started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting system tray: {e}")
            return False

    def _run_tray(self):
        """Internal method to run tray icon (called in separate thread)."""
        try:
            self.icon.run()
        except Exception as e:
            logger.error(f"Error running tray icon: {e}")
        finally:
            self.running = False

    def stop(self):
        """Stop the system tray icon."""
        if not self.running:
            return

        try:
            self.running = False

            if self.icon:
                self.icon.stop()
                self.icon = None

            # Wait for thread to finish (with timeout) - but only if not current thread
            if self.tray_thread and self.tray_thread.is_alive():
                if threading.current_thread() != self.tray_thread:
                    self.tray_thread.join(timeout=2.0)
                else:
                    # We're being called from the tray thread itself, just mark as stopped
                    pass

            logger.info("System tray icon stopped")

        except Exception as e:
            logger.error(f"Error stopping system tray: {e}")

    def is_running(self) -> bool:
        """Check if tray manager is running.

        Returns:
            True if tray is running, False otherwise
        """
        return self.running and self.icon is not None


# Example usage and testing
if __name__ == "__main__":
    # Simple test
    tray = TrayManager()

    if tray.start():
        print("Tray started - right-click the icon in system tray")
        try:
            # Simulate status updates
            time.sleep(2)
            tray.update_status("Connected")

            time.sleep(2)
            tray.update_status("Processing")

            time.sleep(2)
            tray.update_status("Ready")

            # Keep running until interrupted
            while tray.is_running():
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping tray...")
            tray.stop()
    else:
        print("Failed to start system tray")
