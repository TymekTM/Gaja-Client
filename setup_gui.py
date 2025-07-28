"""Simple GUI for GAJA Client setup and configuration.

Provides an easy-to-use interface for initial setup and ongoing configuration
of the GAJA Client, including microphone selection, voice settings, and
other important options.
"""

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, List, Optional

from loguru import logger

# Try to import sounddevice for audio device detection
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None

# Import settings manager
try:
    from modules.settings_manager import SettingsManager
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    SettingsManager = None


class AudioDeviceFrame(ttk.LabelFrame):
    """Frame for audio device configuration."""
    
    def __init__(self, parent, settings_manager=None):
        super().__init__(parent, text="Audio Devices", padding=10)
        self.settings_manager = settings_manager
        self.devices = {"input_devices": [], "output_devices": []}
        
        self.create_widgets()
        self.refresh_devices()
    
    def create_widgets(self):
        """Create audio device selection widgets."""
        # Input device selection
        ttk.Label(self, text="Microphone (Input Device):").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        
        # Frame for input device selection with search
        input_frame = ttk.Frame(self)
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Search entry for input devices
        self.input_search_var = tk.StringVar()
        self.input_search_var.trace("w", self.filter_input_devices)
        self.input_search = ttk.Entry(
            input_frame, 
            textvariable=self.input_search_var,
            font=("Segoe UI", 9)
        )
        self.input_search.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        ttk.Button(input_frame, text="🔄", width=3, command=self.refresh_devices).grid(
            row=0, column=1
        )
        
        # Combobox for input device selection
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(
            input_frame,
            textvariable=self.input_device_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=60
        )
        self.input_device_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.input_device_combo.bind("<<ComboboxSelected>>", self.on_input_device_change)
        
        # Test microphone button
        ttk.Button(
            self, text="Test Microphone", command=self.test_microphone
        ).grid(row=2, column=0, sticky="w", pady=(0, 10))
        
        # Output device selection
        ttk.Label(self, text="Speakers (Output Device):").grid(
            row=3, column=0, sticky="w", pady=(0, 5)
        )
        
        # Frame for output device selection
        output_frame = ttk.Frame(self)
        output_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        
        # Search entry for output devices
        self.output_search_var = tk.StringVar()
        self.output_search_var.trace("w", self.filter_output_devices)
        self.output_search = ttk.Entry(
            output_frame,
            textvariable=self.output_search_var,
            font=("Segoe UI", 9)
        )
        self.output_search.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        # Combobox for output device selection
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(
            output_frame,
            textvariable=self.output_device_var,
            state="readonly",
            font=("Segoe UI", 9),
            width=60
        )
        self.output_device_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.output_device_combo.bind("<<ComboboxSelected>>", self.on_output_device_change)
        
        # Test speakers button
        ttk.Button(
            self, text="Test Speakers", command=self.test_speakers
        ).grid(row=5, column=0, sticky="w")
        
        # Configure column weights
        self.columnconfigure(0, weight=1)
    
    def refresh_devices(self):
        """Refresh the list of available audio devices."""
        try:
            if self.settings_manager and SETTINGS_AVAILABLE:
                self.devices = self.settings_manager.get_audio_devices()
            elif AUDIO_AVAILABLE:
                self.devices = self._get_audio_devices_direct()
            else:
                self.devices = {
                    "input_devices": [{"id": "default", "name": "Default Input Device", "is_default": True}],
                    "output_devices": [{"id": "default", "name": "Default Output Device", "is_default": True}]
                }
            
            self.update_device_combos()
            
        except Exception as e:
            logger.error(f"Error refreshing audio devices: {e}")
            messagebox.showerror("Error", f"Failed to refresh audio devices: {e}")
    
    def _get_audio_devices_direct(self):
        """Get audio devices directly using sounddevice."""
        try:
            if not AUDIO_AVAILABLE or sd is None:
                return {
                    "input_devices": [{"id": "default", "name": "Audio not available", "is_default": True}],
                    "output_devices": [{"id": "default", "name": "Audio not available", "is_default": True}]
                }
            
            devices = sd.query_devices()
            input_devices = []
            output_devices = []
            
            default_input = sd.default.device[0] if sd.default.device else None
            default_output = sd.default.device[1] if sd.default.device else None
            
            for i, device in enumerate(devices):
                device_info = {
                    "id": str(i),
                    "name": device.get("name", f"Device {i}"),
                    "is_default": False,
                }
                
                # Check for input capabilities
                if device.get("max_input_channels", 0) > 0:
                    device_info_input = device_info.copy()
                    device_info_input["is_default"] = i == default_input
                    input_devices.append(device_info_input)
                
                # Check for output capabilities
                if device.get("max_output_channels", 0) > 0:
                    device_info_output = device_info.copy()
                    device_info_output["is_default"] = i == default_output
                    output_devices.append(device_info_output)
            
            return {"input_devices": input_devices, "output_devices": output_devices}
            
        except Exception as e:
            logger.error(f"Error getting audio devices directly: {e}")
            return {
                "input_devices": [{"id": "default", "name": "Error getting devices", "is_default": True}],
                "output_devices": [{"id": "default", "name": "Error getting devices", "is_default": True}]
            }
    
    def update_device_combos(self):
        """Update device comboboxes with current device list."""
        # Update input devices
        input_values = []
        for device in self.devices["input_devices"]:
            display_name = f"{device['name']}"
            if device.get("is_default", False):
                display_name += " (Default)"
            input_values.append(display_name)
        
        self.input_device_combo["values"] = input_values
        self.all_input_devices = input_values.copy()  # Store for filtering
        
        # Update output devices
        output_values = []
        for device in self.devices["output_devices"]:
            display_name = f"{device['name']}"
            if device.get("is_default", False):
                display_name += " (Default)"
            output_values.append(display_name)
        
        self.output_device_combo["values"] = output_values
        self.all_output_devices = output_values.copy()  # Store for filtering
        
        # Select default devices if none selected
        if not self.input_device_var.get() and input_values:
            # Try to find default device
            for i, device in enumerate(self.devices["input_devices"]):
                if device.get("is_default", False):
                    self.input_device_combo.current(i)
                    break
            else:
                # No default found, select first
                self.input_device_combo.current(0)
        
        if not self.output_device_var.get() and output_values:
            # Try to find default device
            for i, device in enumerate(self.devices["output_devices"]):
                if device.get("is_default", False):
                    self.output_device_combo.current(i)
                    break
            else:
                # No default found, select first
                self.output_device_combo.current(0)
    
    def filter_input_devices(self, *args):
        """Filter input devices based on search text."""
        search_text = self.input_search_var.get().lower()
        if not search_text:
            # Show all devices
            self.input_device_combo["values"] = self.all_input_devices
        else:
            # Filter devices
            filtered = [device for device in self.all_input_devices 
                       if search_text in device.lower()]
            self.input_device_combo["values"] = filtered
    
    def filter_output_devices(self, *args):
        """Filter output devices based on search text."""
        search_text = self.output_search_var.get().lower()
        if not search_text:
            # Show all devices
            self.output_device_combo["values"] = self.all_output_devices
        else:
            # Filter devices
            filtered = [device for device in self.all_output_devices 
                       if search_text in device.lower()]
            self.output_device_combo["values"] = filtered
    
    def on_input_device_change(self, event=None):
        """Handle input device selection change."""
        selected = self.input_device_var.get()
        if selected:
            # Find device ID
            display_name = selected.replace(" (Default)", "")
            for device in self.devices["input_devices"]:
                if device["name"] == display_name:
                    logger.info(f"Selected input device: {device['name']} (ID: {device['id']})")
                    break
    
    def on_output_device_change(self, event=None):
        """Handle output device selection change."""
        selected = self.output_device_var.get()
        if selected:
            # Find device ID
            display_name = selected.replace(" (Default)", "")
            for device in self.devices["output_devices"]:
                if device["name"] == display_name:
                    logger.info(f"Selected output device: {device['name']} (ID: {device['id']})")
                    break
    
    def test_microphone(self):
        """Test the selected microphone."""
        try:
            if not AUDIO_AVAILABLE:
                messagebox.showwarning("Warning", "Audio testing not available - sounddevice not installed")
                return
            
            selected = self.input_device_var.get()
            if not selected:
                messagebox.showwarning("Warning", "Please select a microphone first")
                return
            
            # Find device ID
            device_id = None
            display_name = selected.replace(" (Default)", "")
            for device in self.devices["input_devices"]:
                if device["name"] == display_name:
                    device_id = int(device["id"]) if device["id"] != "default" else None
                    break
            
            # Run test in separate thread
            threading.Thread(target=self._test_microphone_thread, args=(device_id,), daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error testing microphone: {e}")
            messagebox.showerror("Error", f"Failed to test microphone: {e}")
    
    def _test_microphone_thread(self, device_id):
        """Test microphone in separate thread."""
        try:
            if not AUDIO_AVAILABLE or sd is None:
                self.after_idle(lambda: messagebox.showwarning("Test Error", "Audio testing not available - sounddevice not installed"))
                return
            
            import numpy as np
            
            duration = 2  # seconds
            sample_rate = 16000
            
            # Show progress dialog
            self.after_idle(lambda: messagebox.showinfo("Testing", f"Recording for {duration} seconds... Speak into the microphone!"))
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                device=device_id,
            )
            sd.wait()
            
            # Analyze audio
            max_amplitude = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            
            # Show results
            if max_amplitude < 0.001:
                result = f"Very low signal detected (max: {max_amplitude:.4f})\nCheck microphone connection."
                self.after_idle(lambda: messagebox.showwarning("Test Result", result))
            elif max_amplitude > 0.1:
                result = f"Good signal detected! (max: {max_amplitude:.4f})\nMicrophone is working well."
                self.after_idle(lambda: messagebox.showinfo("Test Result", result))
            else:
                result = f"Low signal detected (max: {max_amplitude:.4f})\nMicrophone might need adjustment."
                self.after_idle(lambda: messagebox.showinfo("Test Result", result))
            
        except Exception as e:
            logger.error(f"Error in microphone test: {e}")
            self.after_idle(lambda: messagebox.showerror("Test Error", f"Microphone test failed: {e}"))
    
    def test_speakers(self):
        """Test the selected speakers."""
        try:
            if not AUDIO_AVAILABLE:
                messagebox.showwarning("Warning", "Audio testing not available - sounddevice not installed")
                return
            
            # Run test in separate thread
            threading.Thread(target=self._test_speakers_thread, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error testing speakers: {e}")
            messagebox.showerror("Error", f"Failed to test speakers: {e}")
    
    def _test_speakers_thread(self):
        """Test speakers in separate thread."""
        try:
            if not AUDIO_AVAILABLE or sd is None:
                self.after_idle(lambda: messagebox.showwarning("Test Error", "Audio testing not available - sounddevice not installed"))
                return
            
            import numpy as np
            
            # Generate test tone
            duration = 1.0  # seconds
            sample_rate = 44100
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = 0.3 * np.sin(frequency * 2 * np.pi * t)
            
            # Show info
            self.after_idle(lambda: messagebox.showinfo("Testing", "Playing test tone... You should hear a beep!"))
            
            # Play tone
            sd.play(tone, sample_rate)
            sd.wait()
            
            # Ask if user heard it
            def ask_result():
                result = messagebox.askyesno("Test Result", "Did you hear the test tone?")
                if result:
                    messagebox.showinfo("Success", "Speakers are working correctly!")
                else:
                    messagebox.showwarning("Issue", "Check speaker connection and volume settings.")
            
            self.after_idle(ask_result)
            
        except Exception as e:
            logger.error(f"Error in speaker test: {e}")
            self.after_idle(lambda: messagebox.showerror("Test Error", f"Speaker test failed: {e}"))
    
    def get_selected_devices(self):
        """Get currently selected devices."""
        input_device_id = None
        output_device_id = None
        
        # Get input device ID
        selected_input = self.input_device_var.get()
        if selected_input:
            display_name = selected_input.replace(" (Default)", "")
            for device in self.devices["input_devices"]:
                if device["name"] == display_name:
                    input_device_id = device["id"]
                    if input_device_id != "default":
                        input_device_id = int(input_device_id)
                    break
        
        # Get output device ID
        selected_output = self.output_device_var.get()
        if selected_output:
            display_name = selected_output.replace(" (Default)", "")
            for device in self.devices["output_devices"]:
                if device["name"] == display_name:
                    output_device_id = device["id"]
                    if output_device_id != "default":
                        output_device_id = int(output_device_id)
                    break
        
        return {
            "input_device": input_device_id,
            "output_device": output_device_id
        }
    
    def set_selected_devices(self, input_device_id=None, output_device_id=None):
        """Set selected devices by ID."""
        # Set input device
        if input_device_id is not None:
            for i, device in enumerate(self.devices["input_devices"]):
                if str(device["id"]) == str(input_device_id):
                    self.input_device_combo.current(i)
                    break
        
        # Set output device
        if output_device_id is not None:
            for i, device in enumerate(self.devices["output_devices"]):
                if str(device["id"]) == str(output_device_id):
                    self.output_device_combo.current(i)
                    break


class VoiceSettingsFrame(ttk.LabelFrame):
    """Frame for voice and wake word settings."""
    
    def __init__(self, parent):
        super().__init__(parent, text="Voice Settings", padding=10)
        self.create_widgets()
    
    def create_widgets(self):
        """Create voice settings widgets."""
        # Wake word settings
        ttk.Label(self, text="Wake Word:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.wake_word_var = tk.StringVar(value="gaja")
        wake_word_entry = ttk.Entry(self, textvariable=self.wake_word_var, width=20)
        wake_word_entry.grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Sensitivity settings
        ttk.Label(self, text="Sensitivity:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.sensitivity_var = tk.DoubleVar(value=0.65)
        sensitivity_scale = ttk.Scale(
            self, from_=0.1, to=1.0, variable=self.sensitivity_var, orient="horizontal", length=200
        )
        sensitivity_scale.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Sensitivity value label
        self.sensitivity_label = ttk.Label(self, text="0.65")
        self.sensitivity_label.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Update sensitivity label when scale changes
        self.sensitivity_var.trace("w", self.update_sensitivity_label)
        
        # Language settings
        ttk.Label(self, text="Language:").grid(row=2, column=0, sticky="w", pady=(0, 5))
        self.language_var = tk.StringVar(value="pl-PL")
        language_combo = ttk.Combobox(
            self,
            textvariable=self.language_var,
            values=["pl-PL", "en-US", "de-DE", "fr-FR", "es-ES"],
            state="readonly",
            width=17
        )
        language_combo.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Whisper model settings
        ttk.Label(self, text="Whisper Model:").grid(row=3, column=0, sticky="w", pady=(0, 5))
        self.whisper_model_var = tk.StringVar(value="base")
        whisper_combo = ttk.Combobox(
            self,
            textvariable=self.whisper_model_var,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
            width=17
        )
        whisper_combo.grid(row=3, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Configure column weights
        self.columnconfigure(1, weight=1)
    
    def update_sensitivity_label(self, *args):
        """Update sensitivity value label."""
        value = self.sensitivity_var.get()
        self.sensitivity_label.config(text=f"{value:.2f}")
    
    def get_voice_settings(self):
        """Get current voice settings."""
        return {
            "wake_word": self.wake_word_var.get(),
            "sensitivity": self.sensitivity_var.get(),
            "language": self.language_var.get(),
            "whisper_model": self.whisper_model_var.get()
        }
    
    def set_voice_settings(self, settings):
        """Set voice settings."""
        if "wake_word" in settings:
            self.wake_word_var.set(settings["wake_word"])
        if "sensitivity" in settings:
            self.sensitivity_var.set(settings["sensitivity"])
        if "language" in settings:
            self.language_var.set(settings["language"])
        if "whisper_model" in settings:
            self.whisper_model_var.set(settings["whisper_model"])


class ServerSettingsFrame(ttk.LabelFrame):
    """Frame for server connection settings."""
    
    def __init__(self, parent):
        super().__init__(parent, text="Server Connection", padding=10)
        self.create_widgets()
    
    def create_widgets(self):
        """Create server settings widgets."""
        # Server URL
        ttk.Label(self, text="Server URL:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.server_url_var = tk.StringVar(value="ws://localhost:8001/ws/client1")
        server_url_entry = ttk.Entry(self, textvariable=self.server_url_var, width=50)
        server_url_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(0, 5))
        
        # Test connection button
        ttk.Button(self, text="Test Connection", command=self.test_connection).grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        
        # Connection status
        self.status_var = tk.StringVar(value="Not tested")
        self.status_label = ttk.Label(self, textvariable=self.status_var, foreground="gray")
        self.status_label.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
        
        # Configure column weights
        self.columnconfigure(1, weight=1)
    
    def test_connection(self):
        """Test connection to the server."""
        server_url = self.server_url_var.get()
        if not server_url:
            messagebox.showwarning("Warning", "Please enter a server URL")
            return
        
        # Update status
        self.status_var.set("Testing...")
        self.status_label.config(foreground="orange")
        
        # Test in separate thread
        threading.Thread(target=self._test_connection_thread, args=(server_url,), daemon=True).start()
    
    def _test_connection_thread(self, server_url):
        """Test connection in separate thread."""
        try:
            import asyncio
            import websockets
            
            async def test_websocket():
                try:
                    # Try to connect with short timeout
                    async with websockets.connect(server_url, ping_timeout=5, ping_interval=5) as websocket:
                        # Send a simple ping
                        await websocket.send('{"type": "ping"}')
                        # Wait for response or timeout
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        return True, "Connected successfully"
                except asyncio.TimeoutError:
                    return False, "Connection timeout"
                except websockets.exceptions.InvalidURI:
                    return False, "Invalid server URL format"
                except OSError as e:
                    if "refused" in str(e).lower():
                        return False, "Connection refused - server not running"
                    else:
                        return False, f"Connection failed: {str(e)}"
                except Exception as e:
                    return False, f"Connection failed: {str(e)}"
            
            # Run the async test
            try:
                success, message = asyncio.run(test_websocket())
            except Exception as e:
                success, message = False, f"Test failed: {str(e)}"
            
            # Update UI in main thread
            def update_status():
                self.status_var.set(message)
                if success:
                    self.status_label.config(foreground="green")
                    messagebox.showinfo("Connection Test", "✅ Connection successful!")
                else:
                    self.status_label.config(foreground="red")
                    messagebox.showerror("Connection Test", f"❌ {message}")
            
            self.after_idle(update_status)
            
        except Exception as e:
            logger.error(f"Error testing connection: {e}")
            def update_error():
                self.status_var.set("Test error")
                self.status_label.config(foreground="red")
                messagebox.showerror("Error", f"Connection test failed: {e}")
            
            self.after_idle(update_error)
    
    def get_server_settings(self):
        """Get current server settings."""
        return {
            "server_url": self.server_url_var.get()
        }
    
    def set_server_settings(self, settings):
        """Set server settings."""
        if "server_url" in settings:
            self.server_url_var.set(settings["server_url"])


class SetupGUI:
    """Main setup GUI application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GAJA Client Setup")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Initialize settings manager
        self.settings_manager = None
        if SETTINGS_AVAILABLE and SettingsManager is not None:
            self.settings_manager = SettingsManager()
        
        self.config_file = Path(__file__).parent / "client_config.json"
        self.setup_complete_file = Path(__file__).parent / "setup_complete.lock"
        
        self.create_widgets()
        self.load_current_settings()
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Create main GUI widgets."""
        # Main container with scrollable content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🎤 GAJA Client Setup & Configuration", 
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Create notebook for different settings categories
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Audio Settings Tab
        audio_frame = ttk.Frame(self.notebook)
        self.notebook.add(audio_frame, text="🎵 Audio")
        
        self.audio_settings = AudioDeviceFrame(audio_frame, self.settings_manager)
        self.audio_settings.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Voice Settings Tab
        voice_frame = ttk.Frame(self.notebook)
        self.notebook.add(voice_frame, text="🗣️ Voice")
        
        self.voice_settings = VoiceSettingsFrame(voice_frame)
        self.voice_settings.pack(fill="x", padx=10, pady=10)
        
        # Server Settings Tab
        server_frame = ttk.Frame(self.notebook)
        self.notebook.add(server_frame, text="🌐 Server")
        
        self.server_settings = ServerSettingsFrame(server_frame)
        self.server_settings.pack(fill="x", padx=10, pady=10)
        
        # Overlay & UI Settings Tab
        ui_frame = ttk.Frame(self.notebook)
        self.notebook.add(ui_frame, text="🖥️ Interface")
        
        self.create_ui_settings(ui_frame)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
        # Left side buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side="left")
        
        ttk.Button(left_buttons, text="Load Settings", command=self.load_settings_file).pack(side="left", padx=(0, 10))
        ttk.Button(left_buttons, text="Export Settings", command=self.export_settings).pack(side="left", padx=(0, 10))
        ttk.Button(left_buttons, text="Reset to Defaults", command=self.reset_to_defaults).pack(side="left")
        
        # Right side buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side="right")
        
        ttk.Button(right_buttons, text="Cancel", command=self.cancel).pack(side="left", padx=(0, 10))
        ttk.Button(right_buttons, text="Save & Apply", command=self.save_settings, style="Accent.TButton").pack(side="left")
    
    def create_ui_settings(self, parent):
        """Create UI settings widgets."""
        # Overlay settings
        overlay_frame = ttk.LabelFrame(parent, text="Overlay Settings", padding=10)
        overlay_frame.pack(fill="x", padx=10, pady=10)
        
        # Enable overlay
        self.overlay_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            overlay_frame, 
            text="Enable Overlay", 
            variable=self.overlay_enabled_var
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Overlay position
        ttk.Label(overlay_frame, text="Position:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.overlay_position_var = tk.StringVar(value="top-right")
        position_combo = ttk.Combobox(
            overlay_frame,
            textvariable=self.overlay_position_var,
            values=["top-right", "top-left", "bottom-right", "bottom-left", "center"],
            state="readonly",
            width=15
        )
        position_combo.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Auto-hide delay
        ttk.Label(overlay_frame, text="Auto-hide delay (seconds):").grid(row=2, column=0, sticky="w", pady=(0, 5))
        self.auto_hide_var = tk.IntVar(value=10)
        auto_hide_spin = ttk.Spinbox(
            overlay_frame,
            from_=1,
            to=60,
            textvariable=self.auto_hide_var,
            width=10
        )
        auto_hide_spin.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # System settings
        system_frame = ttk.LabelFrame(parent, text="System Settings", padding=10)
        system_frame.pack(fill="x", padx=10, pady=10)
        
        # Enable system tray
        self.tray_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            system_frame, 
            text="Enable System Tray", 
            variable=self.tray_enabled_var
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Enable notifications
        self.notifications_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            system_frame, 
            text="Enable Notifications", 
            variable=self.notifications_var
        ).grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        # Auto-start with Windows
        self.auto_start_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            system_frame, 
            text="Start with Windows", 
            variable=self.auto_start_var
        ).grid(row=2, column=0, sticky="w", pady=(0, 5))
        
        # Daily briefing settings
        briefing_frame = ttk.LabelFrame(parent, text="Daily Briefing", padding=10)
        briefing_frame.pack(fill="x", padx=10, pady=10)
        
        # Enable daily briefing
        self.briefing_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            briefing_frame, 
            text="Enable Daily Briefing", 
            variable=self.briefing_enabled_var
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Startup briefing
        self.startup_briefing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            briefing_frame, 
            text="Briefing on Startup", 
            variable=self.startup_briefing_var
        ).grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        # Location
        ttk.Label(briefing_frame, text="Location:").grid(row=2, column=0, sticky="w", pady=(0, 5))
        self.location_var = tk.StringVar(value="Sosnowiec,PL")
        location_entry = ttk.Entry(briefing_frame, textvariable=self.location_var, width=30)
        location_entry.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(0, 5))
        
        # Configure column weights
        overlay_frame.columnconfigure(1, weight=1)
        briefing_frame.columnconfigure(1, weight=1)
    
    def load_current_settings(self):
        """Load current settings from config file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, encoding="utf-8") as f:
                    config = json.load(f)
                
                # Audio settings
                audio_config = config.get("audio", {})
                if "input_device" in audio_config or "output_device" in audio_config:
                    self.audio_settings.set_selected_devices(
                        audio_config.get("input_device"),
                        audio_config.get("output_device")
                    )
                
                # Voice settings
                voice_settings = {}
                if "wake_word" in config:
                    voice_settings["wake_word"] = config["wake_word"].get("model", "gaja")
                    voice_settings["sensitivity"] = config["wake_word"].get("threshold", 0.65)
                elif "wakeword" in config:
                    voice_settings["wake_word"] = config["wakeword"].get("keyword", "gaja")
                    voice_settings["sensitivity"] = config["wakeword"].get("sensitivity", 0.65)
                
                if "speech" in config:
                    voice_settings["language"] = config["speech"].get("language", "pl-PL")
                
                if "recognition" in config:
                    voice_settings["whisper_model"] = config["recognition"].get("model", "base")
                elif "whisper" in config:
                    voice_settings["whisper_model"] = config["whisper"].get("model", "base")
                
                self.voice_settings.set_voice_settings(voice_settings)
                
                # Server settings
                server_url = None
                if "server" in config:
                    server_url = config["server"].get("websocket_url")
                if not server_url:
                    server_url = config.get("server_url", "ws://localhost:8001/ws/client1")
                
                self.server_settings.set_server_settings({"server_url": server_url})
                
                # UI settings
                ui_config = config.get("ui", {})
                self.overlay_enabled_var.set(ui_config.get("overlay_enabled", True))
                self.tray_enabled_var.set(ui_config.get("tray_enabled", True))
                self.notifications_var.set(ui_config.get("notifications", True))
                self.auto_start_var.set(ui_config.get("auto_start", False))
                
                # Overlay settings
                overlay_config = config.get("overlay", {})
                self.overlay_position_var.set(overlay_config.get("position", "top-right"))
                self.auto_hide_var.set(overlay_config.get("auto_hide_delay", 10))
                
                # Daily briefing
                briefing_config = config.get("daily_briefing", {})
                self.briefing_enabled_var.set(briefing_config.get("enabled", True))
                self.startup_briefing_var.set(briefing_config.get("startup_briefing", True))
                self.location_var.set(briefing_config.get("location", "Sosnowiec,PL"))
                
                logger.info("Settings loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            messagebox.showerror("Error", f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save current settings to config file."""
        try:
            # Get settings from all frames
            audio_devices = self.audio_settings.get_selected_devices()
            voice_settings = self.voice_settings.get_voice_settings()
            server_settings = self.server_settings.get_server_settings()
            
            # Build config structure
            config = {
                "server": {
                    "host": "localhost",
                    "port": 8001,
                    "websocket_url": server_settings["server_url"]
                },
                "server_url": server_settings["server_url"],  # Legacy compatibility
                "user_id": "1",
                "audio": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "chunk_size": 1024,
                    "input_device": audio_devices["input_device"],
                    "output_device": audio_devices["output_device"],
                    "record_duration": 5.0
                },
                "wake_word": {
                    "enabled": True,
                    "model": voice_settings["wake_word"],
                    "threshold": voice_settings["sensitivity"],
                    "timeout": 30
                },
                "wakeword": {  # Alternative format for compatibility
                    "enabled": True,
                    "keyword": voice_settings["wake_word"],
                    "sensitivity": voice_settings["sensitivity"],
                    "device_id": audio_devices["input_device"],
                    "stt_silence_threshold_ms": 2000
                },
                "speech": {
                    "provider": "edge",
                    "language": voice_settings["language"],
                    "voice": f"{voice_settings['language']}-ZofiaNeural" if voice_settings["language"].startswith("pl") else f"{voice_settings['language']}-Standard-A"
                },
                "recognition": {
                    "provider": "faster_whisper",
                    "model": voice_settings["whisper_model"],
                    "language": voice_settings["language"].split("-")[0]
                },
                "whisper": {  # Alternative format for compatibility
                    "model": voice_settings["whisper_model"],
                    "language": voice_settings["language"].split("-")[0]
                },
                "ui": {
                    "overlay_enabled": self.overlay_enabled_var.get(),
                    "tray_enabled": self.tray_enabled_var.get(),
                    "notifications": self.notifications_var.get(),
                    "auto_start": self.auto_start_var.get()
                },
                "overlay": {
                    "enabled": self.overlay_enabled_var.get(),
                    "position": self.overlay_position_var.get(),
                    "opacity": 0.9,
                    "auto_hide_delay": self.auto_hide_var.get()
                },
                "daily_briefing": {
                    "enabled": self.briefing_enabled_var.get(),
                    "startup_briefing": self.startup_briefing_var.get(),
                    "briefing_time": "08:00",
                    "location": self.location_var.get()
                },
                "features": {
                    "voice_activation": True,
                    "continuous_listening": False,
                    "auto_response": True,
                    "keyboard_shortcuts": True
                }
            }
            
            # Create backup of existing file
            if self.config_file.exists():
                backup_path = self.config_file.with_suffix(".json.backup")
                import shutil
                shutil.copy2(self.config_file, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Save new config
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Create setup complete lock file
            with open(self.setup_complete_file, "w", encoding="utf-8") as f:
                json.dump({
                    "setup_completed": True,
                    "timestamp": str(Path(__file__).stat().st_mtime),
                    "version": "1.0.0",
                    "configured_by": "Setup GUI"
                }, f, indent=2)
            
            logger.info("Settings saved successfully")
            messagebox.showinfo("Success", "Settings saved successfully!\n\nYou can now start the GAJA Client.")
            
            # Ask if user wants to close
            if messagebox.askyesno("Close Setup", "Settings saved! Do you want to close the setup window?"):
                self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
    def load_settings_file(self):
        """Load settings from a file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Settings",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=Path(__file__).parent
            )
            
            if file_path:
                with open(file_path, encoding="utf-8") as f:
                    config = json.load(f)
                
                # Temporarily replace config file path
                old_config_file = self.config_file
                self.config_file = Path(file_path)
                
                # Load settings
                self.load_current_settings()
                
                # Restore original config file path
                self.config_file = old_config_file
                
                messagebox.showinfo("Success", f"Settings loaded from {Path(file_path).name}")
                
        except Exception as e:
            logger.error(f"Error loading settings file: {e}")
            messagebox.showerror("Error", f"Failed to load settings: {e}")
    
    def export_settings(self):
        """Export current settings to a file."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Export Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir=Path(__file__).parent
            )
            
            if file_path:
                # Get current settings from UI
                if self.config_file.exists():
                    with open(self.config_file, encoding="utf-8") as f:
                        current_config = json.load(f)
                else:
                    current_config = {}
                
                # Save to export file
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(current_config, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Success", f"Settings exported to {Path(file_path).name}")
                
        except Exception as e:
            logger.error(f"Error exporting settings: {e}")
            messagebox.showerror("Error", f"Failed to export settings: {e}")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all settings to defaults?"):
            try:
                # Reset audio devices
                self.audio_settings.refresh_devices()
                
                # Reset voice settings
                default_voice = {
                    "wake_word": "gaja",
                    "sensitivity": 0.65,
                    "language": "pl-PL",
                    "whisper_model": "base"
                }
                self.voice_settings.set_voice_settings(default_voice)
                
                # Reset server settings
                self.server_settings.set_server_settings({"server_url": "ws://localhost:8001/ws/client1"})
                
                # Reset UI settings
                self.overlay_enabled_var.set(True)
                self.overlay_position_var.set("top-right")
                self.auto_hide_var.set(10)
                self.tray_enabled_var.set(True)
                self.notifications_var.set(True)
                self.auto_start_var.set(False)
                self.briefing_enabled_var.set(True)
                self.startup_briefing_var.set(True)
                self.location_var.set("Sosnowiec,PL")
                
                messagebox.showinfo("Success", "Settings reset to defaults")
                
            except Exception as e:
                logger.error(f"Error resetting settings: {e}")
                messagebox.showerror("Error", f"Failed to reset settings: {e}")
    
    def cancel(self):
        """Cancel setup and close window."""
        if messagebox.askyesno("Confirm Cancel", "Are you sure you want to cancel setup?"):
            self.root.destroy()
    
    def run(self):
        """Run the setup GUI."""
        # Apply a modern theme if available
        try:
            style = ttk.Style()
            available_themes = style.theme_names()
            if "vista" in available_themes:
                style.theme_use("vista")
            elif "clam" in available_themes:
                style.theme_use("clam")
            
            # Configure accent button style
            style.configure("Accent.TButton", foreground="white")
            
        except Exception as e:
            logger.warning(f"Could not apply theme: {e}")
        
        # Start GUI
        self.root.mainloop()


def main():
    """Main function to run the setup GUI."""
    try:
        # Configure logging
        logger.add(
            Path(__file__).parent / "logs" / "setup_gui.log",
            level="INFO",
            retention="7 days",
            compression="zip"
        )
        
        logger.info("Starting GAJA Client Setup GUI")
        
        # Create and run GUI
        setup_gui = SetupGUI()
        setup_gui.run()
        
        logger.info("Setup GUI closed")
        
    except Exception as e:
        logger.error(f"Error running setup GUI: {e}")
        # Show error in popup if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Setup Error", f"Failed to start setup GUI:\n\n{e}")
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()
