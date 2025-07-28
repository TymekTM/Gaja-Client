#!/usr/bin/env python3
"""
GAJA Client - Voice Assistant with First-Run Setup
Enhanced with setup completion check and modern GUI.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Dict, Optional, Any

# Minimal imports for startup
try:
    import websockets
except ImportError:
    websockets = None


def load_env_file(env_path: Optional[Path] = None):
    """Load environment variables from .env file."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        return
    
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse variable
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
                        
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")


class GajaClientStarter:
    """Main client starter with enhanced setup management."""
    
    def __init__(self):
        # Load environment variables first
        load_env_file()
        
        self.client_root = Path(__file__).parent
        self.config_file = self.client_root / "client_config.json"
        self.setup_complete_file = self.client_root / "setup_complete.lock"
        self.log_dir = self.client_root / "logs"
        
        # Setup basic logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup basic logging for startup process."""
        self.log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "client_startup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def check_setup_status(self) -> Dict[str, Any]:
        """Check current setup status using setup_manager."""
        try:
            # Try to import setup manager
            setup_manager_path = self.client_root / "setup_manager.py"
            if setup_manager_path.exists():
                sys.path.insert(0, str(self.client_root))
                setup_manager = __import__('setup_manager')
                
                if hasattr(setup_manager, 'SetupManager'):
                    manager = setup_manager.SetupManager()
                    status = manager.get_setup_status()
                    self.logger.debug(f"Setup status: {status}")
                    return status
                
        except Exception as e:
            self.logger.warning(f"Could not use setup_manager: {e}")
        
        # Fallback to simple check
        return {
            "setup_complete": self.setup_complete_file.exists() and self.config_file.exists(),
            "config_exists": self.config_file.exists(),
            "needs_setup": not (self.setup_complete_file.exists() and self.config_file.exists()),
            "recommendations": ["Run setup GUI if configuration is incomplete"]
        }
    
    def show_setup_choice_dialog(self) -> str:
        """Show dialog to choose setup method."""
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        setup_choice = messagebox.askyesnocancel(
            "GAJA Setup Required",
            "GAJA Client needs to be configured before first use.\n\n"
            "Choose your setup method:\n\n"
            "✅ YES - Open modern setup GUI (Recommended)\n"
            "❌ NO - Use quick command-line setup\n"
            "🚫 CANCEL - Exit and setup manually later",
            icon="question"
        )
        
        root.destroy()
        
        if setup_choice is True:
            return "gui"
        elif setup_choice is False:
            return "cli"
        else:
            return "cancel"
    
    def launch_setup_gui(self) -> bool:
        """Launch the modern setup GUI."""
        try:
            setup_gui_path = self.client_root / "setup_gui.py"
            if not setup_gui_path.exists():
                self.logger.error("setup_gui.py not found")
                return False
            
            self.logger.info("Launching setup GUI...")
            
            # Launch setup GUI as subprocess
            result = subprocess.run([
                sys.executable, str(setup_gui_path)
            ], cwd=str(self.client_root))
            
            if result.returncode == 0:
                self.logger.info("Setup GUI completed successfully")
                return True
            else:
                self.logger.warning(f"Setup GUI exited with code: {result.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error launching setup GUI: {e}")
            return False
    
    def quick_cli_setup(self) -> bool:
        """Quick command-line setup for basic configuration."""
        try:
            print("\n" + "="*50)
            print("🎤 GAJA Client - Quick Setup")
            print("="*50)
            
            # Server settings
            print("\n📡 Server Configuration:")
            server_host = input("Server host [localhost]: ").strip() or "localhost"
            server_port = input("Server port [8001]: ").strip() or "8001"
            
            try:
                server_port = int(server_port)
            except ValueError:
                print("Invalid port, using default 8001")
                server_port = 8001
            
            # Audio device
            print("\n🎵 Audio Configuration:")
            print("Available microphones:")
            devices = self.get_available_audio_devices()
            for i, device in enumerate(devices):
                print(f"  {i}: {device['name']}")
            
            mic_choice = input(f"Select microphone [0]: ").strip() or "0"
            try:
                mic_id = devices[int(mic_choice)]["id"]
            except (ValueError, IndexError):
                print("Invalid choice, using default microphone")
                mic_id = "default"
            
            # Language
            print("\n🗣️ Language Configuration:")
            languages = ["pl-PL", "en-US", "de-DE", "fr-FR"]
            for i, lang in enumerate(languages):
                print(f"  {i}: {lang}")
            
            lang_choice = input("Select language [0]: ").strip() or "0"
            try:
                language = languages[int(lang_choice)]
            except (ValueError, IndexError):
                print("Invalid choice, using Polish")
                language = "pl-PL"
            
            # Create configuration
            config = self.create_default_config()
            config["server"]["host"] = server_host
            config["server"]["port"] = server_port
            config["server"]["websocket_url"] = f"ws://{server_host}:{server_port}/ws/client1"
            config["audio"]["input_device"] = mic_id
            config["speech"]["language"] = language
            config["recognition"]["language"] = language.split("-")[0]
            
            # Save configuration
            self.save_config(config)
            self.mark_setup_complete("cli")
            
            print("\n✅ Setup completed successfully!")
            print(f"Configuration saved to: {self.config_file}")
            print("\nYou can now start GAJA Client normally.")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user")
            return False
        except Exception as e:
            self.logger.error(f"Error in CLI setup: {e}")
            return False
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        if sys.version_info < (3, 11):
            self.logger.error(f"Python 3.11+ required, found {sys.version}")
            return False
        return True
    
    def check_system_requirements(self) -> bool:
        """Check system requirements for voice client."""
        issues = []
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 2 * 1024 * 1024 * 1024:  # 2GB
                issues.append("Minimum 2GB RAM required")
        except ImportError:
            self.logger.warning("Cannot check memory requirements")
        
        # Check audio devices (optional check)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            output_devices = [d for d in devices if d.get('max_output_channels', 0) > 0]
            
            if not input_devices:
                issues.append("No microphone found")
            if not output_devices:
                issues.append("No speakers/headphones found")
                
        except ImportError:
            self.logger.warning("Cannot check audio devices (sounddevice not installed)")
        except Exception as e:
            self.logger.warning(f"Audio device check failed: {e}")
        
        if issues:
            self.logger.warning("System requirement issues found:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
            return False
        
        return True
    
    def get_available_audio_devices(self):
        """Get list of available audio input devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = [
                {
                    "id": i,
                    "name": d.get('name', f'Device {i}'),
                    "channels": d.get('max_input_channels', 0)
                }
                for i, d in enumerate(devices) 
                if d.get('max_input_channels', 0) > 0
            ]
            return input_devices
        except ImportError:
            self.logger.warning("sounddevice not available for device listing")
            return [{"id": "default", "name": "Default Microphone", "channels": 1}]
        except Exception as e:
            self.logger.warning(f"Error getting audio devices: {e}")
            return [{"id": "default", "name": "Default Microphone", "channels": 1}]
    
    def check_server_reachability(self, host: str, port: int) -> bool:
        """Check if server is reachable."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies automatically."""
        self.logger.info("Checking and installing dependencies...")
        
        requirements_file = self.client_root / "requirements_client.txt"
        if not requirements_file.exists():
            # Create minimal requirements if file doesn't exist
            minimal_deps = [
                "websockets>=12.0",
                "aiohttp>=3.11.0",
                "sounddevice>=0.4.6",
                "faster-whisper>=1.1.0",
                "edge-tts>=7.0.0",
                "openwakeword>=0.6.0",
                "numpy>=2.1.0",
                "loguru>=0.7.2",
                "requests>=2.31.0", 
                "pydantic>=2.5.0",
                "psutil>=5.9.6",
                "aiofiles>=0.8.0",
                "python-dotenv>=1.0.0",
                "pystray>=0.19.5",
                "pillow>=10.0.0"
            ]
            
            with open(requirements_file, 'w') as f:
                f.write('\n'.join(minimal_deps))
            
            self.logger.info("Created minimal requirements file")
        
        try:
            # Install requirements
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            self.logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def is_first_run(self) -> bool:
        """Check if this is the first run."""
        return not self.setup_complete_file.exists()
    
    def create_default_config(self) -> Dict:
        """Create default configuration."""
        default_config = {
            "server": {
                "host": "localhost",
                "port": 8001,
                "websocket_url": "ws://localhost:8001/ws"
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "input_device": "default",
                "output_device": "default"
            },
            "wake_word": {
                "enabled": True,
                "model": "gaja",
                "threshold": 0.5,
                "timeout": 30
            },
            "speech": {
                "provider": "edge",  # edge, openai
                "language": "pl-PL",
                "voice": "pl-PL-ZofiaNeural"
            },
            "recognition": {
                "provider": "faster_whisper",  # faster_whisper, openai
                "model": "base",
                "language": "pl"
            },
            "ui": {
                "overlay_enabled": False,
                "tray_enabled": True,
                "notifications": True,
                "auto_start": False
            },
            "features": {
                "voice_activation": True,
                "continuous_listening": False,
                "auto_response": True,
                "keyboard_shortcuts": True
            }
        }
        
        return default_config
    
    def show_setup_ui(self) -> Optional[Dict]:
        """Show first-run setup UI using tkinter."""
        self.logger.info("Showing first-run setup UI")
        
        root = tk.Tk()
        root.title("GAJA Assistant - First Run Setup")
        root.geometry("600x600")
        root.resizable(False, False)
        
        # Center window
        root.eval('tk::PlaceWindow . center')
        
        setup_result = {}
        
        def save_and_close():
            """Save configuration and close setup."""
            try:
                # Server settings
                setup_result["server_host"] = server_host_var.get()
                setup_result["server_port"] = int(server_port_var.get())
                
                # Audio settings
                setup_result["speech_provider"] = speech_provider_var.get()
                setup_result["recognition_provider"] = recognition_provider_var.get()
                setup_result["language"] = language_var.get()
                
                # Audio settings
                selected_mic = microphone_var.get()
                if selected_mic and "ID:" in selected_mic:
                    # Extract device ID from selection (format: "Device Name (ID: 0)")
                    mic_id = selected_mic.split("ID: ")[1].rstrip(")")
                    try:
                        setup_result["microphone_id"] = int(mic_id) if mic_id.isdigit() else mic_id
                    except ValueError:
                        setup_result["microphone_id"] = "default"
                else:
                    setup_result["microphone_id"] = "default"
                
                # Feature settings
                setup_result["overlay_enabled"] = overlay_var.get()
                setup_result["auto_start"] = autostart_var.get()
                setup_result["wake_word_enabled"] = wakeword_var.get()
                
                root.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 GAJA Assistant Setup", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Server settings
        ttk.Label(main_frame, text="Server Settings", 
                 font=("Arial", 12, "bold")).grid(row=1, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(0, 10))
        
        ttk.Label(main_frame, text="Server Host:").grid(row=2, column=0, sticky=tk.W)
        server_host_var = tk.StringVar(value=os.getenv("GAJA_SERVER_HOST", "localhost"))
        ttk.Entry(main_frame, textvariable=server_host_var, width=30).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="Server Port:").grid(row=3, column=0, sticky=tk.W)
        server_port_var = tk.StringVar(value=os.getenv("GAJA_SERVER_PORT", "8001"))
        ttk.Entry(main_frame, textvariable=server_port_var, width=30).grid(row=3, column=1, sticky=tk.W)
        
        # Speech settings
        ttk.Label(main_frame, text="Speech Settings", 
                 font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(20, 10))
        
        ttk.Label(main_frame, text="Text-to-Speech:").grid(row=5, column=0, sticky=tk.W)
        speech_provider_var = tk.StringVar(value=os.getenv("TTS_ENGINE", "edge"))
        speech_combo = ttk.Combobox(main_frame, textvariable=speech_provider_var, 
                                   values=["edge", "openai"], state="readonly", width=27)
        speech_combo.grid(row=5, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="Speech Recognition:").grid(row=6, column=0, sticky=tk.W)
        recognition_provider_var = tk.StringVar(value=os.getenv("STT_ENGINE", "faster_whisper"))
        recognition_combo = ttk.Combobox(main_frame, textvariable=recognition_provider_var, 
                                        values=["faster_whisper", "openai"], 
                                        state="readonly", width=27)
        recognition_combo.grid(row=6, column=1, sticky=tk.W)
        
        ttk.Label(main_frame, text="Language:").grid(row=7, column=0, sticky=tk.W)
        language_var = tk.StringVar(value=os.getenv("UI_LANGUAGE", "pl-PL"))
        language_combo = ttk.Combobox(main_frame, textvariable=language_var, 
                                     values=["pl-PL", "en-US", "de-DE", "fr-FR"], 
                                     state="readonly", width=27)
        language_combo.grid(row=7, column=1, sticky=tk.W)
        
        # Audio settings
        ttk.Label(main_frame, text="Audio Settings", 
                 font=("Arial", 12, "bold")).grid(row=8, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(20, 10))
        
        # Microphone selection
        ttk.Label(main_frame, text="Microphone:").grid(row=9, column=0, sticky=tk.W)
        audio_devices = self.get_available_audio_devices()
        microphone_var = tk.StringVar()
        mic_values = [f"{d['name']} (ID: {d['id']})" for d in audio_devices]
        mic_combo = ttk.Combobox(main_frame, textvariable=microphone_var, 
                                values=mic_values, state="readonly", width=27)
        if mic_values:
            mic_combo.set(mic_values[0])  # Select first device by default
        mic_combo.grid(row=9, column=1, sticky=tk.W)
        
        # Feature settings
        ttk.Label(main_frame, text="Features", 
                 font=("Arial", 12, "bold")).grid(row=10, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(20, 10))
        
        overlay_var = tk.BooleanVar(value=os.getenv("OVERLAY_ENABLED", "false").lower() == "true")
        ttk.Checkbutton(main_frame, text="Enable Visual Overlay (Optional)", 
                       variable=overlay_var).grid(row=11, column=0, columnspan=2, sticky=tk.W)
        
        autostart_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Start with Windows", 
                       variable=autostart_var).grid(row=12, column=0, columnspan=2, sticky=tk.W)
        
        wakeword_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Enable Wake Word Detection (\"Gaja\")", 
                       variable=wakeword_var).grid(row=13, column=0, columnspan=2, sticky=tk.W)
        
        # Info text
        info_text = tk.Text(main_frame, height=4, width=70, wrap=tk.WORD)
        info_text.grid(row=14, column=0, columnspan=2, pady=(20, 0))
        info_text.insert(tk.END, 
            "ℹ️ GAJA Assistant will connect to the server to provide AI-powered voice assistance. "
            "Make sure the GAJA Server is running before starting the client. "
            "You can change these settings later in client_config.json.")
        info_text.config(state=tk.DISABLED)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=15, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(button_frame, text="Cancel", 
                  command=root.destroy).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save & Start GAJA", 
                  command=save_and_close).pack(side=tk.LEFT)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Show window
        root.mainloop()
        
        if not setup_result:
            return None
        
        return setup_result
    
    def apply_setup_to_config(self, setup_data: Dict, config: Dict) -> Dict:
        """Apply setup data to configuration."""
        config["server"]["host"] = setup_data["server_host"]
        config["server"]["port"] = setup_data["server_port"]
        config["server"]["websocket_url"] = f"ws://{setup_data['server_host']}:{setup_data['server_port']}/ws"
        
        config["speech"]["provider"] = setup_data["speech_provider"]
        config["recognition"]["provider"] = setup_data["recognition_provider"]
        config["speech"]["language"] = setup_data["language"]
        config["recognition"]["language"] = setup_data["language"].split("-")[0]
        
        # Audio device settings
        if "microphone_id" in setup_data:
            config["audio"]["input_device"] = setup_data["microphone_id"]
        
        config["ui"]["overlay_enabled"] = setup_data["overlay_enabled"]
        config["ui"]["auto_start"] = setup_data["auto_start"]
        config["wake_word"]["enabled"] = setup_data["wake_word_enabled"]
        
        return config
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {self.config_file}")
    
    def mark_setup_complete(self, method: str = "unknown"):
        """Mark setup as complete."""
        try:
            # Try to use setup_manager for creating lock file
            setup_manager_path = self.client_root / "setup_manager.py"
            if setup_manager_path.exists():
                sys.path.insert(0, str(self.client_root))
                setup_manager = __import__('setup_manager')
                
                if hasattr(setup_manager, 'SetupManager'):
                    manager = setup_manager.SetupManager()
                    manager.create_lock_file(method)
                    self.logger.info("Setup marked as complete using setup_manager")
                    return
        except Exception as e:
            self.logger.warning(f"Could not use setup_manager: {e}")
        
        # Fallback to simple lock file
        with open(self.setup_complete_file, 'w') as f:
            f.write("setup_complete")
        self.logger.info("Setup marked as complete")
    
    def load_config(self) -> Dict:
        """Load existing configuration."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.logger.info("Configuration loaded")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}, using defaults")
            return self.create_default_config()
    
    async def test_server_connection(self, websocket_url: str) -> bool:
        """Test connection to server."""
        if not websockets:
            self.logger.warning("WebSockets not available, skipping connection test")
            return True
        
        try:
            async with websockets.connect(websocket_url, timeout=5) as ws:
                await ws.send('{"type": "ping"}')
                response = await ws.recv()
                self.logger.info("Server connection test successful")
                return True
        except Exception as e:
            self.logger.warning(f"Server connection test failed: {e}")
            return False
    
    async def start_client(self, config: Dict, development: bool = False):
        """Start the GAJA client."""
        self.logger.info("Starting GAJA Client...")
        
        # Import client modules after dependencies are installed
        try:
            # Try to import main client module
            if development:
                self.logger.info("Starting in development mode")
            
            # Check if main client exists
            client_main_path = self.client_root / "client_main.py"
            if not client_main_path.exists():
                self.logger.error(f"client_main.py not found at {client_main_path}")
                self.logger.error("Make sure you have the complete GAJA Client installation")
                return False
            
            # Try dynamic import
            sys.path.insert(0, str(self.client_root))
            client_main = __import__('client_main')
            
            if hasattr(client_main, 'GajaClient'):
                GajaClient = client_main.GajaClient
            elif hasattr(client_main, 'main'):
                # Fallback: if there's a main function, run it
                self.logger.info("Running client via main function")
                await client_main.main()
                return True
            else:
                self.logger.error("No GajaClient class or main function found in client_main.py")
                return False
                
        except ImportError as e:
            self.logger.error(f"Failed to import client modules: {e}")
            self.logger.error("Please check if all dependencies are installed")
            self.logger.error("Try: python start.py --install-deps")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error importing client: {e}")
            return False
        
        # Test server connection
        websocket_url = config["server"]["websocket_url"]
        if not await self.test_server_connection(websocket_url):
            if development:
                self.logger.warning("Server connection failed, but continuing in development mode")
            else:
                self.logger.error("Cannot connect to server. Please ensure GAJA Server is running.")
                self.logger.error(f"Expected server at: {websocket_url}")
                return False
        
        # Start client
        try:
            client = GajaClient(config)
            await client.run()
        except Exception as e:
            self.logger.error(f"Client startup failed: {e}")
            if development:
                self.logger.error("Full traceback:", exc_info=True)
            return False
        
        return True
    
    def print_startup_info(self, config: Dict, development: bool = False):
        """Print startup information."""
        print("\n" + "="*60)
        print("🎙️ GAJA Client - Beta Release")
        print("="*60)
        print(f"Server: {config['server']['websocket_url']}")
        print(f"Speech Provider: {config['speech']['provider']}")
        print(f"Recognition Provider: {config['recognition']['provider']}")
        print(f"Language: {config['speech']['language']}")
        print(f"Wake Word: {'Enabled' if config['wake_word']['enabled'] else 'Disabled'}")
        print(f"Overlay: {'Enabled' if config['ui']['overlay_enabled'] else 'Disabled'}")
        print(f"Mode: {'Development' if development else 'Production'}")
        print("="*60)
        print("Configuration:")
        print(f"  - Config file: {self.config_file}")
        print(f"  - Logs directory: {self.log_dir}")
        print(f"  - Audio input device: {config.get('audio', {}).get('input_device', 'default')}")
        print(f"  - Audio output device: {config.get('audio', {}).get('output_device', 'default')}")
        print("="*60)
        if config['wake_word']['enabled']:
            print("Say 'Gaja' to wake up the assistant")
        print("Press Ctrl+C to stop the client")
        print()
    
    async def run(self, args):
        """Main run method with enhanced setup management."""
        self.logger.info("GAJA Client Starter - Enhanced Release")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return 1
        
        # Step 2: Install dependencies if needed
        if args.install_deps and not self.install_dependencies():
            return 1
        
        # Step 3: Check system requirements (warnings only)
        if not args.skip_checks:
            self.check_system_requirements()
        
        # Step 4: Enhanced setup status checking
        setup_status = self.check_setup_status()
        
        if setup_status["needs_setup"] and not args.skip_setup:
            self.logger.info("Setup required - configuration incomplete or missing")
            
            # Show setup choice dialog
            setup_choice = self.show_setup_choice_dialog()
            
            if setup_choice == "cancel":
                self.logger.info("Setup cancelled by user")
                return 0
            elif setup_choice == "gui":
                # Launch setup GUI
                if self.launch_setup_gui():
                    self.logger.info("Setup GUI completed, reloading configuration")
                    # Reload setup status after GUI completion
                    setup_status = self.check_setup_status()
                    if setup_status["needs_setup"]:
                        self.logger.error("Setup still incomplete after GUI")
                        return 1
                else:
                    self.logger.error("Setup GUI failed or cancelled")
                    return 1
            elif setup_choice == "cli":
                # Run CLI setup
                if not self.quick_cli_setup():
                    self.logger.error("CLI setup failed or cancelled")
                    return 1
        
        # Step 5: Load configuration
        config = self.load_config()
        
        # Step 6: Override config with command line arguments
        if args.server_host:
            config["server"]["host"] = args.server_host
            config["server"]["websocket_url"] = f"ws://{args.server_host}:{config['server']['port']}/ws/client1"
        
        if args.server_port:
            config["server"]["port"] = args.server_port
            config["server"]["websocket_url"] = f"ws://{config['server']['host']}:{args.server_port}/ws/client1"
        
        # Step 7: Check server reachability (if not in dev mode)
        if not args.dev and not args.force:
            host = config["server"]["host"]
            port = config["server"]["port"]
            if not self.check_server_reachability(host, port):
                self.logger.warning(f"Cannot reach server at {host}:{port}")
                self.logger.warning("Make sure GAJA Server is running")
                if not args.force:
                    self.logger.error("Use --force to start anyway")
                    return 1
        
        # Step 8: Print startup info
        self.print_startup_info(config, args.dev)
        
        # Step 9: Start client
        try:
            success = await self.start_client(config, development=args.dev)
            return 0 if success else 1
        except KeyboardInterrupt:
            self.logger.info("Client stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Client failed: {e}")
            if args.dev:
                self.logger.error("Full traceback:", exc_info=True)
            return 1


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GAJA Client - Voice Assistant with Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                     # First run with setup UI
  python start.py --skip-setup        # Skip setup UI (use existing config)
  python start.py --install-deps      # Force dependency installation
  python start.py --server-host 192.168.1.100  # Connect to remote server
  python start.py --dev               # Development mode with verbose logging
  python start.py --force             # Start even if server not reachable

First Run:
  1. python start.py --install-deps   # Install dependencies
  2. Follow setup wizard              # Configure basic settings
  3. Make sure GAJA Server is running
  4. Start talking to GAJA!

Development:
  python start.py --dev --skip-checks # Development mode, skip system checks
  python start.py --force             # Force start without server check

Production:
  python start.py                     # Normal start with all checks
  python start.py --server-host remote.server.com --server-port 8001

Advanced:
  Edit client_config.json for detailed configuration
        """
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Start in development mode (verbose logging, detailed errors)"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Force installation of dependencies"
    )
    
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip first-run setup UI"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip system requirement checks"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force start even if server not reachable"
    )
    
    parser.add_argument(
        "--server-host",
        type=str,
        help="Override server host (default: localhost)"
    )
    
    parser.add_argument(
        "--server-port",
        type=int,
        help="Override server port (default: 8001)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (default: client_config.json)"
    )
    
    args = parser.parse_args()
    
    # Create starter instance
    starter = GajaClientStarter()
    
    # Override config file if specified
    if args.config:
        starter.config_file = Path(args.config)
    
    # Set logging level for development mode
    if args.dev:
        logging.getLogger().setLevel(logging.DEBUG)
        starter.logger.setLevel(logging.DEBUG)
    else:
        # Set reduced logging for production mode
        import os
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING) 
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("PIL.Image").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.INFO)
        logging.getLogger("websockets.client").setLevel(logging.INFO)
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("audio_modules").setLevel(logging.INFO)
    
    # Run the client
    try:
        exit_code = asyncio.run(starter.run(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nClient stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.dev:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
