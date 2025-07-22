#!/usr/bin/env python3
"""
GAJA Client - Voice Assistant with First-Run Setup
Beta release with automatic setup UI and dependency management.
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
from typing import Dict, Optional

# Minimal imports for startup
try:
    import websockets
except ImportError:
    websockets = None


def load_env_file(env_path: Path = None):
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
    """Main client starter with first-run setup UI."""
    
    def __init__(self):
        # Load environment variables first
        load_env_file()
        
        self.client_root = Path(__file__).parent
        self.config_file = self.client_root / "client_config.json"
        self.setup_complete_file = self.client_root / ".setup_complete"
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
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            
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
        root.geometry("600x500")
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
                
                # Feature settings
                setup_result["overlay_enabled"] = overlay_var.get()
                setup_result["auto_start"] = autostart_var.get()
                setup_result["wake_word_enabled"] = wakeword_var.get()
                
                root.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
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
        
        # Feature settings
        ttk.Label(main_frame, text="Features", 
                 font=("Arial", 12, "bold")).grid(row=8, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(20, 10))
        
        overlay_var = tk.BooleanVar(value=os.getenv("OVERLAY_ENABLED", "false").lower() == "true")
        ttk.Checkbutton(main_frame, text="Enable Visual Overlay (Optional)", 
                       variable=overlay_var).grid(row=9, column=0, columnspan=2, sticky=tk.W)
        
        autostart_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="Start with Windows", 
                       variable=autostart_var).grid(row=10, column=0, columnspan=2, sticky=tk.W)
        
        wakeword_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Enable Wake Word Detection (\"Gaja\")", 
                       variable=wakeword_var).grid(row=11, column=0, columnspan=2, sticky=tk.W)
        
        # Info text
        info_text = tk.Text(main_frame, height=4, width=70, wrap=tk.WORD)
        info_text.grid(row=12, column=0, columnspan=2, pady=(20, 0))
        info_text.insert(tk.END, 
            "ℹ️ GAJA Assistant will connect to the server to provide AI-powered voice assistance. "
            "Make sure the GAJA Server is running before starting the client. "
            "You can change these settings later in client_config.json.")
        info_text.config(state=tk.DISABLED)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=13, column=0, columnspan=2, pady=(20, 0))
        
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
        
        config["ui"]["overlay_enabled"] = setup_data["overlay_enabled"]
        config["ui"]["auto_start"] = setup_data["auto_start"]
        config["wake_word"]["enabled"] = setup_data["wake_word_enabled"]
        
        return config
    
    def save_config(self, config: Dict):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {self.config_file}")
    
    def mark_setup_complete(self):
        """Mark setup as complete."""
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
        """Main run method."""
        self.logger.info("GAJA Client Starter - Beta Release")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return 1
        
        # Step 2: Install dependencies if needed
        if args.install_deps and not self.install_dependencies():
            return 1
        
        # Step 3: Check system requirements (warnings only)
        if not args.skip_checks:
            self.check_system_requirements()
        
        # Step 4: Check if first run and show setup UI
        if self.is_first_run() and not args.skip_setup:
            self.logger.info("First run detected, showing setup UI")
            
            config = self.create_default_config()
            setup_data = self.show_setup_ui()
            
            if not setup_data:
                self.logger.info("Setup cancelled by user")
                return 0
            
            config = self.apply_setup_to_config(setup_data, config)
            self.save_config(config)
            self.mark_setup_complete()
        else:
            # Load existing configuration
            config = self.load_config()
        
        # Step 5: Override config with command line arguments
        if args.server_host:
            config["server"]["host"] = args.server_host
            config["server"]["websocket_url"] = f"ws://{args.server_host}:{config['server']['port']}/ws"
        
        if args.server_port:
            config["server"]["port"] = args.server_port
            config["server"]["websocket_url"] = f"ws://{config['server']['host']}:{args.server_port}/ws"
        
        # Step 6: Check server reachability (if not in dev mode)
        if not args.dev:
            host = config["server"]["host"]
            port = config["server"]["port"]
            if not self.check_server_reachability(host, port):
                self.logger.warning(f"Cannot reach server at {host}:{port}")
                self.logger.warning("Make sure GAJA Server is running")
                if not args.force:
                    self.logger.error("Use --force to start anyway")
                    return 1
        
        # Step 7: Print startup info
        self.print_startup_info(config, args.dev)
        
        # Step 8: Start client
        try:
            await self.start_client(config, development=args.dev)
            return 0
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
