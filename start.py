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
    
    def install_dependencies(self) -> bool:
        """Install required dependencies automatically."""
        self.logger.info("Installing dependencies...")
        
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
                "tkinter"  # Should be included with Python
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
    
    async def start_client(self, config: Dict):
        """Start the GAJA client."""
        self.logger.info("Starting GAJA Client...")
        
        # Import client modules after dependencies are installed
        try:
            from client_main import GajaClient
        except ImportError as e:
            self.logger.error(f"Failed to import client modules: {e}")
            self.logger.error("Please check if all dependencies are installed")
            return False
        
        # Test server connection
        websocket_url = config["server"]["websocket_url"]
        if not await self.test_server_connection(websocket_url):
            self.logger.warning("Cannot connect to server, client may not work properly")
        
        # Start client
        try:
            client = GajaClient(config)
            await client.run()
        except Exception as e:
            self.logger.error(f"Client startup failed: {e}")
            return False
        
        return True
    
    def print_startup_info(self, config: Dict):
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
        print("="*60)
        print("Say 'Gaja' to wake up the assistant (if enabled)")
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
        
        # Step 3: Check if first run and show setup UI
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
        
        # Override config with command line arguments
        if args.server_host:
            config["server"]["host"] = args.server_host
            config["server"]["websocket_url"] = f"ws://{args.server_host}:{config['server']['port']}/ws"
        
        if args.server_port:
            config["server"]["port"] = args.server_port
            config["server"]["websocket_url"] = f"ws://{config['server']['host']}:{args.server_port}/ws"
        
        # Step 4: Print startup info
        self.print_startup_info(config)
        
        # Step 5: Start client
        try:
            await self.start_client(config)
            return 0
        except KeyboardInterrupt:
            self.logger.info("Client stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Client failed: {e}")
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

First Run:
  1. python start.py --install-deps   # Install dependencies
  2. Follow setup wizard             # Configure basic settings
  3. Make sure GAJA Server is running
  4. Start talking to GAJA!

Advanced:
  Edit client_config.json for detailed configuration
        """
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
    
    # Run the client
    try:
        exit_code = asyncio.run(starter.run(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nClient stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
