"""Setup completion check and lock file management for GAJA Client.

This module provides utilities to check if the GAJA Client has been properly
configured and to manage the setup completion lock file.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from loguru import logger


class SetupManager:
    """Manages setup completion status and lock file."""
    
    def __init__(self, client_dir: Optional[Path] = None):
        """Initialize setup manager.
        
        Args:
            client_dir: Directory containing the client files. Defaults to current directory.
        """
        self.client_dir = client_dir or Path(__file__).parent
        self.config_file = self.client_dir / "client_config.json"
        self.lock_file = self.client_dir / "setup_complete.lock" 
        self.template_file = self.client_dir / "client_config.template.json"
    
    def is_setup_complete(self) -> bool:
        """Check if setup has been completed.
        
        Returns:
            True if setup is complete, False otherwise
        """
        try:
            # Check if lock file exists
            if not self.lock_file.exists():
                logger.debug("Setup lock file does not exist")
                return False
            
            # Check if config file exists
            if not self.config_file.exists():
                logger.warning("Config file missing despite lock file existence")
                return False
            
            # Verify lock file content
            try:
                with open(self.lock_file, encoding="utf-8") as f:
                    lock_data = json.load(f)
                
                if not lock_data.get("setup_completed", False):
                    logger.debug("Lock file indicates setup not completed")
                    return False
                
                logger.debug("Setup completion verified")
                return True
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid lock file format: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking setup completion: {e}")
            return False
    
    def get_setup_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current setup.
        
        Returns:
            Dictionary with setup information or None if not setup
        """
        try:
            if not self.is_setup_complete():
                return None
            
            with open(self.lock_file, encoding="utf-8") as f:
                lock_data = json.load(f)
            
            # Add config file timestamp
            if self.config_file.exists():
                config_stat = self.config_file.stat()
                lock_data["config_modified"] = config_stat.st_mtime
                lock_data["config_modified_readable"] = time.ctime(config_stat.st_mtime)
            
            return lock_data
            
        except Exception as e:
            logger.error(f"Error getting setup info: {e}")
            return None
    
    def create_lock_file(self, setup_method: str = "manual", version: str = "1.0.0") -> bool:
        """Create setup completion lock file.
        
        Args:
            setup_method: Method used to complete setup (e.g., "gui", "manual", "automatic")
            version: Version of the setup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            lock_data = {
                "setup_completed": True,
                "timestamp": time.time(),
                "timestamp_readable": time.ctime(),
                "version": version,
                "configured_by": setup_method,
                "config_file": str(self.config_file.name),
                "client_dir": str(self.client_dir)
            }
            
            # Add configuration verification
            if self.config_file.exists():
                try:
                    with open(self.config_file, encoding="utf-8") as f:
                        config = json.load(f)
                    
                    # Check for essential configuration
                    has_audio = "audio" in config
                    has_server = "server_url" in config or "server" in config
                    has_voice = "wake_word" in config or "wakeword" in config
                    
                    lock_data["config_validation"] = {
                        "has_audio_config": has_audio,
                        "has_server_config": has_server,
                        "has_voice_config": has_voice,
                        "essential_complete": has_audio and has_server and has_voice
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not validate config during lock creation: {e}")
                    lock_data["config_validation"] = {"error": str(e)}
            
            with open(self.lock_file, "w", encoding="utf-8") as f:
                json.dump(lock_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Setup lock file created: {self.lock_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating lock file: {e}")
            return False
    
    def remove_lock_file(self) -> bool:
        """Remove setup completion lock file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
                logger.info("Setup lock file removed")
                return True
            else:
                logger.debug("Lock file does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Error removing lock file: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate the current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            "config_exists": False,
            "config_valid": False,
            "has_audio": False,
            "has_server": False,
            "has_voice": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if config file exists
            if not self.config_file.exists():
                validation["errors"].append("Configuration file does not exist")
                return validation
            
            validation["config_exists"] = True
            
            # Try to load and validate config
            try:
                with open(self.config_file, encoding="utf-8") as f:
                    config = json.load(f)
                
                validation["config_valid"] = True
                
                # Check audio configuration
                if "audio" in config:
                    audio_config = config["audio"]
                    validation["has_audio"] = True
                    
                    if not audio_config.get("input_device"):
                        validation["warnings"].append("No input device specified")
                    
                    if not audio_config.get("sample_rate"):
                        validation["warnings"].append("No sample rate specified")
                
                # Check server configuration
                if "server_url" in config or "server" in config:
                    validation["has_server"] = True
                    
                    server_url = config.get("server_url")
                    if not server_url and "server" in config:
                        server_url = config["server"].get("websocket_url")
                    
                    if not server_url:
                        validation["warnings"].append("Server URL not specified")
                    elif not server_url.startswith(("ws://", "wss://")):
                        validation["warnings"].append("Server URL should be a WebSocket URL")
                
                # Check voice configuration
                if "wake_word" in config or "wakeword" in config:
                    validation["has_voice"] = True
                    
                    wake_word_config = config.get("wake_word") or config.get("wakeword", {})
                    if not wake_word_config.get("model") and not wake_word_config.get("keyword"):
                        validation["warnings"].append("No wake word specified")
                
                # Overall validation
                if validation["has_audio"] and validation["has_server"] and validation["has_voice"]:
                    if not validation["warnings"]:
                        logger.info("Configuration validation passed")
                    else:
                        logger.info(f"Configuration validation passed with warnings: {validation['warnings']}")
                else:
                    missing = []
                    if not validation["has_audio"]:
                        missing.append("audio")
                    if not validation["has_server"]:
                        missing.append("server")
                    if not validation["has_voice"]:
                        missing.append("voice")
                    validation["errors"].append(f"Missing essential configuration: {', '.join(missing)}")
                
            except json.JSONDecodeError as e:
                validation["errors"].append(f"Invalid JSON in configuration file: {e}")
            except Exception as e:
                validation["errors"].append(f"Error reading configuration: {e}")
                
        except Exception as e:
            validation["errors"].append(f"Error validating configuration: {e}")
            logger.error(f"Error in config validation: {e}")
        
        return validation
    
    def ensure_config_exists(self) -> bool:
        """Ensure configuration file exists, create from template if needed.
        
        Returns:
            True if config exists or was created, False otherwise
        """
        try:
            if self.config_file.exists():
                logger.debug("Configuration file already exists")
                return True
            
            # Try to create from template
            if self.template_file.exists():
                logger.info("Creating configuration from template")
                import shutil
                shutil.copy2(self.template_file, self.config_file)
                return True
            
            # Create minimal default config
            logger.info("Creating default configuration")
            default_config = {
                "server": {
                    "host": "localhost",
                    "port": 8001,
                    "websocket_url": "ws://localhost:8001/ws/client1"
                },
                "server_url": "ws://localhost:8001/ws/client1",
                "user_id": "1",
                "audio": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "chunk_size": 1024,
                    "input_device": None,
                    "output_device": "default"
                },
                "wake_word": {
                    "enabled": True,
                    "model": "gaja",
                    "threshold": 0.5,
                    "timeout": 30
                },
                "wakeword": {
                    "enabled": True,
                    "keyword": "gaja",
                    "sensitivity": 0.6,
                    "device_id": None,
                    "stt_silence_threshold_ms": 2000
                },
                "speech": {
                    "provider": "edge",
                    "language": "pl-PL",
                    "voice": "pl-PL-ZofiaNeural"
                },
                "recognition": {
                    "provider": "faster_whisper",
                    "model": "base",
                    "language": "pl"
                },
                "whisper": {
                    "model": "base",
                    "language": "pl"
                },
                "ui": {
                    "overlay_enabled": False,
                    "tray_enabled": True,
                    "notifications": True,
                    "auto_start": False
                },
                "overlay": {
                    "enabled": False,
                    "position": "top-right",
                    "opacity": 0.9,
                    "auto_hide_delay": 10
                },
                "daily_briefing": {
                    "enabled": True,
                    "startup_briefing": True,
                    "briefing_time": "08:00",
                    "location": "Sosnowiec,PL"
                },
                "features": {
                    "voice_activation": True,
                    "continuous_listening": False,
                    "auto_response": True,
                    "keyboard_shortcuts": True
                }
            }
            
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Default configuration created: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring config exists: {e}")
            return False
    
    def get_setup_status(self) -> Dict[str, Any]:
        """Get comprehensive setup status.
        
        Returns:
            Dictionary with complete setup status information
        """
        status = {
            "setup_complete": False,
            "config_exists": False,  
            "config_valid": False,
            "lock_exists": False,
            "needs_setup": True,
            "setup_info": None,
            "validation": None,
            "recommendations": []
        }
        
        try:
            # Check setup completion
            status["setup_complete"] = self.is_setup_complete()
            status["lock_exists"] = self.lock_file.exists()
            
            # Get setup info if available
            if status["setup_complete"]:
                status["setup_info"] = self.get_setup_info()
            
            # Validate configuration
            validation = self.validate_config()
            status["config_exists"] = validation["config_exists"]
            status["config_valid"] = validation["config_valid"]
            status["validation"] = validation
            
            # Determine if setup is needed
            if status["setup_complete"] and status["config_valid"] and not validation["errors"]:
                status["needs_setup"] = False
                if validation["warnings"]:
                    status["recommendations"].append("Review configuration warnings")
            else:
                status["needs_setup"] = True
                
                if not status["config_exists"]:
                    status["recommendations"].append("Run setup GUI to create configuration")
                elif not status["config_valid"]:
                    status["recommendations"].append("Fix configuration file errors")
                elif validation["errors"]:
                    status["recommendations"].append("Complete missing configuration sections")
                
                if not status["lock_exists"]:
                    status["recommendations"].append("Complete setup process")
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting setup status: {e}")
            status["error"] = str(e)
            return status


def check_setup_status() -> bool:
    """Quick check if setup is complete.
    
    Returns:
        True if setup is complete, False otherwise
    """
    setup_manager = SetupManager()
    return setup_manager.is_setup_complete()


def ensure_setup() -> bool:
    """Ensure minimal setup exists.
    
    Returns:
        True if setup is ready, False if manual setup needed
    """
    setup_manager = SetupManager()
    
    # Ensure config file exists
    if not setup_manager.ensure_config_exists():
        return False
    
    # If we have a valid config but no lock, create lock
    validation = setup_manager.validate_config()
    if validation["config_valid"] and not validation["errors"]:
        if not setup_manager.is_setup_complete():
            setup_manager.create_lock_file("automatic")
        return True
    
    return False


def main():
    """Main function for command-line usage."""
    import sys
    
    setup_manager = SetupManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            status = setup_manager.get_setup_status()
            print("=== GAJA Client Setup Status ===")
            print(f"Setup Complete: {status['setup_complete']}")
            print(f"Config Exists: {status['config_exists']}")
            print(f"Config Valid: {status['config_valid']}")
            print(f"Needs Setup: {status['needs_setup']}")
            
            if status.get("setup_info"):
                info = status["setup_info"]
                print(f"\nSetup Info:")
                print(f"  Configured by: {info.get('configured_by', 'Unknown')}")
                print(f"  Timestamp: {info.get('timestamp_readable', 'Unknown')}")
                print(f"  Version: {info.get('version', 'Unknown')}")
            
            if status.get("validation", {}).get("errors"):
                print("\nErrors:")
                for error in status["validation"]["errors"]:
                    print(f"  - {error}")
            
            if status.get("validation", {}).get("warnings"):
                print("\nWarnings:")
                for warning in status["validation"]["warnings"]:
                    print(f"  - {warning}")
            
            if status.get("recommendations"):
                print("\nRecommendations:")
                for rec in status["recommendations"]:
                    print(f"  - {rec}")
        
        elif command == "create-lock":
            method = sys.argv[2] if len(sys.argv) > 2 else "manual"
            if setup_manager.create_lock_file(method):
                print("✅ Setup lock file created successfully")
            else:
                print("❌ Failed to create setup lock file")
        
        elif command == "remove-lock":
            if setup_manager.remove_lock_file():
                print("✅ Setup lock file removed successfully")
            else:
                print("❌ Failed to remove setup lock file")
        
        elif command == "ensure":
            if ensure_setup():
                print("✅ Setup is ready")
            else:
                print("❌ Manual setup required")
        
        else:
            print("Unknown command. Available commands: status, create-lock, remove-lock, ensure")
    
    else:
        # Default: show status
        if check_setup_status():
            print("✅ GAJA Client setup is complete")
        else:
            print("❌ GAJA Client setup is not complete")
            print("Run 'python setup_gui.py' to complete setup")


if __name__ == "__main__":
    main()
