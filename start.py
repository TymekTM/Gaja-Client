#!/usr/bin/env python3
"""
Starter script for the new modular GAJA client.
Use this to test the new modular architecture.
"""

import asyncio
import signal
import sys
from pathlib import Path

# Add the modules directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.app import GajaClient


async def main():
    """Run the new modular GAJA client."""
    print("🚀 Starting GAJA Client (Modular Version)")
    print("=" * 50)
    
    client = GajaClient()
    
    # Handle shutdown signals gracefully
    shutdown_called = False
    def signal_handler():
        nonlocal shutdown_called
        if shutdown_called:
            return
        shutdown_called = True
        print("\n🛑 Shutdown signal received...")
        client.running = False
    
    # Set up signal handlers (Windows compatible)
    if sys.platform == "win32":
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler())
        signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler())
    else:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 GAJA Client stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
