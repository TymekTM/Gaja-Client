#!/usr/bin/env python3
"""
Starter script for the new modular GAJA client.
Use this to test the new modular architecture.
"""

import asyncio
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
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 GAJA Client stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
