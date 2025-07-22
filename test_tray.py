"""Test script for GAJA System Tray Manager Run this to test the tray functionality
independently."""

import sys
import time
from pathlib import Path

# Add client modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

try:
    from modules.tray_manager import TrayManager

    print("Testing GAJA System Tray Manager...")
    print("This will show a system tray icon. Right-click to see the menu.")
    print("Press Ctrl+C to stop.")

    # Create and start tray manager
    tray = TrayManager()

    if tray.start():
        print("✅ System tray started successfully!")
        print("Look for the GAJA icon in your system tray (bottom-right corner)")

        try:
            # Simulate status changes
            statuses = [
                ("Starting...", 2),
                ("Connected", 3),
                ("Processing", 2),
                ("Ready", 3),
                ("Recording", 2),
                ("Listening...", 5),
                ("Error", 2),
                ("Disconnected", 3),
                ("Ready", 0),  # Final state
            ]

            for status, wait_time in statuses:
                print(f"📊 Status: {status}")
                tray.update_status(status)
                if wait_time > 0:
                    time.sleep(wait_time)

            print("✅ Status simulation complete. Tray will remain active.")
            print("Right-click the tray icon to test menu options.")
            print("Press Ctrl+C to exit.")

            # Keep running until interrupted
            while tray.is_running():
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Stopping tray...")
            tray.stop()
            print("✅ Tray stopped successfully")

    else:
        print("❌ Failed to start system tray")
        print("Possible issues:")
        print("- pystray not installed: pip install pystray")
        print("- Pillow not installed: pip install pillow")
        print("- System tray not available on this system")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure pystray and pillow are installed:")
    print("pip install pystray pillow")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
