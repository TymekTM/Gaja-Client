"""Quick test for the improved system tray menu."""
import sys
import time
from pathlib import Path

# Add client modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

from modules.tray_manager import TrayManager

print("🧪 Testing improved GAJA System Tray Menu...")
print("📋 Available menu options:")
print("   • Status")
print("   • Settings")
print("   • Connect to Server")
print("   • Toggle Monitoring")
print("   • Restart Client")
print("   • About")
print("   • Quit")
print()
print("Right-click the system tray icon to test the new menu!")


# Create mock client app
class MockClientApp:
    def __init__(self):
        self.monitoring_wakeword = True
        self.running = True

    def show_overlay(self):
        print("📱 Mock: Overlay shown")

    def update_status(self, status):
        print(f"📊 Mock: Status updated to '{status}'")


# Create and start tray manager
mock_client = MockClientApp()
tray = TrayManager(client_app=mock_client)

if tray.start():
    print("✅ Improved system tray menu started!")
    print("🖱️ Right-click the tray icon to see new menu options")

    try:
        # Test status updates
        statuses = ["Testing", "Menu Items", "Working", "Ready"]
        for status in statuses:
            print(f"🔄 Testing status: {status}")
            tray.update_status(status)
            time.sleep(1)

        print("✅ Menu test ready - try right-clicking the icon!")
        print("Press Ctrl+C to exit...")

        while tray.is_running():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopping test...")
        tray.stop()
        print("✅ Test completed")

else:
    print("❌ Failed to start tray manager")
