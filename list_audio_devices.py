"""Audio device listing utility for GAJA Client."""

import sys
from pathlib import Path

# Add client path to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))


def list_audio_devices():
    """List all available audio input devices."""
    try:
        import sounddevice as sd

        print("=== Available Audio Input Devices ===")
        devices = sd.query_devices()
        default_input = sd.default.device[0] if sd.default.device else None

        input_devices = []

        for i, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:  # Input device
                is_default = i == default_input
                status = " (DEFAULT)" if is_default else ""
                print(f"ID {i:2d}: {device['name']}{status}")
                print(
                    f"       Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}"
                )

                input_devices.append(
                    {
                        "id": i,
                        "name": device["name"],
                        "is_default": is_default,
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )

        print(f"\nFound {len(input_devices)} input devices")

        if default_input is not None:
            print(f"Current default input device: ID {default_input}")
        else:
            print("No default input device set")

        return input_devices

    except ImportError:
        print("❌ sounddevice not available - cannot list audio devices")
        print("Install with: pip install sounddevice")
        return []
    except Exception as e:
        print(f"❌ Error listing audio devices: {e}")
        return []


def test_device(device_id):
    """Test recording from a specific device."""
    try:
        import numpy as np
        import sounddevice as sd

        print(f"\n🎤 Testing device ID {device_id}...")

        # Record 2 seconds of audio
        duration = 2  # seconds
        sample_rate = 16000

        print(f"Recording {duration} seconds at {sample_rate}Hz...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
        )
        sd.wait()  # Wait until recording is finished

        # Analyze the audio
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))

        print("✅ Recording successful!")
        print(f"   Max amplitude: {max_amplitude:.4f}")
        print(f"   RMS level: {rms_level:.4f}")

        if max_amplitude < 0.001:
            print("⚠️  Very low signal - check microphone connection")
        elif max_amplitude > 0.1:
            print("✅ Good signal level detected")
        else:
            print("🔸 Low signal detected - might need adjustment")

        return True

    except Exception as e:
        print(f"❌ Error testing device {device_id}: {e}")
        return False


def main():
    """Main function."""
    print("🎵 GAJA Audio Device Manager")
    print("=" * 40)

    devices = list_audio_devices()

    if not devices:
        return

    print("\n" + "=" * 40)
    print("Commands:")
    print("  python list_audio_devices.py test <device_id>  - Test specific device")
    print("  python list_audio_devices.py                   - List all devices")

    # Check for test command
    if len(sys.argv) >= 3 and sys.argv[1] == "test":
        try:
            device_id = int(sys.argv[2])
            test_device(device_id)
        except ValueError:
            print("❌ Invalid device ID. Must be a number.")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
