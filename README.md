# 🎙️ GAJA Client

**Voice Assistant Client for GAJA Assistant - Beta Release**

GAJA Client is a voice-activated assistant that connects to GAJA Server for AI processing. Features wake word detection, speech recognition, text-to-speech, and an optional visual overlay.

## 🚀 Quick Start

### Option 1: Plug & Play (Recommended)

```bash
# Clone or download this client folder
git clone <repo-url> gaja-client
cd gaja-client

# First run - installs dependencies and shows setup wizard
python start.py --install-deps

# Follow the setup wizard to configure basic settings
# Start using GAJA by saying "Gaja, hello!"
```

### Option 2: Manual Setup

```bash
# Install Python 3.11+
# Install dependencies
pip install -r requirements_client.txt

# Copy and edit config
cp client_config.template.json client_config.json
nano client_config.json

# Start client
python client_main.py
```

## 📋 Requirements

- **Python 3.11+** (Required)
- **GAJA Server running** (on localhost:8001 or remote)
- **Microphone and speakers** (for voice interaction)
- **2GB RAM** minimum
- **Internet connection** (for AI features)

## 🎛️ First Run Setup

On first run, GAJA Client will show a setup wizard with the following options:

### Setup Options

1. **Server Connection**
   - Host: Where GAJA Server is running (localhost for same machine)
   - Port: Server port (default: 8001)

2. **Speech Settings**
   - Text-to-Speech: Edge TTS (free) or OpenAI (requires API key)
   - Speech Recognition: Faster Whisper (local) or OpenAI (cloud)
   - Language: Polish, English, German, French

3. **Features**
   - Visual Overlay: Optional floating window
   - Auto-start: Start with Windows
   - Wake Word: Voice activation with "Gaja"

## ⚙️ Configuration

After setup, settings are saved in `client_config.json`:

```json
{
  "server": {
    "host": "localhost",
    "port": 8001,
    "websocket_url": "ws://localhost:8001/ws"
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
  "wake_word": {
    "enabled": true,
    "model": "gaja",
    "threshold": 0.5
  },
  "ui": {
    "overlay_enabled": false,
    "tray_enabled": true,
    "notifications": true
  }
}
```

## 🛠️ CLI Usage

The `start.py` script provides a convenient CLI interface:

```bash
# Normal start (shows setup on first run)
python start.py

# Skip setup wizard (use existing config)
python start.py --skip-setup

# Force install dependencies
python start.py --install-deps

# Development mode with verbose logging
python start.py --dev

# Connect to remote server
python start.py --server-host 192.168.1.100

# Use different port
python start.py --server-port 9001

# Force start even if server not reachable
python start.py --force

# Skip system requirement checks
python start.py --skip-checks

# Use custom config file
python start.py --config my_config.json
```

## 🎯 Features

### Voice Interaction
- **Wake Word**: Say "Gaja" to activate
- **Continuous Listening**: Always ready mode
- **Push-to-Talk**: Keyboard shortcut activation
- **Natural Language**: Talk naturally to GAJA

### Audio Providers

#### Text-to-Speech
- **Edge TTS**: Free, high-quality, multiple voices
- **OpenAI TTS**: Premium quality (requires API key)

#### Speech Recognition
- **Faster Whisper**: Local, private, fast
- **OpenAI Whisper**: Cloud-based, highly accurate

### User Interface
- **System Tray**: Minimal background presence
- **Visual Overlay**: Optional floating status window
- **Notifications**: Desktop notifications for responses
- **Settings UI**: Easy configuration management

## 🔌 Integration

### With GAJA Server

The client connects via WebSocket:
```
Server: http://localhost:8001
WebSocket: ws://localhost:8001/ws
```

### With Overlay (Optional)

If overlay is enabled, a floating window shows:
- Current status (listening, processing, speaking)
- Voice visualization
- Quick controls

## 📁 Project Structure

```
gaja-client/
├── start.py                 # 🚀 Plug & Play starter with setup UI
├── client_main.py          # 🎙️ Main client application
├── client_config.json      # ⚙️ Configuration
├── requirements_client.txt # 📦 Dependencies
├── README.md              # 📖 This file
├── 
├── audio_modules/          # 🔊 Audio processing
│   ├── whisper_asr.py      # Speech recognition
│   ├── tts_module.py       # Text-to-speech
│   ├── wakeword_detector.py # Wake word detection
│   └── ...
├── 
├── modules/                # 🧩 Feature modules
│   ├── tray_manager.py     # System tray
│   ├── settings_manager.py # Settings UI
│   └── ...
├── 
├── overlay/                # 🎨 Visual overlay (optional)
│   ├── gaja-overlay.exe    # Rust-based overlay
│   └── ...
├── 
├── resources/              # 📁 Assets
│   ├── gaja.ico           # Icon
│   ├── sounds/            # Sound effects
│   └── openWakeWord/      # Wake word models
├── 
├── logs/                   # 📝 Client logs
├── user_data/             # 💾 User data
└── tests/                 # 🧪 Unit tests
```

## Main Files

- `start.py` - Plug & Play starter with first-run setup UI
- `client_main.py` - Main client entry point and application logic
- `client_config.json` - Client configuration settings
- `requirements_client.txt` - Client-specific Python dependencies

## Supporting Modules

- `active_window_module.py` - Active window detection functionality
- `config.py` - Configuration management
- `shared_state.py` - Shared state management between components
- `list_audio_devices.py` - Audio device enumeration utility

## 🎮 Usage

### Basic Voice Commands

```
"Gaja, what's the weather like?"
"Gaja, play some music"
"Gaja, search for Python tutorials"
"Gaja, set a reminder for 3 PM"
"Gaja, what time is it?"
```

### Keyboard Shortcuts

- **Ctrl+Shift+G**: Manual activation (push-to-talk)
- **Ctrl+Shift+M**: Toggle microphone
- **Ctrl+Shift+O**: Toggle overlay
- **Ctrl+Shift+S**: Open settings

### System Tray

Right-click the tray icon for:
- Start/Stop listening
- Open settings
- View logs
- Exit application

## 🧪 Testing

```bash
# Test audio devices
python list_audio_devices.py

# Test wake word detection
python test_wakeword.py

# Test server connection
python test_connection.py

# Run unit tests
python -m pytest tests/ -v
```

## 🔧 Development

### Development Mode

Development mode provides enhanced debugging and verbose logging:

```bash
# Start with debug logging and detailed error traces
python start.py --dev

# Skip system checks in development
python start.py --dev --skip-checks

# Force start without server connection check
python start.py --dev --force

# Monitor logs
tail -f logs/client_startup.log  # Linux/Mac
Get-Content logs/client_startup.log -Wait  # Windows PowerShell
```

### Runtime Modes

| Mode | Command | Description |
|------|---------|-------------|
| **First Run** | `python start.py` | Shows setup wizard, installs deps |
| **Normal** | `python start.py --skip-setup` | Production mode with all checks |
| **Development** | `python start.py --dev` | Verbose logging, detailed errors |
| **Force Start** | `python start.py --force` | Start without server connection |
| **Remote Server** | `python start.py --server-host IP` | Connect to remote GAJA Server |

### Audio Configuration

For better performance, configure audio devices:

```json
{
  "audio": {
    "input_device": 1,    // Use specific device ID
    "output_device": 2,   // Use specific device ID
    "sample_rate": 16000,
    "chunk_size": 1024
  }
}
```

Find device IDs with:
```bash
python list_audio_devices.py
```

## 🚨 Troubleshooting

### Common Issues

1. **No microphone detected**
   ```bash
   python list_audio_devices.py
   # Set correct input_device in config
   # Or use: python start.py --skip-checks
   ```

2. **Cannot connect to server**
   ```bash
   # Check if server is running
   curl http://localhost:8001/health
   # Or use different host/port
   python start.py --server-host 192.168.1.100
   # Or force start anyway
   python start.py --force
   ```

3. **Dependencies installation failed**
   ```bash
   # Try upgrading pip first
   python -m pip install --upgrade pip
   # Then install dependencies
   python start.py --install-deps
   # Or manually install
   pip install -r requirements_client.txt
   ```

4. **Wake word not working**
   - Check microphone permissions
   - Adjust threshold in config
   - Test with `python test_wakeword.py`
   - Try: `python start.py --dev` for debug info

5. **Audio quality issues**
   - Update audio drivers
   - Try different audio devices
   - Adjust sample rate and chunk size
   - Use: `python start.py --skip-checks` to bypass audio checks

6. **First-run setup issues**
   ```bash
   # Skip setup and use defaults
   python start.py --skip-setup
   # Or delete setup marker and try again
   rm .setup_complete
   python start.py
   ```

### Debug Mode

For detailed troubleshooting:

```bash
# Full debug mode
python start.py --dev

# Skip all checks and force start
python start.py --dev --skip-checks --force

# Check what went wrong
cat logs/client_startup.log  # Linux/Mac
type logs\client_startup.log  # Windows
```

## 📊 Performance

**Tested Configuration:**
- Python 3.11
- 4GB RAM
- Modern CPU with AVX2

**Benchmarks:**
- ✅ Wake word detection: <100ms
- ✅ Speech recognition: 1-3 seconds
- ✅ Server communication: <200ms
- ✅ 24/7 operation tested

## 🔒 Privacy & Security

### Data Handling

- **Local Processing**: Faster Whisper runs locally
- **Encrypted Communication**: WebSocket with TLS (in production)
- **No Audio Storage**: Audio is processed and discarded
- **Configurable Privacy**: Choose local vs cloud providers

## 📝 License

MIT License - see LICENSE file for details.

---

**Status: ✅ Beta Ready**
**Version: 1.0.0-beta**
**Last Updated: January 22, 2025**
