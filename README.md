# Gaja Client

**Gaja Client** is a voice-based application within the [GAJA Assistant](https://github.com/TymekTM/Gaja) ecosystem, a self-hosted, modular AI home assistant emphasizing privacy, control, and extensibility.

Gaja Client offers a user-friendly experience with straightforward, code-free setup.

> **Note**: The user setup process is currently being enhanced for clarity. Only a basic GUI is available for initial configuration; other operations use the console.

## Table of Contents

* [What is Gaja?](#what-is-gaja)
* [Quick Start](#quick-start)
* [Requirements](#requirements)
* [Features](#features)
* [Configuration](#configuration)
* [Architecture](#architecture)
* [Troubleshooting](#troubleshooting)
* [Development](#development)

## What is Gaja?

Gaja is a **self-hosted AI home assistant** designed for:

* **Privacy**: Data remains exclusively on your hardware.
* **Modularity**: Easily extendable through plugins.
* **Multi-User Support**: Ideal for families and teams.
* **Real-time Communication**: Fast responses via WebSockets.
* **Flexible AI Providers**: Compatible with OpenAI, Anthropic, LMStudio, Ollama, etc.

## Quick Start

### Installation

```bash
git clone https://github.com/TymekTM/Gaja-Client gaja-client
cd gaja-client

# Install dependencies and launch setup wizard
python start.py --install-deps
```

### Running the Client

```bash
# Standard launch
python start.py

# Development mode
python start.py --dev

# Connect to a specific server
python start.py --server-host 192.168.1.100 --server-port 8001
```

## Requirements

### System Requirements

* **Python 3.11+** (3.13 recommended)
* **Windows 10/11** (primary)
* **1 GB Dedicated RAM**&#x20;
* **Microphone and speakers**
* **Internet connection** (initial setup, model downloads, cloud services)

### Recommended Hardware

* Quality USB or built-in microphone
* Fast internet connection
* SSD storage for optimal performance

## Features

### Voice Interaction

* **Wake Word Detection**: Activate by saying "Gaja" (customizable)
* **Natural Language Processing**: Intuitive interaction without specific commands

### Speech Recognition (ASR)

* **Faster Whisper**: Local, fast, privacy-focused ASR
* **OpenAI Whisper**: High-accuracy, cloud-based ASR
* **Multi-language Support**: Polish, English, German, French, and more
* **Automatic Language Detection**

### Text-to-Speech (TTS)

* **Edge TTS**: Microsoft Edge TTS, free, low quality
* **OpenAI TTS**: Premium cloud-based synthesis
* **CMS**: Localy runing high quality tts model **(Work in progres)**

### User Interface

* **System Tray**: Runs minimally in the background
* **Visual Overlay**: Optional status indicator **(Being reworked)**

## Configuration

The client is configured through `client_config.json`:

**Server Settings**

```json
{
  "server": {
    "host": "localhost",
    "port": 8001,
    "websocket_url": "ws://localhost:8001/ws",
    "timeout": 30,
    "reconnect_interval": 5,
    "max_reconnect_attempts": 10
  }
}
```

**Audio Configuration**

```json
{
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "input_device": "default",
    "output_device": "default"
  }
}
```

**Wake Word Settings**

```json
{
  "wake_word": {
    "enabled": true,
    "model": "gaja",
    "threshold": 0.5,
    "timeout": 30,
    "verification_threshold": 0.7
  }
}
```

## Architecture

### Components

* **Wake Word Detection**: Uses OpenWakeWord

* **Speech Recognition**: Faster Whisper (local), OpenAI Whisper (cloud)

* **Text-to-Speech**: Edge TTS (local), OpenAI TTS (cloud)

* **WebSocket Client**: Communication with GAJA Server

* **Visual Overlay**: Optional Rust-based status indicator

### Audio Pipeline

1. Audio monitoring
2. Wake word detection
3. Voice activity detection
4. Speech-to-text conversion
5. WebSocket-based command processing
6. Response playback

## Troubleshooting

### Common Issues

**Cannot connect to server**

```bash
# Verify server status
python start.py --server-host YOUR_HOST --force

# Connectivity check
telnet YOUR_HOST 8001
```

**Audio device not detected**

```bash
# Check available devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Update device ID in config ("input_device": device_id)
```

### Logs

* `logs/client_startup.log` – Initialization logs
* `logs/client_YYYY-MM-DD.log` – Daily operation logs
* `logs/setup_gui.log` – Setup logs

## Development

### Running in Development Mode

```bash
# Verbose logging
python start.py --dev

# Skip checks
python start.py --dev --skip-checks

# Custom config
python start.py --dev --config custom_config.json
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Test with `python start.py --dev`
4. Submit a pull request
