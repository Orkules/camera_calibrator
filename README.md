# Luminar Camera Calibration System

A helper system for technicians to calibrate Luminar cameras. This application provides a user interface for camera calibration operations including zoom, registration, focus, and gain adjustments.

## Features

- **Stream Display**: Real-time RTSP stream display with processing pipeline
- **Terminal Management**: Connection to three terminals (Linux, Debug, FPGA)
- **Direct Camera Control**: Full implementation of direct camera control via serial terminals
  - **Zoom Control**: Direct zoom adjustment (0-26) with real-time value updates
  - **Focus Control**: Manual focus control with motor position management (0 to motor max)
  - **Gain Control**: Gain adjustment (0-255) with real-time value synchronization
  - **UV Registration**: Four-directional arrow controls for offset and magnification (shift/stretch-compress modes)
- **Automatic Callback System**: Real-time value updates from camera responses
  - Automatic parsing of terminal responses based on config patterns
  - Background listeners update UI values automatically
  - Supports zoom, gain, focus, UV offset, and UV magnification callbacks
- **Console**: Real-time logging and data display with terminal communication tracking

## Setup Instructions

### 1. Camera Network Configuration

Before using the application, configure the camera for direct network connection:

#### Set Camera Static IP (Linux Terminal)
```bash
/opt/ofil/send.out IC_NETISETH0 STATIC 192.168.0.100 255.255.255.0 0.0.0.0
```

#### Set PC Adapter to Static IP (Windows CMD)
```cmd
netsh interface ip set address name="Ethernet" static 192.168.0.10 255.255.255.0 0.0.0.0
```

#### Enable RTSP Stream (Linux Terminal)
```bash
/opt/ofil/send.out IC_STRS1
```

#### Revert to DHCP After Use

**PC:**
```cmd
netsh interface ip set address name="Ethernet" dhcp
```

**Camera (Linux Terminal):**
```bash
/opt/ofil/send.out IC_NETIS ETH0 DHCP 0.0.0.0 0.0.0.0 0.0.0.0
```

### 2. Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration

Edit `config.yaml` to configure:
- RTSP stream URL (default: `rtsp://192.168.0.100:9079/vis`)
- COM ports for each terminal (Linux, Debug, FPGA)
- Terminal settings (baudrate, encoding, etc.)

### 4. Running the Application

```bash
python app.py
```

The application will start a web server on `http://localhost:5000`. Open this URL in your web browser to access the calibration interface.

## Usage

1. **Connect Terminals**: Click "Connect Terminals" to establish serial connections to all three terminals (Linux, Debug, FPGA)
2. **Start Stream**: Click "Start Stream" to begin RTSP video feed
3. **Calibration Controls**: Use the toolbar sections to adjust camera parameters:
   - **Zoom Station**: Use arrows to adjust zoom (0-26). Values update automatically from camera responses
   - **Registration**: Use directional arrows for UV image registration (offset and magnification)
   - **Focus Calibration**: Adjust focus manually (0 to motor max). Motor position is automatically detected
   - **Gain Calibration**: Control gain (0-255) with automatic value synchronization
4. **Automatic Updates**: Camera values are automatically synchronized via callback system - no manual refresh needed

## Project Structure

```
luminar_camera_calibration/
├── app.py                  # Flask web application
├── terminal_manager.py     # Terminal connection management
├── stream_processor.py     # RTSP stream processing
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── com_port_scanner.py    # COM port detection utility
├── luminar_eth_operation.txt  # Camera operation reference
└── templates/
    └── index.html          # Web UI template
```

## Code Organization

The code is organized into sections with clear headers:

- **Registration Processing**: Image registration functions
- **Focus Processing**: Focus-related processing
- **Gain Processing**: Gain-related processing
- **Zoom Processing**: Zoom-related processing

## Recent Updates

### ✅ Direct Camera Control Implementation
All camera control functionality is now fully implemented and operational:

- **Zoom Control**: Direct commands to Debug terminal (`MZS<val>`) with automatic response parsing (`MZR<val>`)
- **Focus Control**: Multi-step commands including auto-focus disable and motor position control via FPGA terminal
- **Gain Control**: Direct gain commands (`GAS<val>`) with response tracking (`GAR<val>`)
- **UV Registration**: Complete implementation of UV offset and magnification controls
- **Motor Max Detection**: Automatic motor maximum position detection for focus range validation

### ✅ Automatic Callback System
A comprehensive callback system has been implemented for automatic value synchronization:

- **Response Pattern Matching**: Configurable response patterns (e.g., `MZR<val>`, `MIOR-<val>`, `GAR<val>`)
- **Automatic Value Updates**: Background listeners automatically update global state variables when camera responds
- **Multi-Variable Support**: Single response can update multiple variables if needed
- **Validation Functions**: Optional validation functions ensure only valid values are accepted
- **Real-time Synchronization**: UI values stay synchronized with actual camera state

### ✅ Terminal Communication
- **Background Listeners**: Each terminal has a background thread listening for responses
- **Command Logging**: All sent commands and received responses are logged to console
- **Response Filtering**: Configurable ignore patterns for unwanted responses
- **Automatic Reconnection**: Lazy reconnection support for dropped connections

## Technical Details

### Callback Registration
The system automatically registers callbacks based on `config.yaml`:
- Commands section defines control commands and their expected responses
- Get values section defines query commands and response patterns
- Operations section supports multi-step operations (e.g., motor max detection)

### Value Update Flow
1. User action triggers command via API endpoint
2. Command sent to appropriate terminal (Debug/FPGA/Linux)
3. Background listener receives response
4. Response matched against registered patterns
5. Value extracted and validated
6. Global variable updated automatically
7. UI reflects new value on next refresh

All calibration functions are fully operational and communicate directly with the camera hardware.

## Troubleshooting

- **Stream not connecting**: Verify camera IP configuration and RTSP stream is enabled
- **Terminals not connecting**: Check COM ports in `config.yaml` match your system
- **No video display**: Ensure camera is connected via Ethernet cable and RTSP is enabled

## Future Development

- Smart calibration algorithms for automated adjustment
- Advanced image processing in the stream pipeline
- Data logging and export functionality
- Calibration presets and profiles
- Batch calibration operations
