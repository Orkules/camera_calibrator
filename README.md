# Luminar Camera Calibration System

A helper system for technicians to calibrate Luminar cameras. This application provides a user interface for camera calibration operations including zoom, registration, focus, and gain adjustments.

## Features

- **Stream Display**: Real-time RTSP stream display with processing pipeline
- **Terminal Management**: Connection to three terminals (Linux, Debug, FPGA)
- **Zoom Station**: Control zoom values (1-26)
- **Registration**: Four-directional arrow controls with shift/stretch-compress modes
- **Focus Calibration**: Manual focus control (0-75) with smart calibration
- **Gain Calibration**: Gain control (0-255) with CPS display and time interval input
- **Console**: Real-time logging and data display

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

1. **Connect Terminals**: Click "Connect Terminals" to establish serial connections
2. **Start Stream**: Click "Start Stream" to begin RTSP video feed
3. **Calibration Controls**: Use the toolbar sections to adjust camera parameters:
   - **Zoom Station**: Use arrows to adjust zoom (1-26)
   - **Registration**: Use directional arrows for image registration
   - **Focus Calibration**: Adjust focus manually or use smart calibration
   - **Gain Calibration**: Control gain with CPS monitoring

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

## Notes

- Currently, UI actions are implemented with placeholder functions
- Stream processing pipeline is ready for future implementation
- Terminal commands will be implemented in future updates
- All calibration functions pass through the stream processing pipeline

## Troubleshooting

- **Stream not connecting**: Verify camera IP configuration and RTSP stream is enabled
- **Terminals not connecting**: Check COM ports in `config.yaml` match your system
- **No video display**: Ensure camera is connected via Ethernet cable and RTSP is enabled

## Future Development

- Implementation of terminal commands for zoom, gain, focus, and registration
- Smart calibration algorithms
- Advanced image processing in the stream pipeline
- Data logging and export functionality

