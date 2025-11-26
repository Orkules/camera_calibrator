#!/usr/bin/env python3
"""
Luminar Camera Calibration System
Flask web application for camera calibration assistance
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import yaml
import logging
import base64
import re
import threading
import time
from pathlib import Path
from io import BytesIO
from collections import deque
from datetime import datetime

from terminal_manager import TerminalManager
from stream_processor import StreamProcessor

app = Flask(__name__)

# Global variables
stream_processor = None
terminal_manager = None
config = None

# State variables
zoom_value = 0  # Changed from 1 to 0 (range is 0-26)
focus_value = 0
gain_value = 0
gain_min_value = 0  # Minimum gain value
gain_medium_value = 0  # Medium gain value
gain_max_value = 0  # Maximum gain value
registration_mode = "shift"  # "shift" or "stretch_compress"
reg_offset_x = 0
reg_offset_y = 0
reg_stretch_x = 0
reg_stretch_y = 0
gain_mode = "min"  # "min" or "max" (kept for backward compatibility)
cps_value = 0
time_interval_value = ""
motor_max = None  # Motor maximum position value
camera_serial = ""  # Camera serial number
technician_name = ""  # Technician name
auto_focus_enabled = False  # Auto focus mode state
vis_position = 0  # VIS position value (for future use)
uv_position = 0  # UV position value (for future use)
zoom_status = ""  # Zoom status value (for future use)
distance = 0  # Distance value (for future use)
distance_to_target = 1  # Distance to target (1-100)
uv_motor_position = 0  # UV motor position value (from get_uv_motor_position)
uv_vis_focus_info = {}  # Dictionary with uv_motor_pos, vis_focus_point, zoom_status, distance
camera_mode = 0  # Camera mode: 1=VIS ONLY, 2=UV ONLY, 3=UV & VIS

# Interval update thread
update_thread = None
update_thread_running = False

# Console messages queue
console_messages = deque(maxlen=1000)  # Keep last 1000 messages
console_lock = threading.Lock()


def load_config():
    """Load configuration from YAML file."""
    global config
    config_path = Path(__file__).parent / 'config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Debug: Log config structure
        if config:
            logging.debug(f"Config loaded. Available sections: {list(config.keys())}")
            if 'get_values' in config:
                get_values_keys = list(config['get_values'].keys())
                logging.debug(f"get_values section contains: {get_values_keys}")
        
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}", exc_info=True)
        return {}


def add_console_message(message: str):
    """Add a message to the console queue."""
    global console_messages, console_lock
    with console_lock:
        timestamp = datetime.now().strftime('%H:%M:%S')
        console_messages.append(f"[{timestamp}] {message}")
    
    # Also log to Python logging for debugging
    logging.debug(f"Console: {message}")


def load_calibration_file_header():
    """Load camera_serial, technician_name, and distance_to_target from calibration file."""
    global camera_serial, technician_name, distance_to_target
    
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            
            if 'camera_serial' in data:
                camera_serial = data['camera_serial']
                logging.info(f"Loaded camera_serial from file: {camera_serial}")
            
            if 'technician_name' in data:
                technician_name = data['technician_name']
                logging.info(f"Loaded technician_name from file: {technician_name}")
            
            if 'distance_to_target' in data:
                distance_to_target = data['distance_to_target']
                logging.info(f"Loaded distance_to_target from file: {distance_to_target}")
        except Exception as e:
            logging.warning(f"Error loading calibration file header: {e}")


def init_components():
    """Initialize terminal manager and stream processor."""
    global stream_processor, terminal_manager, config
    
    if config is None:
        config = load_config()
    
    # Load calibration file header (camera_serial, technician_name)
    load_calibration_file_header()
    
    # Pass log callback to terminal manager
    terminal_manager = TerminalManager(config, log_callback=add_console_message)
    
    stream_processor = StreamProcessor(
        rtsp_url=config.get('stream', {}).get('rtsp_url', 'rtsp://192.168.0.100:9079/vis')
    )


def register_terminal_callbacks():
    """Register callbacks for automatic parsing of terminal responses based on config."""
    global terminal_manager, config, zoom_value, gain_value, focus_value, motor_max
    global reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y, uv_vis_focus_info
    global gain_min_value, gain_medium_value, gain_max_value, camera_mode, uv_motor_position
    
    if terminal_manager is None or config is None:
        return
    
    # Mapping from config key to (response_pattern, global_variable_name, validation_function)
    # This maps config entries to the global variables they should update
    callback_mapping = {
        'zoom_command': ('zoom_value', None),
        'gain_command': ('gain_value', None),
        'focus_command': ('focus_value', None),
        'uv_offset_x_command': ('reg_offset_x', None),
        'uv_offset_y_command': ('reg_offset_y', None),
        'uv_magnify_x_command': ('reg_stretch_x', None),
        'uv_magnify_y_command': ('reg_stretch_y', None),
        'get_zoom_value': ('zoom_value', None),
        'get_gain_value': ('gain_value', None),
        'get_focus_value': ('focus_value', None),
        'get_uv_offset_x_value': ('reg_offset_x', None),
        'get_uv_offset_y_value': ('reg_offset_y', None),
        'get_uv_magnify_x_value': ('reg_stretch_x', None),
        'get_uv_magnify_y_value': ('reg_stretch_y', None),
        'set_min_gain_command': ('gain_min_value', None),
        'set_medium_gain_command': ('gain_medium_value', None),
        'set_max_gain_command': ('gain_max_value', None),
        'set_camera_to_uv_command': ('camera_mode', None),
        'set_camera_to_vis_command': ('camera_mode', None),
        'set_camera_to_combined_command': ('camera_mode', None),
        'get_camera_mode_value': ('camera_mode', None),
        'get_uv_motor_position': ('uv_motor_position', None),
    }
    
    # Collect all response patterns from config
    response_to_globals = {}  # response_pattern -> list of (global_var_name, validation_func)
    
    # Read from commands section
    if 'commands' in config:
        for cmd_name, cmd_steps in config['commands'].items():
            if cmd_name in callback_mapping:
                global_var_name, validation_func = callback_mapping[cmd_name]
                for step in cmd_steps:
                    response_pattern = step.get('response')
                    if response_pattern:
                        if response_pattern not in response_to_globals:
                            response_to_globals[response_pattern] = []
                        response_to_globals[response_pattern].append((global_var_name, validation_func))
    
    # Read from get_values section
    if 'get_values' in config:
        for get_name, get_steps in config['get_values'].items():
            if get_name in callback_mapping:
                global_var_name, validation_func = callback_mapping[get_name]
                for step in get_steps:
                    response_pattern = step.get('response')
                    if response_pattern:
                        if response_pattern not in response_to_globals:
                            response_to_globals[response_pattern] = []
                        response_to_globals[response_pattern].append((global_var_name, validation_func))
    
    # Handle auto_responses with special handlers
    handler_functions = {
        'parse_uv_vis_focus_info': parse_uv_vis_focus_info,
    }
    
    if 'auto_responses' in config:
        for auto_name, auto_config in config['auto_responses'].items():
            if isinstance(auto_config, dict):
                response_pattern = auto_config.get('response')
                handler_name = auto_config.get('handler')
                
                if response_pattern and handler_name:
                    if handler_name in handler_functions:
                        handler_func = handler_functions[handler_name]
                        
                        # Create callback that calls the special handler
                        # Simple check: if message starts with "set m", send to handler
                        pattern_prefix = "set m"  # Simple prefix for terminal_manager to match
                        
                        def callback(terminal_name: str, response: str):
                            # Simple check: if message starts with "set m", send to handler
                            if response.strip().startswith("set m"):
                                handler_func(response)
                        
                        # Register with simple prefix pattern so terminal_manager can match it
                        terminal_manager.register_response_callback(pattern_prefix, callback)
                        logging.info(f"Registered auto_response '{auto_name}' with pattern '{response_pattern}' -> handler '{handler_name}'")
                    else:
                        logging.warning(f"Handler function '{handler_name}' not found for auto_response '{auto_name}'")
    
    # Create and register callbacks for each response pattern
    for response_pattern, global_vars in response_to_globals.items():
        def make_callback(pattern, vars_list):
            def callback(terminal_name: str, response: str):
                parsed = parse_response_value(response, pattern)
                if parsed is not None:
                    for global_var_name, validation_func in vars_list:
                        if validation_func is None or validation_func(parsed):
                            # Update the global variable
                            if global_var_name == 'zoom_value':
                                globals()['zoom_value'] = parsed
                                logging.info(f"Auto-updated zoom value: {parsed} from {response}")
                            elif global_var_name == 'gain_value':
                                globals()['gain_value'] = parsed
                                logging.info(f"Auto-updated gain value: {parsed} from {response}")
                            elif global_var_name == 'focus_value':
                                globals()['focus_value'] = parsed
                                logging.info(f"Auto-updated focus value: {parsed} from {response}")
                            elif global_var_name == 'uv_motor_position':
                                old_value = globals()['uv_motor_position']
                                globals()['uv_motor_position'] = parsed
                                logging.info(f"Auto-updated UV motor position: {old_value} -> {parsed} from {response}")
                                add_console_message(f"UV motor position updated: {parsed}")
                            elif global_var_name == 'reg_offset_x':
                                globals()['reg_offset_x'] = parsed
                                logging.info(f"Auto-updated UV offset X: {parsed} from {response}")
                            elif global_var_name == 'reg_offset_y':
                                globals()['reg_offset_y'] = parsed
                                logging.info(f"Auto-updated UV offset Y: {parsed} from {response}")
                            elif global_var_name == 'reg_stretch_x':
                                globals()['reg_stretch_x'] = parsed
                                logging.info(f"Auto-updated UV magnify X: {parsed} from {response}")
                            elif global_var_name == 'reg_stretch_y':
                                globals()['reg_stretch_y'] = parsed
                                logging.info(f"Auto-updated UV magnify Y: {parsed} from {response}")
                            elif global_var_name == 'gain_min_value':
                                globals()['gain_min_value'] = parsed
                                logging.info(f"Auto-updated gain min value: {parsed} from {response}")
                            elif global_var_name == 'gain_medium_value':
                                globals()['gain_medium_value'] = parsed
                                logging.info(f"Auto-updated gain medium value: {parsed} from {response}")
                            elif global_var_name == 'gain_max_value':
                                globals()['gain_max_value'] = parsed
                                logging.info(f"Auto-updated gain max value: {parsed} from {response}")
                            elif global_var_name == 'camera_mode':
                                old_mode = globals()['camera_mode']
                                globals()['camera_mode'] = parsed
                                logging.info(f"Auto-updated camera mode: {old_mode} -> {parsed} from {response}")
                                add_console_message(f"Camera mode updated: {parsed}")
            return callback
        
        callback_func = make_callback(response_pattern, global_vars)
        terminal_manager.register_response_callback(response_pattern, callback_func)
        logging.info(f"Registered callback for pattern '{response_pattern}' -> {[v[0] for v in global_vars]}")
    
    logging.info(f"Registered {len(response_to_globals)} terminal response callbacks from config")


def encode_frame(frame):
    """Encode frame to base64 for HTML display"""
    if frame is None:
        return ""
    
    # OpenCV typically reads frames in BGR format, but some RTSP streams/decoders
    # may output RGB directly. Since VLC shows correct colors, we should try
    # using the frame as-is first (assuming it's already RGB from the decoder)
    try:
        # Check if frame is grayscale (2D) or color (3D)
        if len(frame.shape) == 2:
            # Grayscale - convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            # Color image - try using frame as-is first (may already be RGB)
            # If colors are wrong, the stream decoder might be outputting RGB already
            # OpenCV's VideoCapture with FFMPEG backend sometimes outputs RGB directly
            frame_rgb = frame  # Use as-is - assume already RGB from decoder
        else:
            # Unknown format, use as-is
            frame_rgb = frame
    except Exception as e:
        logging.warning(f"Color conversion error, using frame as-is: {e}")
        frame_rgb = frame
    
    # Encode to JPEG with better quality for smoother display
    _, buffer = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Convert to base64
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"


# ============================================
# HELPER FUNCTIONS FOR COMMAND EXECUTION
# ============================================
def parse_response_value(response: str, response_pattern: str) -> int:
    """
    Parse a value from terminal response.
    Example: parse_response_value("MZR15", "MZR<val>") returns 15
    Example: parse_response_value("MIOR-1234", "MIOR-<val>") returns 1234
    
    Args:
        response: The response string from terminal
        response_pattern: Pattern like "MZR<val>" or "MIOR-<val>" or "GAR<val>"
    
    Returns:
        Parsed integer value, or None if parsing fails
    """
    if not response or not response_pattern:
        return None
    
    # Replace <val> with regex pattern to capture number (including negative)
    # Handle patterns like "MIOR-<val>" or "MZR<val>"
    pattern = response_pattern.replace('<val>', r'(-?\d+)')
    
    match = re.search(pattern, response)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    
    return None


def parse_uv_vis_focus_info(response: str):
    """
    Parse the special "set m<val1> <val2> narrow/wide <val3>" message.
    Updates the global uv_vis_focus_info dictionary.
    
    Args:
        response: The full response string, e.g., "set m10 20 narrow 30" or "set m10 20 wide 30"
    """
    global uv_vis_focus_info
    
    # Pattern: "set m<val1> <val2> narrow/wide <val3>"
    # Example: "set m10 20 narrow 30" or "set m10 20 wide 30"
    pattern = r'set m(-?\d+)\s+(-?\d+)\s+(narrow|wide)\s+(-?\d+)'
    
    match = re.search(pattern, response)
    if match:
        try:
            val1 = int(match.group(1))  # uv_motor_pos
            val2 = int(match.group(2))  # vis_focus_point
            zoom_status = match.group(3)  # "narrow" or "wide"
            val3 = int(match.group(4))  # distance
            
            uv_vis_focus_info = {
                'uv_motor_pos': val1,
                'vis_focus_point': val2,
                'zoom_status': zoom_status,
                'distance': val3
            }
            
            logging.info(f"Updated uv_vis_focus_info: {uv_vis_focus_info} from response: {response}")
        except (ValueError, IndexError) as e:
            logging.error(f"Error parsing uv_vis_focus_info from response '{response}': {e}")
    else:
        logging.warning(f"Failed to match pattern in response: {response}")


def execute_command_from_config(command_name: str, value: int = None):
    """
    Execute a command from config.yaml and return the response.
    
    Args:
        command_name: Name of command in config (e.g., "zoom_command")
        value: Value to replace <val> with (if needed)
    
    Returns:
        Tuple of (success, response_string, parsed_value)
        parsed_value is None if response doesn't contain a value
    """
    global terminal_manager, config
    
    if terminal_manager is None or config is None:
        logging.error("Terminal manager or config not initialized")
        return False, None, None
    
    command_config = config.get('commands', {}).get(command_name)
    if not command_config:
        logging.error(f"Command '{command_name}' not found in config")
        return False, None, None
    
    # Execute each step in the command sequence
    last_response = None
    last_response_pattern = None
    
    for step in command_config:
        terminal_name = step.get('terminal')
        command_template = step.get('command', '')
        response_pattern = step.get('response')
        
        # Replace <val> with actual value if provided
        if value is not None and '<val>' in command_template:
            command = command_template.replace('<val>', str(value))
        else:
            command = command_template
        
        # Send command to terminal
        terminal = terminal_manager.get_terminal(terminal_name)
        if not terminal:
            logging.error(f"Terminal '{terminal_name}' not found")
            return False, None, None
        
        # Check if terminal is connected
        if not terminal.serial_conn or not terminal.serial_conn.is_open:
            logging.error(f"Terminal '{terminal_name}' is not connected (port: {terminal.port})")
            return False, None, None
        
        # Send command (background listener will catch the response)
        terminal.send_command(command)
        
        # For backward compatibility, we still track the response pattern
        # but the actual response will be caught by the background listener
        last_response = None
        last_response_pattern = response_pattern
    
    # Parse value from response if pattern is provided
    parsed_value = None
    if last_response and last_response_pattern:
        parsed_value = parse_response_value(last_response, last_response_pattern)
    
    return True, last_response, parsed_value


def execute_operation_from_config(operation_name: str):
    """
    Execute an operation script from config.yaml.
    
    Args:
        operation_name: Name of operation in config (e.g., "motor_max_script")
    
    Returns:
        Dictionary with results, or None if failed
    """
    global terminal_manager, config
    
    if terminal_manager is None or config is None:
        logging.error("Terminal manager or config not initialized")
        return None
    
    operation_config = config.get('operations', {}).get(operation_name)
    if not operation_config:
        logging.error(f"Operation '{operation_name}' not found in config")
        return None
    
    results = {}
    last_response = None
    last_response_pattern = None
    
    # Execute each step in the operation sequence
    for step in operation_config:
        # Check if this step is a delay
        if 'delay' in step:
            delay_ms = step.get('delay', 0)
            if delay_ms > 0:
                import time
                delay_seconds = delay_ms / 1000.0
                logging.info(f"Waiting {delay_ms}ms ({delay_seconds}s) before next step...")
                time.sleep(delay_seconds)
            continue
        
        terminal_name = step.get('terminal')
        command = step.get('command', '')
        response_pattern = step.get('response')
        
        terminal = terminal_manager.get_terminal(terminal_name)
        if not terminal:
            logging.error(f"Terminal '{terminal_name}' not found")
            return None
        
        # Check if terminal is connected
        if not terminal.serial_conn or not terminal.serial_conn.is_open:
            logging.error(f"Terminal '{terminal_name}' is not connected (port: {terminal.port})")
            return None
        
        # Send command (background listener will catch the response)
        terminal.send_command(command)
        
        # For backward compatibility, we still track the response pattern
        # but the actual response will be caught by the background listener
        last_response = None
        last_response_pattern = response_pattern
    
    # Parse value from last response if pattern is provided
    parsed_value = None
    if last_response and last_response_pattern:
        parsed_value = parse_response_value(last_response, last_response_pattern)
        results['value'] = parsed_value
        results['response'] = last_response
    
    return results


def find_motor_max() -> int:
    """
    Find motor maximum position by executing motor_max_script.
    Waits for callback to update uv_motor_position from the response.
    Then sets motor_max = uv_motor_position.
    
    Returns:
        Motor max value as integer, or None if failed
    """
    global motor_max, terminal_manager, config, uv_motor_position
    
    if terminal_manager is None or config is None:
        logging.error("Terminal manager or config not initialized")
        return None
    
    logging.info("Finding motor max position...")
    add_console_message("Finding motor max position...")
    
    operation_config = config.get('operations', {}).get('motor_max_script')
    if not operation_config:
        logging.error("motor_max_script not found in config")
        add_console_message("ERROR: motor_max_script not found in config")
        return None
    
    # Execute each step in the operation sequence (send commands, background listener will catch responses)
    for step in operation_config:
        # Handle delay
        if 'delay' in step:
            delay_ms = step.get('delay', 0)
            if delay_ms > 0:
                delay_seconds = delay_ms / 1000.0
                logging.info(f"Waiting {delay_ms}ms ({delay_seconds}s) before next step...")
                time.sleep(delay_seconds)
            continue
        
        terminal_name = step.get('terminal')
        command = step.get('command', '')
        response_pattern = step.get('response')
        
        terminal = terminal_manager.get_terminal(terminal_name)
        if not terminal:
            logging.error(f"Terminal '{terminal_name}' not found")
            return None
        
        if not terminal.serial_conn or not terminal.serial_conn.is_open:
            logging.error(f"Terminal '{terminal_name}' is not connected")
            return None
        
        # Send command - background listener will catch the response
        if response_pattern:
            logging.info(f"Sending command '{command}' to terminal '{terminal_name}' (expecting pattern: {response_pattern})...")
        else:
            logging.info(f"Sending command '{command}' to terminal '{terminal_name}' (no response expected)...")
        add_console_message(f"Sending: {command} to {terminal_name}")
        terminal.send_command(command)
        
        # Only wait for response if this step expects one
        if response_pattern and response_pattern != "None":
            # Wait for callback to update uv_motor_position (max 20 seconds, script has 10s delay)
            max_wait_time = 20.0
            start_time = time.time()
            initial_uv_motor_position = uv_motor_position
            logging.info(f"Waiting for UV motor position update (max {max_wait_time}s)...")
            add_console_message(f"Waiting for UV motor position: {response_pattern}")
            while time.time() - start_time < max_wait_time:
                # Check if uv_motor_position was updated
                if uv_motor_position is not None and uv_motor_position > 0:
                    if initial_uv_motor_position != uv_motor_position:
                        # Update motor_max with uv_motor_position value
                        motor_max = uv_motor_position
                        logging.info(f"Motor max set to UV motor position: {motor_max}")
                        add_console_message(f"Motor max calculated: {motor_max}")
                        # Update calibration file asynchronously
                        def update_file_async():
                            time.sleep(0.1)  # Small delay to avoid blocking
                            initialize_calibration_file()
                        threading.Thread(target=update_file_async, daemon=True).start()
                        return motor_max
                elapsed = time.time() - start_time
                if int(elapsed) % 2 == 0 and elapsed > 0.5:  # Log every 2 seconds
                    logging.debug(f"Still waiting for UV motor position... (elapsed: {elapsed:.1f}s, current: {uv_motor_position})")
                time.sleep(0.2)  # Check every 200ms
            
            logging.warning(f"Timeout waiting for UV motor position. Current value: {uv_motor_position}")
            add_console_message(f"WARNING: Timeout waiting for UV motor position. Current value: {uv_motor_position}")
    
    # If we got a value, use it anyway
    if uv_motor_position is not None and uv_motor_position > 0:
        motor_max = uv_motor_position
        logging.info(f"Motor max set to UV motor position: {motor_max}")
        add_console_message(f"Motor max calculated: {motor_max}")
        initialize_calibration_file()
        return motor_max
    
    logging.warning("Motor max calculation failed - no valid UV motor position received")
    return None


def terminal_initialization():
    """
    Initialize terminal values: get zoom, gain, focus, UV registration from terminal and find motor max.
    Updates global variables and UI.
    """
    global zoom_value, gain_value, focus_value, motor_max, reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y, camera_mode
    
    logging.info("Starting terminal initialization...")
    
    # Execute unsgrad_command_script if configured
    if config and 'operations' in config and 'unsgrad_command_script' in config['operations']:
        logging.info("Executing unsgrad_command_script...")
        add_console_message("Executing unsgrad_command_script...")
        result = execute_operation_from_config('unsgrad_command_script')
        if result is not None:
            logging.info("unsgrad_command_script executed successfully")
            add_console_message("unsgrad_command_script executed successfully")
        else:
            logging.warning("unsgrad_command_script execution failed or returned no result")
            add_console_message("WARNING: unsgrad_command_script execution failed")
    
    # Execute set_camera_ip_command if configured
    if config and 'commands' in config and 'set_camera_ip_command' in config['commands']:
        logging.info("Executing set_camera_ip_command...")
        add_console_message("Setting camera IP address...")
        success, response, parsed_value = execute_command_from_config('set_camera_ip_command')
        if success:
            logging.info("set_camera_ip_command executed successfully")
            add_console_message("Camera IP address set successfully")
        else:
            logging.warning("set_camera_ip_command execution failed")
            add_console_message("WARNING: Failed to set camera IP address")
    
    # Get zoom value from terminal
    get_value_from_config('get_zoom_value')
    import time
    time.sleep(0.1)  # Give background listener time to process
    logging.info(f"Initialized zoom value: {zoom_value}")
    
    # Get gain value from terminal
    get_value_from_config('get_gain_value')
    time.sleep(0.1)
    logging.info(f"Initialized gain value: {gain_value}")
    
    # Get focus value from terminal
    get_value_from_config('get_focus_value')
    time.sleep(0.1)
    logging.info(f"Initialized focus value: {focus_value}")
    
    # Get UV offset X value
    get_value_from_config('get_uv_offset_x_value')
    time.sleep(0.1)
    logging.info(f"Initialized UV offset X: {reg_offset_x}")
    
    # Get UV offset Y value
    get_value_from_config('get_uv_offset_y_value')
    time.sleep(0.1)
    logging.info(f"Initialized UV offset Y: {reg_offset_y}")
    
    # Get UV magnify X value
    get_value_from_config('get_uv_magnify_x_value')
    time.sleep(0.1)
    logging.info(f"Initialized UV magnify X: {reg_stretch_x}")
    
    # Get UV magnify Y value
    get_value_from_config('get_uv_magnify_y_value')
    time.sleep(0.1)
    logging.info(f"Initialized UV magnify Y: {reg_stretch_y}")
    
    # Get camera mode value
    get_value_from_config('get_camera_mode_value')
    time.sleep(0.1)
    logging.info(f"Initialized camera mode: {camera_mode}")
    
    # Find motor max position (waits for response)
    motor_max_result = find_motor_max()
    if motor_max_result is not None and motor_max_result > 0:
        logging.info(f"Initialized motor max: {motor_max}")
        # Initialize calibration file with motor_max (only if we got a valid value)
        initialize_calibration_file()
    else:
        logging.warning(f"Motor max initialization failed or returned invalid value: {motor_max}")
    
    logging.info("Terminal initialization completed")


def save_calibration_to_file(calibration_type: str, values: dict):
    """
    Save calibration data to YAML file.
    For 'focus' and 'registration' types: If entry of same type, zoom value AND distance_to_target exists, it will be overwritten.
    Otherwise, a new entry will be added (allowing multiple entries of same type with different zoom values or distance_to_target).
    For other types: If entry of same type AND zoom value exists, it will be overwritten.
    
    Args:
        calibration_type: Type of calibration ('zoom', 'focus', 'gain', 'registration')
        values: Dictionary with calibration values including zoom_value and distance_to_target (for focus/registration)
    """
    from collections import OrderedDict
    global motor_max, zoom_value, distance_to_target
    
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    
    # Load existing data or create new structure
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Error loading calibration file: {e}, creating new file")
            data = {}
    else:
        data = {}
    
    # Initialize structure if needed
    if 'motor_max' not in data:
        data['motor_max'] = motor_max if motor_max is not None else 0
    
    if 'distance_to_target' not in data:
        data['distance_to_target'] = distance_to_target
    
    if 'calibrations' not in data:
        data['calibrations'] = []
    
    # Create calibration entry (without timestamp)
    entry = {
        'type': calibration_type,
        'zoom': zoom_value,
        **values  # Add all other values
    }
    
    # For focus and registration, also include distance_to_target in the entry
    if calibration_type in ['focus', 'registration']:
        entry['distance_to_target'] = distance_to_target
    
    # Find existing entry and replace it, or add new one
    found = False
    for i, existing_entry in enumerate(data['calibrations']):
        if existing_entry.get('type') == calibration_type:
            # For focus and registration, check both zoom and distance_to_target
            if calibration_type in ['focus', 'registration']:
                existing_zoom = existing_entry.get('zoom')
                existing_distance = existing_entry.get('distance_to_target')
                if existing_zoom == zoom_value and existing_distance == distance_to_target:
                    data['calibrations'][i] = entry  # Overwrite existing entry
                    found = True
                    break
            else:
                # For other types, check only zoom
                if existing_entry.get('zoom') == zoom_value:
                    data['calibrations'][i] = entry  # Overwrite existing entry
                    found = True
                    break
    
    if not found:
        # Add new entry if not found
        data['calibrations'].append(entry)
    
    # Create ordered dict with header fields first
    ordered_data = OrderedDict()
    if 'camera_serial' in data:
        ordered_data['camera_serial'] = data['camera_serial']
    if 'technician_name' in data:
        ordered_data['technician_name'] = data['technician_name']
    ordered_data['motor_max'] = data.get('motor_max', 0)
    if 'distance_to_target' in data:
        ordered_data['distance_to_target'] = data.get('distance_to_target', 1)
    ordered_data['calibrations'] = data.get('calibrations', [])
    
    # Save to file
    try:
        # Ensure directory exists
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(calibration_file, 'w', encoding='utf-8') as f:
            yaml.dump(dict(ordered_data), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        if calibration_type in ['focus', 'registration']:
            logging.info(f"Calibration saved: {calibration_type} at zoom {zoom_value}, distance_to_target {distance_to_target}")
        else:
            logging.info(f"Calibration saved: {calibration_type} at zoom {zoom_value}")
        return True
    except Exception as e:
        logging.error(f"Error saving calibration file: {e}", exc_info=True)
        return False


def update_calibration_file_header(camera_serial: str = None, technician_name: str = None):
    """Update header fields in calibration file (camera_serial, technician_name, distance_to_target)."""
    from collections import OrderedDict
    global distance_to_target
    
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    
    # Load existing data or create new structure
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Error loading calibration file: {e}, creating new file")
            data = {}
    else:
        data = {}
    
    # Initialize structure if needed
    if 'motor_max' not in data:
        data['motor_max'] = 0
    
    if 'distance_to_target' not in data:
        data['distance_to_target'] = distance_to_target
    
    if 'calibrations' not in data:
        data['calibrations'] = []
    
    # Update fields if provided
    if camera_serial is not None:
        data['camera_serial'] = camera_serial
        logging.info(f"Updated camera_serial in calibration file: {camera_serial}")
    
    if technician_name is not None:
        data['technician_name'] = technician_name
        logging.info(f"Updated technician_name in calibration file: {technician_name}")
    
    # Always update distance_to_target from global variable
    data['distance_to_target'] = distance_to_target
    
    # Create ordered dict with header fields first
    ordered_data = OrderedDict()
    if 'camera_serial' in data:
        ordered_data['camera_serial'] = data['camera_serial']
    if 'technician_name' in data:
        ordered_data['technician_name'] = data['technician_name']
    ordered_data['motor_max'] = data.get('motor_max', 0)
    ordered_data['distance_to_target'] = data.get('distance_to_target', 1)
    ordered_data['calibrations'] = data.get('calibrations', [])
    
    # Save to file
    try:
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(calibration_file, 'w', encoding='utf-8') as f:
            yaml.dump(dict(ordered_data), f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        logging.error(f"Error updating calibration file header: {e}", exc_info=True)
        return False


def initialize_calibration_file():
    """Initialize calibration file with motor_max and distance_to_target values."""
    global motor_max, distance_to_target
    
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    
    # Use 0 as default if motor_max is not available yet
    current_motor_max = motor_max if (motor_max is not None and motor_max > 0) else 0
    
    # Load existing data or create new structure
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            # Update motor_max if it changed or if it's 0 and we have a new value
            updated = False
            if 'motor_max' not in data or (data['motor_max'] != current_motor_max and current_motor_max > 0):
                data['motor_max'] = current_motor_max
                updated = True
            # Update distance_to_target
            if 'distance_to_target' not in data or data.get('distance_to_target') != distance_to_target:
                data['distance_to_target'] = distance_to_target
                updated = True
            # Ensure calibrations list exists
            if 'calibrations' not in data:
                data['calibrations'] = []
            # Save if updated
            if updated:
                with open(calibration_file, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                logging.info(f"Calibration file updated: motor_max={current_motor_max}, distance_to_target={distance_to_target}")
            return
        except Exception as e:
            logging.warning(f"Error loading calibration file: {e}, creating new file")
    
    # Initialize new file with motor_max and distance_to_target
    data = {
        'motor_max': current_motor_max,
        'distance_to_target': distance_to_target,
        'calibrations': []
    }
    
    try:
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(calibration_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logging.info(f"Calibration file initialized with motor_max: {current_motor_max}, distance_to_target: {distance_to_target}")
    except Exception as e:
        logging.error(f"Error initializing calibration file: {e}", exc_info=True)


def get_value_from_config(get_command_name: str) -> int:
    """
    Get a value from terminal using get_values config.
    
    Args:
        get_command_name: Name of get command in config (e.g., "get_zoom_value")
    
    Returns:
        Parsed integer value, or None if failed
    """
    global terminal_manager, config
    
    if terminal_manager is None or config is None:
        logging.debug("Terminal manager or config not initialized")
        return None
    
    # Check if get_values section exists
    if 'get_values' not in config:
        logging.debug("'get_values' section not found in config")
        return None
    
    get_config = config.get('get_values', {}).get(get_command_name)
    if not get_config:
        # Log available keys for debugging
        available_keys = list(config.get('get_values', {}).keys())
        logging.debug(f"Get command '{get_command_name}' not found in config. Available keys: {available_keys}")
        return None
    
    # Execute the get command
    for step in get_config:
        terminal_name = step.get('terminal')
        command = step.get('command', '')
        response_pattern = step.get('response')
        
        terminal = terminal_manager.get_terminal(terminal_name)
        if not terminal:
            logging.debug(f"Terminal '{terminal_name}' not found in terminal manager")
            return None
        
        # Check if terminal is connected
        if not terminal.serial_conn or not terminal.serial_conn.is_open:
            logging.debug(f"Terminal '{terminal_name}' is not connected (port: {terminal.port})")
            return None
        
        logging.info(f"[DEBUG] Sending command '{command}' to terminal '{terminal_name}' for '{get_command_name}'")
        # Send command - background listener will catch the response and update via callbacks
        # For initialization, we wait a bit for the response
        terminal.send_command(command)
        
        # Wait a short time for response (background listener should catch it quickly)
        import time
        time.sleep(0.3)  # Give background listener time to catch response
        
        # The value should now be updated via callback, but we can't return it here
        # This function is mainly for initialization - the background listener handles updates
        logging.info(f"[DEBUG] Command sent, background listener will handle response")
        return None  # Background listener will update values via callbacks


@app.route('/')
def index():
    """Main page with video display and controls"""
    return render_template('index.html')


# ============================================
# STREAM CONTROL FUNCTIONS
# ============================================
@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start RTSP stream"""
    global stream_processor
    
    if stream_processor is None:
        init_components()
    
    data = request.get_json()
    rtsp_url = data.get('rtsp_url', stream_processor.rtsp_url)
    
    if stream_processor.start_stream():
        return jsonify({'success': True, 'message': 'Stream started successfully'})
    else:
        return jsonify({'success': False, 'error': 'Could not open RTSP stream'})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop RTSP stream"""
    global stream_processor
    
    if stream_processor is not None:
        stream_processor.stop_stream()
    
    return jsonify({'success': True, 'message': 'Stream stopped'})


@app.route('/get_frame')
def get_frame_route():
    """Get current frame for display - DIRECT from camera, NO processing delays"""
    global stream_processor
    
    if stream_processor is None or not stream_processor.is_streaming:
        return jsonify({'frame': ""})
    
    # Get frame directly - no processing, no delays
    frame = stream_processor.get_frame()
    
    if frame is None:
        return jsonify({'frame': ""})
    
    # Encode and return immediately - original frame only
    return jsonify({
        'frame': encode_frame(frame)
    })


@app.route('/connect_terminals', methods=['POST'])
def connect_terminals():
    """Connect to all terminals"""
    global terminal_manager
    
    if terminal_manager is None:
        init_components()
    
    if terminal_manager.connect_all():
        # Register callbacks for automatic value updates (after connection)
        register_terminal_callbacks()
        
        # Start background listener for automatic response parsing
        terminal_manager.start_listener()
        
        # Initialize terminal values after successful connection
        terminal_initialization()
        return jsonify({'success': True, 'message': 'Terminals connected successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to connect to terminals'})


# ============================================
# ZOOM FUNCTIONS
# ============================================
@app.route('/zoom_increase', methods=['POST'])
def zoom_increase():
    """Increase zoom value"""
    global zoom_value
    
    if zoom_value >= 26:
        return jsonify({'success': False, 'error': 'Zoom at maximum'})
    
    new_value = zoom_value + 1
    
    # Send command to terminal - background listener will update zoom_value via callback
    success, response, parsed_value = execute_command_from_config('zoom_command', new_value)
    
    if success:
        # Command sent - background listener will update zoom_value when response arrives
        return jsonify({'success': True, 'zoom_value': zoom_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send zoom command to terminal'})


@app.route('/zoom_decrease', methods=['POST'])
def zoom_decrease():
    """Decrease zoom value"""
    global zoom_value
    
    if zoom_value <= 0:
        return jsonify({'success': False, 'error': 'Zoom at minimum'})
    
    new_value = zoom_value - 1
    
    # Send command to terminal - background listener will update zoom_value via callback
    success, response, parsed_value = execute_command_from_config('zoom_command', new_value)
    
    if success:
        # Command sent - background listener will update zoom_value when response arrives
        return jsonify({'success': True, 'zoom_value': zoom_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send zoom command to terminal'})


@app.route('/set_zoom', methods=['POST'])
def set_zoom():
    """Set zoom to absolute value"""
    global zoom_value
    
    data = request.get_json()
    target_value = data.get('value')
    
    if target_value is None:
        return jsonify({'success': False, 'error': 'Value not provided'})
    
    if target_value < 0 or target_value > 26:
        return jsonify({'success': False, 'error': 'Zoom value out of range (0-26)'})
    
    # Send command to terminal with absolute target value
    success, response, parsed_value = execute_command_from_config('zoom_command', target_value)
    
    if success:
        # Command sent - background listener will update zoom_value when response arrives
        return jsonify({'success': True, 'zoom_value': zoom_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send zoom command to terminal'})


@app.route('/zoom_ok', methods=['POST'])
def zoom_ok():
    """Save zoom value to calibration file"""
    global zoom_value
    
    values = {
        'zoom_value': zoom_value
    }
    
    if save_calibration_to_file('zoom', values):
        return jsonify({'success': True, 'message': f'Zoom value {zoom_value} saved to calibration file'})
    else:
        return jsonify({'success': False, 'error': 'Failed to save zoom value to calibration file'})


@app.route('/get_zoom', methods=['GET'])
def get_zoom():
    """Get current zoom value"""
    global zoom_value
    return jsonify({'zoom_value': zoom_value})


# ============================================
# REGISTRATION FUNCTIONS
# ============================================
@app.route('/reg_up', methods=['POST'])
def reg_up():
    """Registration up arrow"""
    global reg_offset_y, reg_stretch_y, registration_mode
    
    if registration_mode == "shift":
        new_value = reg_offset_y + 1
        # Send command to terminal - background listener will update reg_offset_y via callback
        success, response, parsed_value = execute_command_from_config('uv_offset_y_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV offset Y command'})
    else:
        new_value = reg_stretch_y + 1
        # Send command to terminal - background listener will update reg_stretch_y via callback
        success, response, parsed_value = execute_command_from_config('uv_magnify_y_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV magnify Y command'})


@app.route('/reg_down', methods=['POST'])
def reg_down():
    """Registration down arrow"""
    global reg_offset_y, reg_stretch_y, registration_mode
    
    if registration_mode == "shift":
        new_value = reg_offset_y - 1
        # Send command to terminal - background listener will update reg_offset_y via callback
        success, response, parsed_value = execute_command_from_config('uv_offset_y_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV offset Y command'})
    else:
        new_value = reg_stretch_y - 1
        # Send command to terminal - background listener will update reg_stretch_y via callback
        success, response, parsed_value = execute_command_from_config('uv_magnify_y_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV magnify Y command'})


@app.route('/reg_left', methods=['POST'])
def reg_left():
    """Registration left arrow"""
    global reg_offset_x, reg_stretch_x, registration_mode
    
    if registration_mode == "shift":
        new_value = reg_offset_x - 1
        # Send command to terminal - background listener will update reg_offset_x via callback
        success, response, parsed_value = execute_command_from_config('uv_offset_x_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV offset X command'})
    else:
        new_value = reg_stretch_x - 1
        # Send command to terminal - background listener will update reg_stretch_x via callback
        success, response, parsed_value = execute_command_from_config('uv_magnify_x_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV magnify X command'})


@app.route('/reg_right', methods=['POST'])
def reg_right():
    """Registration right arrow"""
    global reg_offset_x, reg_stretch_x, registration_mode
    
    if registration_mode == "shift":
        new_value = reg_offset_x + 1
        # Send command to terminal - background listener will update reg_offset_x via callback
        success, response, parsed_value = execute_command_from_config('uv_offset_x_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV offset X command'})
    else:
        new_value = reg_stretch_x + 1
        # Send command to terminal - background listener will update reg_stretch_x via callback
        success, response, parsed_value = execute_command_from_config('uv_magnify_x_command', new_value)
        if success:
            return jsonify({
                'success': True,
                'offset_x': reg_offset_x,
                'offset_y': reg_offset_y,
                'stretch_x': reg_stretch_x,
                'stretch_y': reg_stretch_y
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to send UV magnify X command'})


@app.route('/set_registration', methods=['POST'])
def set_registration():
    """Set registration to absolute values"""
    global reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y, registration_mode
    
    data = request.get_json()
    direction = data.get('direction')  # 'up', 'down', 'left', 'right'
    target_value = data.get('value')
    
    if direction is None or target_value is None:
        return jsonify({'success': False, 'error': 'Direction or value not provided'})
    
    if registration_mode == "shift":
        if direction in ['up', 'down']:
            # Send command to terminal with absolute target value
            success, response, parsed_value = execute_command_from_config('uv_offset_y_command', target_value)
        else:  # left, right
            success, response, parsed_value = execute_command_from_config('uv_offset_x_command', target_value)
    else:  # stretch_compress
        if direction in ['up', 'down']:
            success, response, parsed_value = execute_command_from_config('uv_magnify_y_command', target_value)
        else:  # left, right
            success, response, parsed_value = execute_command_from_config('uv_magnify_x_command', target_value)
    
    if success:
        return jsonify({
            'success': True,
            'offset_x': reg_offset_x,
            'offset_y': reg_offset_y,
            'stretch_x': reg_stretch_x,
            'stretch_y': reg_stretch_y
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to send registration command'})


@app.route('/reg_ok', methods=['POST'])
def reg_ok():
    """Registration OK button - saves all registration values (both shift and stretch/compress)"""
    global reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y
    
    # Save all four values (both shift and stretch/compress)
    values = {
        'offset_x': reg_offset_x,      # Shift X
        'offset_y': reg_offset_y,      # Shift Y
        'stretch_x': reg_stretch_x,    # Stretch/Compress X
        'stretch_y': reg_stretch_y     # Stretch/Compress Y
    }
    
    if save_calibration_to_file('registration', values):
        return jsonify({
            'success': True, 
            'message': f'Registration values saved: shift({reg_offset_x}, {reg_offset_y}), stretch({reg_stretch_x}, {reg_stretch_y})'
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to save registration values to calibration file'})


@app.route('/set_registration_mode', methods=['POST'])
def set_registration_mode():
    """Set registration mode (shift or stretch_compress)"""
    global registration_mode
    
    data = request.get_json()
    mode = data.get('mode', 'shift')
    
    if mode in ['shift', 'stretch_compress']:
        registration_mode = mode
        return jsonify({'success': True, 'mode': registration_mode})
    else:
        return jsonify({'success': False, 'error': 'Invalid mode'})


@app.route('/smart_registration', methods=['POST'])
def smart_registration():
    """Smart registration calibration"""
    # TODO: Implement smart registration
    return jsonify({'success': True, 'message': 'Smart Registration Calibration'})


@app.route('/get_registration_mode', methods=['GET'])
def get_registration_mode():
    """Get current registration mode and values"""
    global registration_mode, reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y
    return jsonify({
        'mode': registration_mode,
        'offset_x': reg_offset_x,
        'offset_y': reg_offset_y,
        'stretch_x': reg_stretch_x,
        'stretch_y': reg_stretch_y
    })


# ============================================
# FOCUS FUNCTIONS
# ============================================
@app.route('/focus_increase', methods=['POST'])
def focus_increase():
    """Increase focus value"""
    global focus_value, motor_max, auto_focus_enabled
    
    if auto_focus_enabled:
        return jsonify({'success': False, 'error': 'Auto focus is enabled. Disable auto focus to adjust manually.'})
    
    if motor_max is None or motor_max == 0:
        return jsonify({'success': False, 'error': 'Motor max not initialized'})
    
    if focus_value >= motor_max:
        return jsonify({'success': False, 'error': 'Focus at maximum'})
    
    new_value = focus_value + 1
    
    # Send command to terminal - background listener will update focus_value via callback
    success, response, parsed_value = execute_command_from_config('focus_command', new_value)
    
    if success:
        # Command sent - background listener will update focus_value when response arrives
        return jsonify({'success': True, 'focus_value': focus_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send focus command to terminal'})


@app.route('/focus_decrease', methods=['POST'])
def focus_decrease():
    """Decrease focus value"""
    global focus_value, auto_focus_enabled
    
    if auto_focus_enabled:
        return jsonify({'success': False, 'error': 'Auto focus is enabled. Disable auto focus to adjust manually.'})
    
    if focus_value <= 0:
        return jsonify({'success': False, 'error': 'Focus at minimum'})
    
    new_value = focus_value - 1
    
    # Send command to terminal - background listener will update focus_value via callback
    success, response, parsed_value = execute_command_from_config('focus_command', new_value)
    
    if success:
        # Command sent - background listener will update focus_value when response arrives
        return jsonify({'success': True, 'focus_value': focus_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send focus command to terminal'})


@app.route('/set_focus', methods=['POST'])
def set_focus():
    """Set focus to absolute value"""
    global focus_value, motor_max, auto_focus_enabled
    
    if auto_focus_enabled:
        return jsonify({'success': False, 'error': 'Auto focus is enabled. Disable auto focus to adjust manually.'})
    
    data = request.get_json()
    target_value = data.get('value')
    
    if target_value is None:
        return jsonify({'success': False, 'error': 'Value not provided'})
    
    if motor_max is None or motor_max == 0:
        return jsonify({'success': False, 'error': 'Motor max not initialized'})
    
    if target_value < 0 or target_value > motor_max:
        return jsonify({'success': False, 'error': f'Focus value out of range (0-{motor_max})'})
    
    # Send command to terminal with absolute target value
    success, response, parsed_value = execute_command_from_config('focus_command', target_value)
    
    if success:
        # Command sent - background listener will update focus_value when response arrives
        return jsonify({'success': True, 'focus_value': focus_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send focus command to terminal'})


@app.route('/focus_ok', methods=['POST'])
def focus_ok():
    """Save focus value and related info to calibration file"""
    global focus_value, uv_vis_focus_info
    
    values = {
        'focus_value': focus_value,
        'vis_position': uv_vis_focus_info.get('vis_focus_point', -1),
        'uv_position': uv_vis_focus_info.get('uv_motor_pos', -1),
        'zoom_status': uv_vis_focus_info.get('zoom_status', ''),
        'distance': uv_vis_focus_info.get('distance', -1)
    }
    
    if save_calibration_to_file('focus', values):
        return jsonify({'success': True, 'message': f'Focus calibration saved: focus={focus_value}, vis={values["vis_position"]}, uv={values["uv_position"]}, zoom={values["zoom_status"]}, distance={values["distance"]}'})
    else:
        return jsonify({'success': False, 'error': 'Failed to save focus value to calibration file'})


@app.route('/smart_focus', methods=['POST'])
def smart_focus():
    """Smart focus calibration"""
    # TODO: Implement smart focus CALL THE FUNCTION AUTOCALI
    return jsonify({'success': True, 'message': 'Smart Focus Calibration'})


@app.route('/get_focus', methods=['GET'])
def get_focus():
    """Get current focus value"""
    global focus_value, auto_focus_enabled, uv_vis_focus_info, uv_motor_position
    # If auto focus is enabled, use uv_vis_focus_info, otherwise use uv_motor_position
    if auto_focus_enabled:
        uv_position = uv_vis_focus_info.get('uv_motor_pos', -1)
    else:
        uv_position = uv_motor_position if uv_motor_position > 0 else -1
    
    return jsonify({
        'focus_value': focus_value,
        'auto_focus_enabled': auto_focus_enabled,
        'vis_position': uv_vis_focus_info.get('vis_focus_point', -1),
        'uv_position': uv_position,
        'zoom_status': uv_vis_focus_info.get('zoom_status', ''),
        'distance': uv_vis_focus_info.get('distance', -1)
    })


@app.route('/toggle_auto_focus', methods=['POST'])
def toggle_auto_focus():
    """Toggle auto focus mode"""
    global auto_focus_enabled, terminal_manager, config
    
    data = request.get_json()
    enable = data.get('enable')
    
    if enable is None:
        # Toggle current state
        auto_focus_enabled = not auto_focus_enabled
    else:
        auto_focus_enabled = bool(enable)
    
    # Send appropriate command based on state
    if auto_focus_enabled:
        # Enable auto focus
        success, response, parsed_value = execute_command_from_config('auto_focus_command')
        if success:
            add_console_message("Auto focus enabled")
            return jsonify({'success': True, 'auto_focus_enabled': True, 'message': 'Auto focus enabled'})
        else:
            auto_focus_enabled = False  # Revert on failure
            return jsonify({'success': False, 'error': 'Failed to enable auto focus'})
    else:
        # Disable auto focus (enable manual)
        success, response, parsed_value = execute_command_from_config('manual_focus_command')
        if success:
            add_console_message("Auto focus disabled (manual mode)")
            return jsonify({'success': True, 'auto_focus_enabled': False, 'message': 'Auto focus disabled'})
        else:
            auto_focus_enabled = True  # Revert on failure
            return jsonify({'success': False, 'error': 'Failed to disable auto focus'})


# ============================================
# GAIN FUNCTIONS
# ============================================
@app.route('/gain_increase', methods=['POST'])
def gain_increase():
    """Increase gain value"""
    global gain_value
    
    if gain_value >= 255:
        return jsonify({'success': False, 'error': 'Gain at maximum'})
    
    new_value = gain_value + 1
    
    # Send command to terminal - background listener will update gain_value via callback
    success, response, parsed_value = execute_command_from_config('gain_command', new_value)
    
    if success:
        # Command sent - background listener will update gain_value when response arrives
        return jsonify({'success': True, 'gain_value': gain_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send gain command to terminal'})


@app.route('/gain_decrease', methods=['POST'])
def gain_decrease():
    """Decrease gain value"""
    global gain_value
    
    if gain_value <= 0:
        return jsonify({'success': False, 'error': 'Gain at minimum'})
    
    new_value = gain_value - 1
    
    # Send command to terminal - background listener will update gain_value via callback
    success, response, parsed_value = execute_command_from_config('gain_command', new_value)
    
    if success:
        # Command sent - background listener will update gain_value when response arrives
        return jsonify({'success': True, 'gain_value': gain_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send gain command to terminal'})


@app.route('/set_gain', methods=['POST'])
def set_gain():
    """Set gain to absolute value"""
    global gain_value
    
    data = request.get_json()
    target_value = data.get('value')
    
    if target_value is None:
        return jsonify({'success': False, 'error': 'Value not provided'})
    
    if target_value < 0 or target_value > 255:
        return jsonify({'success': False, 'error': 'Gain value out of range (0-255)'})
    
    # Send command to terminal with absolute target value
    success, response, parsed_value = execute_command_from_config('gain_command', target_value)
    
    if success:
        # Command sent - background listener will update gain_value when response arrives
        return jsonify({'success': True, 'gain_value': gain_value})
    else:
        return jsonify({'success': False, 'error': 'Failed to send gain command to terminal'})


@app.route('/gain_ok', methods=['POST'])
def gain_ok():
    """Save gain min/medium/max values to calibration file"""
    global gain_min_value, gain_medium_value, gain_max_value
    
    values = {
        'gain_min_value': gain_min_value,
        'gain_medium_value': gain_medium_value,
        'gain_max_value': gain_max_value
    }
    
    if save_calibration_to_file('gain', values):
        return jsonify({'success': True, 'message': f'Gain calibration saved: min={gain_min_value}, medium={gain_medium_value}, max={gain_max_value}'})
    else:
        return jsonify({'success': False, 'error': 'Failed to save gain values to calibration file'})


@app.route('/toggle_gain_mode', methods=['POST'])
def toggle_gain_mode():
    """Toggle between min and max gain mode"""
    global gain_mode
    
    if gain_mode == "min":
        gain_mode = "max"
    else:
        gain_mode = "min"
    
    return jsonify({'success': True, 'mode': gain_mode})


@app.route('/get_gain_mode', methods=['GET'])
def get_gain_mode():
    """Get current gain mode"""
    global gain_mode
    return jsonify({'mode': gain_mode})


@app.route('/smart_gain', methods=['POST'])
def smart_gain():
    """Smart gain calibration"""
    # TODO: Implement smart gain
    return jsonify({'success': True, 'message': 'Smart Gain Calibration'})


@app.route('/get_gain', methods=['GET'])
def get_gain():
    """Get current gain value and min/medium/max values"""
    global gain_value, gain_min_value, gain_medium_value, gain_max_value
    return jsonify({
        'gain_value': gain_value,
        'gain_min_value': gain_min_value,
        'gain_medium_value': gain_medium_value,
        'gain_max_value': gain_max_value
    })


@app.route('/set_gain_min', methods=['POST'])
def set_gain_min():
    """Send set min gain command to terminal"""
    success, response, parsed_value = execute_command_from_config('set_min_gain_command')
    
    if success:
        return jsonify({'success': True, 'message': 'Set min gain command sent'})
    else:
        return jsonify({'success': False, 'error': 'Failed to send set min gain command'})


@app.route('/set_gain_medium', methods=['POST'])
def set_gain_medium():
    """Send set medium gain command to terminal"""
    success, response, parsed_value = execute_command_from_config('set_medium_gain_command')
    
    if success:
        return jsonify({'success': True, 'message': 'Set medium gain command sent'})
    else:
        return jsonify({'success': False, 'error': 'Failed to send set medium gain command'})


@app.route('/set_gain_max', methods=['POST'])
def set_gain_max():
    """Send set max gain command to terminal"""
    success, response, parsed_value = execute_command_from_config('set_max_gain_command')
    
    if success:
        return jsonify({'success': True, 'message': 'Set max gain command sent'})
    else:
        return jsonify({'success': False, 'error': 'Failed to send set max gain command'})


@app.route('/clear_gain_values', methods=['POST'])
def clear_gain_values():
    """Send clear gain values command to terminal"""
    success, response, parsed_value = execute_command_from_config('clear_gain_value_command')
    
    if success:
        return jsonify({'success': True, 'message': 'Clear gain values command sent'})
    else:
        return jsonify({'success': False, 'error': 'Failed to send clear gain values command'})


@app.route('/start_gain_counting', methods=['POST'])
def start_gain_counting():
    """Start gain counting process"""
    # TODO: Implement gain counting logic
    logging.info("Gain counting started")
    return jsonify({'success': True, 'message': 'Gain counting started'})


@app.route('/get_cps', methods=['GET'])
def get_cps():
    """Get current CPS value"""
    global cps_value
    return jsonify({'cps_value': cps_value})


@app.route('/set_time_interval', methods=['POST'])
def set_time_interval():
    """Set time interval value"""
    global time_interval_value
    
    data = request.get_json()
    time_interval_value = data.get('time_interval', '')
    return jsonify({'success': True, 'time_interval': time_interval_value})


@app.route('/get_motor_max', methods=['GET'])
def get_motor_max():
    """Get current motor max value"""
    global motor_max
    return jsonify({'motor_max': motor_max if motor_max is not None else 0})


@app.route('/calculate_motor_max', methods=['POST'])
def calculate_motor_max():
    """Calculate motor max by executing motor_max_script and getting UV motor position"""
    global motor_max
    
    motor_max_result = find_motor_max()
    if motor_max_result is not None and motor_max_result > 0:
        return jsonify({
            'success': True, 
            'message': f'Motor max calculated: {motor_max}',
            'motor_max': motor_max
        })
    else:
        return jsonify({
            'success': False, 
            'error': f'Failed to calculate motor max. Current value: {motor_max if motor_max is not None else 0}'
        })


@app.route('/get_camera_serial', methods=['GET'])
def get_camera_serial():
    """Get camera serial number from calibration file"""
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    camera_serial_value = ''
    
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            camera_serial_value = data.get('camera_serial', '')
        except Exception as e:
            logging.warning(f"Error loading camera_serial from file: {e}")
    
    return jsonify({'camera_serial': camera_serial_value})


@app.route('/set_camera_serial', methods=['POST'])
def set_camera_serial():
    """Save camera serial number to calibration file"""
    data = request.get_json()
    camera_serial_value = data.get('camera_serial', '')
    
    if update_calibration_file_header(camera_serial=camera_serial_value):
        return jsonify({'success': True, 'message': f'Camera serial number saved: {camera_serial_value}'})
    else:
        return jsonify({'success': False, 'error': 'Failed to save camera serial to calibration file'})


@app.route('/get_technician_name', methods=['GET'])
def get_technician_name():
    """Get technician name from calibration file"""
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    technician_name_value = ''
    
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            technician_name_value = data.get('technician_name', '')
        except Exception as e:
            logging.warning(f"Error loading technician_name from file: {e}")
    
    return jsonify({'technician_name': technician_name_value})


@app.route('/set_technician_name', methods=['POST'])
def set_technician_name():
    """Save technician name to calibration file"""
    data = request.get_json()
    technician_name_value = data.get('technician_name', '')
    
    if update_calibration_file_header(technician_name=technician_name_value):
        return jsonify({'success': True, 'message': f'Technician name saved: {technician_name_value}'})
    else:
        return jsonify({'success': False, 'error': 'Failed to save technician name to calibration file'})


@app.route('/get_distance_to_target', methods=['GET'])
def get_distance_to_target():
    """Get current distance to target value"""
    global distance_to_target
    return jsonify({'distance_to_target': distance_to_target})


@app.route('/set_distance_to_target', methods=['POST'])
def set_distance_to_target():
    """Set distance to target value"""
    global distance_to_target
    data = request.get_json()
    distance_value = data.get('distance_to_target')
    
    if distance_value is None:
        return jsonify({'success': False, 'error': 'Distance value not provided'})
    
    try:
        distance_value = int(distance_value)
        if distance_value < 1 or distance_value > 100:
            return jsonify({'success': False, 'error': 'Distance value out of range (1-100)'})
        
        distance_to_target = distance_value
        # Update calibration file header with new distance_to_target
        update_calibration_file_header()
        return jsonify({'success': True, 'message': f'Distance to target set to: {distance_to_target}', 'distance_to_target': distance_to_target})
    except (ValueError, TypeError):
        return jsonify({'success': False, 'error': 'Invalid distance value'})


@app.route('/get_camera_mode', methods=['GET'])
def get_camera_mode():
    """Get current camera mode"""
    global camera_mode
    return jsonify({'camera_mode': camera_mode})


@app.route('/set_camera_mode', methods=['POST'])
def set_camera_mode():
    """Set camera mode: 1=VIS ONLY, 2=UV ONLY, 3=UV & VIS"""
    global camera_mode
    data = request.get_json()
    mode = data.get('mode')
    
    if mode == 1:
        command_name = 'set_camera_to_vis_command'
    elif mode == 2:
        command_name = 'set_camera_to_uv_command'
    elif mode == 3:
        command_name = 'set_camera_to_combined_command'
    else:
        return jsonify({'success': False, 'error': 'Invalid camera mode'})
    
    success, response, parsed_value = execute_command_from_config(command_name)
    
    if success:
        # Update camera_mode immediately (callback will also update it when response arrives)
        # This ensures the UI shows the correct state even if callback is delayed
        camera_mode = mode
        logging.info(f"Camera mode set to {mode} (will be confirmed by callback)")
        return jsonify({'success': True, 'message': f'Camera mode set to {mode}', 'camera_mode': mode})
    else:
        return jsonify({'success': False, 'error': 'Failed to set camera mode'})


# ============================================
# CONSOLE FUNCTIONS
# ============================================
@app.route('/get_console', methods=['GET'])
def get_console():
    """Get console messages"""
    global console_messages, console_lock
    
    with console_lock:
        # Return all messages as a single string (newline separated)
        messages_text = '\n'.join(console_messages)
    
    return jsonify({'messages': messages_text})


@app.route('/send_terminal_command', methods=['POST'])
def send_terminal_command():
    """Send a command to a specific terminal"""
    global terminal_manager
    
    if terminal_manager is None:
        return jsonify({'success': False, 'error': 'Terminal manager not initialized'})
    
    data = request.get_json()
    terminal_name = data.get('terminal')
    command = data.get('command', '')
    
    if not terminal_name:
        return jsonify({'success': False, 'error': 'Terminal name not provided'})
    
    if not command:
        return jsonify({'success': False, 'error': 'Command not provided'})
    
    # Send command to terminal (background listener will catch response)
    terminal = terminal_manager.get_terminal(terminal_name)
    if terminal is None:
        return jsonify({'success': False, 'error': f'Terminal "{terminal_name}" not found'})
    elif not terminal.serial_conn or not terminal.serial_conn.is_open:
        return jsonify({'success': False, 'error': f'Terminal "{terminal_name}" is not connected'})
    else:
        # Command sent successfully (background listener will handle response)
        terminal_manager.send_command(terminal_name, command)
        return jsonify({'success': True, 'message': f'Command sent to {terminal_name}'})


# ============================================
# INTERVAL UPDATE FUNCTION
# ============================================
def update_values_from_terminal():
    """
    Background thread function that periodically updates zoom, gain, and focus values
    from the terminal.
    """
    global zoom_value, gain_value, focus_value, update_thread_running, config, terminal_manager
    
    while update_thread_running:
        try:
            # Check if config and terminal_manager are initialized
            if config is None or terminal_manager is None:
                logging.debug("Config or terminal manager not initialized, skipping update")
                time.sleep(2)
                continue
            
            # Get update interval from config (default 2000ms)
            interval_ms = config.get('settings', {}).get('update_interval_ms', 2000)
            interval_seconds = interval_ms / 1000.0
            
            # Check if at least one terminal is connected before trying to query
            has_connected_terminal = False
            if terminal_manager:
                for terminal in terminal_manager.terminals.values():
                    if terminal.serial_conn and terminal.serial_conn.is_open:
                        has_connected_terminal = True
                        break
            
            if not has_connected_terminal:
                logging.debug("No terminals connected, skipping value update")
                time.sleep(interval_seconds)
                continue
            
            # Update zoom value
            zoom_val = get_value_from_config('get_zoom_value')
            if zoom_val is not None:
                zoom_value = zoom_val
                logging.debug(f"Updated zoom value from terminal: {zoom_val}")
            
            # Update gain value
            logging.info("=== GAIN UPDATE DEBUG START ===")
            logging.info(f"Current gain_value before update: {gain_value}")
            gain_val = get_value_from_config('get_gain_value')
            logging.info(f"get_value_from_config('get_gain_value') returned: {gain_val}")
            if gain_val is not None:
                old_gain = gain_value
                gain_value = gain_val
                logging.info(f"Updated gain value from {old_gain} to {gain_value}")
            else:
                logging.warning("=== GAIN UPDATE FAILED - get_value_from_config returned None ===")
            logging.info("=== GAIN UPDATE DEBUG END ===")
            
            # Update focus value
            focus_val = get_value_from_config('get_focus_value')
            if focus_val is not None:
                focus_value = focus_val
                logging.debug(f"Updated focus value from terminal: {focus_val}")
            
            # Update UV motor position (only if auto focus is disabled)
            global auto_focus_enabled, uv_motor_position
            if not auto_focus_enabled:
                get_value_from_config('get_uv_motor_position')
                logging.debug(f"Requested UV motor position update (current: {uv_motor_position})")
            
            # Sleep until next update
            time.sleep(interval_seconds)
            
        except Exception as e:
            logging.error(f"Error in update_values_from_terminal: {e}", exc_info=True)
            time.sleep(1)  # Short sleep on error to prevent tight loop


def start_update_thread():
    """Start the background thread for updating values from terminal."""
    global update_thread, update_thread_running
    
    if update_thread_running:
        return  # Already running
    
    update_thread_running = True
    update_thread = threading.Thread(target=update_values_from_terminal, daemon=True)
    update_thread.start()
    logging.info("Started interval update thread")


def stop_update_thread():
    """Stop the background thread for updating values."""
    global update_thread_running
    
    update_thread_running = False
    logging.info("Stopped interval update thread")


if __name__ == '__main__':
    # Setup logging - only WARNING and above for Flask/Werkzeug to reduce noise
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Disable Flask/Werkzeug request logging (too verbose)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Add initial console message
    add_console_message("Application started")
    
    # Initialize components
    init_components()
    
    # Start interval update thread
    start_update_thread()
    
    # Run Flask app (debug=False to reduce logging)
    app.run(debug=False, host='0.0.0.0', port=5000)

