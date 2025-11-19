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
registration_mode = "shift"  # "shift" or "stretch_compress"
reg_offset_x = 0
reg_offset_y = 0
reg_stretch_x = 0
reg_stretch_y = 0
gain_mode = "min"  # "min" or "max"
cps_value = 0
time_interval_value = ""
motor_max = None  # Motor maximum position value

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


def init_components():
    """Initialize terminal manager and stream processor."""
    global stream_processor, terminal_manager, config
    
    if config is None:
        config = load_config()
    
    # Pass log callback to terminal manager
    terminal_manager = TerminalManager(config, log_callback=add_console_message)
    
    stream_processor = StreamProcessor(
        rtsp_url=config.get('stream', {}).get('rtsp_url', 'rtsp://192.168.0.100:9079/vis')
    )


def register_terminal_callbacks():
    """Register callbacks for automatic parsing of terminal responses based on config."""
    global terminal_manager, config, zoom_value, gain_value, focus_value, motor_max
    global reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y
    
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
    }
    
    # Special case for motor_max (from operations)
    motor_max_pattern = None
    if 'operations' in config and 'motor_max_script' in config['operations']:
        for step in config['operations']['motor_max_script']:
            response = step.get('response')
            # Skip None values, but keep looking for actual response pattern
            if response is not None and response != 'None' and response:
                motor_max_pattern = response
                break
    
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
    
    # Add motor_max pattern
    if motor_max_pattern:
        logging.info(f"Found motor_max_pattern in config: '{motor_max_pattern}'")
        if motor_max_pattern not in response_to_globals:
            response_to_globals[motor_max_pattern] = []
        # Motor max should be positive
        def validate_motor_max(val):
            return val > 0
        response_to_globals[motor_max_pattern].append(('motor_max', validate_motor_max))
        logging.info(f"Added motor_max callback for pattern '{motor_max_pattern}'")
    else:
        logging.warning("motor_max_pattern not found in config operations!")
    
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
                            elif global_var_name == 'motor_max':
                                old_value = globals()['motor_max']
                                globals()['motor_max'] = parsed
                                logging.info(f"Motor max updated: {old_value} -> {parsed} from response: {response}")
                                add_console_message(f"Motor max updated: {parsed}")
                                # Update calibration file asynchronously (don't block callback)
                                def update_file_async():
                                    time.sleep(0.1)  # Small delay to avoid blocking
                                    initialize_calibration_file()
                                threading.Thread(target=update_file_async, daemon=True).start()
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
        
        # Send command without waiting for response (background listener will catch it)
        terminal.send_command(command, wait_for_response=False)
        
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
        
        # Send command without waiting for response (background listener will catch it)
        terminal.send_command(command, wait_for_response=False)
        
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
    Background listener will catch the response and update motor_max via callback.
    This function waits for the callback to update the value.
    
    Returns:
        Motor max value as integer, or None if failed
    """
    global motor_max, terminal_manager, config
    
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
    
    # Store initial motor_max value
    initial_motor_max = motor_max
    logging.info(f"Initial motor_max value: {initial_motor_max}")
    
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
        terminal.send_command(command, wait_for_response=False)
        
        # Only wait for response if this step expects one
        if response_pattern and response_pattern != "None":
            # Wait for callback to update motor_max (max 20 seconds, script has 10s delay)
            max_wait_time = 20.0
            start_time = time.time()
            logging.info(f"Waiting for response matching pattern '{response_pattern}' (max {max_wait_time}s)...")
            add_console_message(f"Waiting for response: {response_pattern}")
            while time.time() - start_time < max_wait_time:
                # Check if motor_max was updated (either from None to a value, or changed value)
                if motor_max is not None and motor_max > 0:
                    if initial_motor_max is None or initial_motor_max == 0 or motor_max != initial_motor_max:
                        logging.info(f"Motor max updated via callback: {motor_max}")
                        add_console_message(f"Motor max found: {motor_max}")
                        return motor_max
                elapsed = time.time() - start_time
                if int(elapsed) % 2 == 0 and elapsed > 0.5:  # Log every 2 seconds
                    logging.debug(f"Still waiting for motor_max... (elapsed: {elapsed:.1f}s, current: {motor_max})")
                time.sleep(0.2)  # Check every 200ms (less frequent to reduce CPU)
            
            logging.warning(f"Timeout waiting for motor max response via callback. Current value: {motor_max}")
            add_console_message(f"WARNING: Timeout waiting for motor max. Current value: {motor_max}")
    
    # Return current motor_max (may have been updated by callback)
    if motor_max is not None and motor_max > 0:
        return motor_max
    
    logging.warning("Motor max script completed but no valid value received")
    return motor_max


def terminal_initialization():
    """
    Initialize terminal values: get zoom, gain, focus, UV registration from terminal and find motor max.
    Updates global variables and UI.
    """
    global zoom_value, gain_value, focus_value, motor_max, reg_offset_x, reg_offset_y, reg_stretch_x, reg_stretch_y
    
    logging.info("Starting terminal initialization...")
    
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
    
    # Find motor max position (waits for response)
    motor_max_result = find_motor_max()
    if motor_max_result is not None and motor_max_result > 0:
        logging.info(f"Initialized motor max: {motor_max}")
        # Initialize calibration file with motor_max (only if we got a valid value)
        # Note: If motor_max updates later via callback, the file will be updated asynchronously
        initialize_calibration_file()
    else:
        logging.warning(f"Motor max initialization failed or returned invalid value: {motor_max}")
        # Don't create file with 0 - wait for callback to update it later
    
    logging.info("Terminal initialization completed")


def save_calibration_to_file(calibration_type: str, values: dict):
    """
    Save calibration data to YAML file.
    If entry of same type exists, it will be overwritten.
    
    Args:
        calibration_type: Type of calibration ('zoom', 'focus', 'gain', 'registration')
        values: Dictionary with calibration values including zoom_value
    """
    global motor_max, zoom_value
    
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
    
    if 'calibrations' not in data:
        data['calibrations'] = []
    
    # Create calibration entry (without timestamp)
    entry = {
        'type': calibration_type,
        'zoom': zoom_value,
        **values  # Add all other values
    }
    
    # Find existing entry of same type and replace it, or add new one
    found = False
    for i, existing_entry in enumerate(data['calibrations']):
        if existing_entry.get('type') == calibration_type:
            data['calibrations'][i] = entry  # Overwrite existing entry
            found = True
            break
    
    if not found:
        # Add new entry if not found
        data['calibrations'].append(entry)
    
    # Save to file
    try:
        # Ensure directory exists
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(calibration_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        logging.info(f"Calibration saved: {calibration_type} at zoom {zoom_value}")
        return True
    except Exception as e:
        logging.error(f"Error saving calibration file: {e}", exc_info=True)
        return False


def initialize_calibration_file():
    """Initialize calibration file with motor_max value."""
    global motor_max
    
    calibration_file = Path(__file__).parent / 'calibration_files' / 'calibrated_file.yaml'
    
    # Use 0 as default if motor_max is not available yet
    current_motor_max = motor_max if (motor_max is not None and motor_max > 0) else 0
    
    # Load existing data or create new structure
    if calibration_file.exists():
        try:
            with open(calibration_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            # Update motor_max if it changed or if it's 0 and we have a new value
            if 'motor_max' not in data or (data['motor_max'] != current_motor_max and current_motor_max > 0):
                data['motor_max'] = current_motor_max
                # Ensure calibrations list exists
                if 'calibrations' not in data:
                    data['calibrations'] = []
                # Save updated motor_max
                with open(calibration_file, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                logging.info(f"Calibration file motor_max updated to: {current_motor_max}")
            return
        except Exception as e:
            logging.warning(f"Error loading calibration file: {e}, creating new file")
    
    # Initialize new file with motor_max (even if 0, it will be updated later)
    data = {
        'motor_max': current_motor_max,
        'calibrations': []
    }
    
    try:
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(calibration_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logging.info(f"Calibration file initialized with motor_max: {current_motor_max}")
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
        terminal.send_command(command, wait_for_response=False)
        
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
    global focus_value, motor_max
    
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
    global focus_value
    
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
    global focus_value, motor_max
    
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
    """Save focus value to calibration file"""
    global focus_value
    
    values = {
        'focus_value': focus_value
    }
    
    if save_calibration_to_file('focus', values):
        return jsonify({'success': True, 'message': f'Focus value {focus_value} saved to calibration file'})
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
    global focus_value
    return jsonify({'focus_value': focus_value})


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
    """Save gain value to calibration file"""
    global gain_value, gain_mode
    
    values = {
        'gain_value': gain_value,
        'gain_mode': gain_mode
    }
    
    if save_calibration_to_file('gain', values):
        return jsonify({'success': True, 'message': f'Gain value {gain_value} saved to calibration file'})
    else:
        return jsonify({'success': False, 'error': 'Failed to save gain value to calibration file'})


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
    """Get current gain value"""
    global gain_value
    return jsonify({'gain_value': gain_value})


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

