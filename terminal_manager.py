#!/usr/bin/env python3
"""
Terminal Manager
Manages connections to three terminals: Linux, Debug, and FPGA
Based on SerialSession from bad_pixel_automation project
"""

import serial
import logging
import time
import threading
import re
from typing import Dict, Optional, Any, List, Callable
from collections import deque
from datetime import datetime


class TerminalSession:
    """Manages a single serial terminal connection."""
    
    def __init__(self, name: str, config: Dict[str, Any], log_callback: Optional[Callable[[str], None]] = None):
        self.name = name
        self.config = config
        self.port = config['port']
        self.baudrate = config['baudrate']
        self.eol = config.get('eol', '\n')
        self.encoding = config.get('encoding', 'utf-8')
        self.timeout_ms = config.get('timeout_ms', 3000)
        self.ignore_responses = config.get('ignore_responses', [])
        self.serial_conn: Optional[serial.Serial] = None
        self.log_callback = log_callback  # Callback to log messages to console
        
        # Convert timeout to seconds for pySerial
        self.timeout_seconds = self.timeout_ms / 1000.0
        
    def connect(self) -> bool:
        """
        Open serial connection.
        Returns True if successful, False otherwise.
        """
        try:
            # Close existing connection if any
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            
            # Configure serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout_seconds,
                write_timeout=self.timeout_seconds
            )
            
            # Clear any existing data in buffers
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            logging.info(f"Terminal '{self.name}' connected to {self.port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to connect terminal '{self.name}' to {self.port}: {e}")
            self.serial_conn = None
            return False
    
    def disconnect(self):
        """Close the serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logging.info(f"Terminal '{self.name}' disconnected")
    
    def send_command(self, command: str):
        """
        Send command. Background listener will catch the response.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            # Attempt lazy reconnect
            if not self.connect():
                return
        
        try:
            # Send command
            if not command.endswith(self.eol):
                command += self.eol
            
            # Log command being sent
            command_to_send = command.rstrip(self.eol)
            if self.log_callback:
                self.log_callback(f"SEND {self.name}: {command_to_send}")
            
            self.serial_conn.write(command.encode(self.encoding))
            self.serial_conn.flush()
                
        except Exception as e:
            logging.error(f"Error executing command '{command}' on terminal '{self.name}': {e}")
    
    def read_line(self) -> Optional[str]:
        """Read a single line from the terminal (non-blocking)."""
        if not self.serial_conn or not self.serial_conn.is_open:
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                line = self.serial_conn.readline()
                if line:
                    decoded_line = line.decode(self.encoding, errors='replace').rstrip('\r\n')
                    if decoded_line:
                        # Check if should ignore
                        for ignore_substring in self.ignore_responses:
                            if ignore_substring in decoded_line:
                                return None  # Ignore this line
                        return decoded_line
        except Exception as e:
            logging.debug(f"Error reading line from {self.name}: {e}")
        
        return None


class TerminalManager:
    """Manages all three terminal connections: Linux, Debug, and FPGA."""
    
    def __init__(self, config: Dict[str, Any], log_callback: Optional[Callable[[str], None]] = None):
        self.config = config
        self.terminals: Dict[str, TerminalSession] = {}
        self.log_callback = log_callback
        self._init_terminals()
        
        # Background listener thread
        self.listener_thread = None
        self.listener_running = False
        self.response_callbacks: Dict[str, List[Callable[[str, str], None]]] = {}  # pattern -> callbacks
        self._lock = threading.Lock()
    
    def _init_terminals(self):
        """Initialize all terminal sessions."""
        for terminal_config in self.config.get('terminals', []):
            name = terminal_config['name']
            terminal = TerminalSession(name, terminal_config, log_callback=self.log_callback)
            self.terminals[name] = terminal
    
    def connect_all(self) -> bool:
        """Connect to all terminals. Returns True if at least one connected successfully."""
        success_count = 0
        for terminal in self.terminals.values():
            if terminal.connect():
                success_count += 1
        
        if success_count == 0:
            logging.error("Failed to connect to any terminals")
            return False
        
        if success_count < len(self.terminals):
            logging.warning(f"Connected to {success_count}/{len(self.terminals)} terminals")
        
        return True
    
    def disconnect_all(self):
        """Disconnect from all terminals."""
        for terminal in self.terminals.values():
            terminal.disconnect()
    
    def get_terminal(self, name: str) -> Optional[TerminalSession]:
        """Get a terminal session by name."""
        return self.terminals.get(name)
    
    def send_command(self, terminal_name: str, command: str):
        """Send a command to a specific terminal. Background listener will catch the response."""
        terminal = self.get_terminal(terminal_name)
        if terminal:
            terminal.send_command(command)
        else:
            logging.error(f"Terminal '{terminal_name}' not found")
    
    def register_response_callback(self, response_pattern: str, callback: Callable[[str, str], None]):
        """
        Register a callback for a specific response pattern.
        Callback will be called with (terminal_name, response_message) when pattern matches.
        
        Args:
            response_pattern: Pattern like "MZR<val>", "GAR<val>", "MIOR-<val>"
            callback: Function(terminal_name: str, response: str) -> None
        """
        with self._lock:
            if response_pattern not in self.response_callbacks:
                self.response_callbacks[response_pattern] = []
            self.response_callbacks[response_pattern].append(callback)
            logging.debug(f"Registered callback for pattern '{response_pattern}'")
    
    def _parse_and_notify(self, terminal_name: str, message: str):
        """Parse message and notify registered callbacks if pattern matches."""
        with self._lock:
            for pattern, callbacks in self.response_callbacks.items():
                # Convert pattern to regex (e.g., "MZR<val>" -> "MZR(-?\d+)")
                regex_pattern = pattern.replace('<val>', r'(-?\d+)')
                match = re.search(regex_pattern, message)
                if match:
                    logging.info(f"Pattern '{pattern}' (regex: '{regex_pattern}') MATCHED message '{message}' from {terminal_name}")
                    for callback in callbacks:
                        try:
                            callback(terminal_name, message)
                        except Exception as e:
                            logging.error(f"Error in callback for pattern '{pattern}': {e}", exc_info=True)
                else:
                    # Log if this looks like a motor response but didn't match
                    if 'MIOR' in message:
                        logging.warning(f"Message '{message}' contains 'MIOR' but pattern '{pattern}' (regex: '{regex_pattern}') did NOT match!")
    
    def _background_listener(self):
        """Background thread that continuously reads from all terminals."""
        logging.info("Background terminal listener started")
        while self.listener_running:
            try:
                for terminal_name, terminal in self.terminals.items():
                    if not terminal.serial_conn or not terminal.serial_conn.is_open:
                        continue
                    
                    line = terminal.read_line()
                    if line:
                        # Log received message
                        if self.log_callback:
                            self.log_callback(f"RECV {terminal_name}: {line}")
                        
                        # Parse and notify callbacks
                        self._parse_and_notify(terminal_name, line)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)  # 10ms
                
            except Exception as e:
                logging.error(f"Error in background listener: {e}")
                time.sleep(0.1)
        
        logging.info("Background terminal listener stopped")
    
    def start_listener(self):
        """Start the background listener thread."""
        if self.listener_thread is None or not self.listener_thread.is_alive():
            self.listener_running = True
            self.listener_thread = threading.Thread(target=self._background_listener, daemon=True)
            self.listener_thread.start()
            logging.info("Started background terminal listener")
    
    def stop_listener(self):
        """Stop the background listener thread."""
        self.listener_running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
        logging.info("Stopped background terminal listener")

