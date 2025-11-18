#!/usr/bin/env python3
"""
Terminal Manager
Manages connections to three terminals: Linux, Debug, and FPGA
Based on SerialSession from bad_pixel_automation project
"""

import serial
import logging
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
    
    def send_command(self, command: str) -> Optional[str]:
        """
        Send command and read response with filtering.
        Returns filtered response or None if no meaningful response.
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            # Attempt lazy reconnect
            if not self.connect():
                return None
        
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
            
            # Read response with quiet window
            import time
            response_lines = []
            start_time = time.time()
            quiet_start = None
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check overall timeout
                if elapsed >= self.timeout_seconds:
                    break
                
                # Try to read a line
                try:
                    line = self.serial_conn.readline()
                    if line:
                        # Got data, reset quiet window
                        quiet_start = None
                        decoded_line = line.decode(self.encoding, errors='replace').rstrip('\r\n')
                        if decoded_line:  # Skip empty lines
                            response_lines.append(decoded_line)
                    else:
                        # No data received
                        if quiet_start is None:
                            quiet_start = current_time
                        elif current_time - quiet_start >= 0.1:  # 100ms quiet window
                            break
                        time.sleep(0.01)  # Small sleep to prevent busy waiting
                        
                except Exception:
                    break
            
            # Filter response lines
            filtered_lines = []
            for line in response_lines:
                should_ignore = False
                
                # Skip if it's the command echo (first line that matches the sent command)
                if line.strip() == command.strip():
                    should_ignore = True
                
                # Skip if it matches ignore_responses
                if not should_ignore:
                    for ignore_substring in self.ignore_responses:
                        if ignore_substring in line:
                            should_ignore = True
                            break
                
                if not should_ignore:
                    filtered_lines.append(line)
                    # Log response line
                    if self.log_callback:
                        self.log_callback(f"RECV {self.name}: {line}")
            
            # Return filtered response or None
            if filtered_lines:
                return '\n'.join(filtered_lines)
            else:
                return None
                
        except Exception as e:
            logging.error(f"Error executing command '{command}' on terminal '{self.name}': {e}")
            return None


class TerminalManager:
    """Manages all three terminal connections: Linux, Debug, and FPGA."""
    
    def __init__(self, config: Dict[str, Any], log_callback: Optional[Callable[[str], None]] = None):
        self.config = config
        self.terminals: Dict[str, TerminalSession] = {}
        self.log_callback = log_callback
        self._init_terminals()
    
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
    
    def send_command(self, terminal_name: str, command: str) -> Optional[str]:
        """Send a command to a specific terminal and return the response."""
        terminal = self.get_terminal(terminal_name)
        if terminal:
            return terminal.send_command(command)
        else:
            logging.error(f"Terminal '{terminal_name}' not found")
            return None

