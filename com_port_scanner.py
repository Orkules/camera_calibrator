#!/usr/bin/env python3
"""
COM Port Scanner
Detects and displays information about all serial/COM ports connected to the computer.
"""

import serial
import serial.tools.list_ports
import sys
from datetime import datetime


def scan_com_ports():
    """
    Scan for available COM ports and return detailed information.
    
    Returns:
        list: List of port information dictionaries
    """
    ports = []
    available_ports = serial.tools.list_ports.comports()
    
    for port in available_ports:
        port_info = {
            'device': port.device,
            'description': port.description,
            'hwid': port.hwid,
            'vid': port.vid,
            'pid': port.pid,
            'serial_number': port.serial_number,
            'manufacturer': port.manufacturer,
            'product': port.product,
            'location': port.location
        }
        ports.append(port_info)
    
    return ports


def test_port_connectivity(port_device):
    """
    Test if a COM port can be opened and closed successfully.
    
    Args:
        port_device (str): The port device name (e.g., 'COM3')
    
    Returns:
        dict: Connection test results
    """
    test_result = {
        'port': port_device,
        'accessible': False,
        'error': None
    }
    
    try:
        # Try to open the port with common settings
        with serial.Serial(port_device, 9600, timeout=1) as ser:
            test_result['accessible'] = True
            test_result['is_open'] = ser.is_open
    except serial.SerialException as e:
        test_result['error'] = str(e)
    except Exception as e:
        test_result['error'] = f"Unexpected error: {str(e)}"
    
    return test_result


def print_port_info(port_info, test_result=None):
    """
    Print formatted information about a COM port.
    
    Args:
        port_info (dict): Port information dictionary
        test_result (dict, optional): Connection test results
    """
    print(f"\n{'='*60}")
    print(f"PORT: {port_info['device']}")
    print(f"{'='*60}")
    
    print(f"Description:    {port_info['description'] or 'N/A'}")
    print(f"Hardware ID:    {port_info['hwid'] or 'N/A'}")
    print(f"Manufacturer:   {port_info['manufacturer'] or 'N/A'}")
    print(f"Product:        {port_info['product'] or 'N/A'}")
    print(f"Serial Number:  {port_info['serial_number'] or 'N/A'}")
    print(f"Location:       {port_info['location'] or 'N/A'}")
    
    # Display VID/PID if available
    if port_info['vid'] and port_info['pid']:
        print(f"VID:PID:        {port_info['vid']:04X}:{port_info['pid']:04X}")
    
    # Display connectivity test results
    if test_result:
        status = "✓ ACCESSIBLE" if test_result['accessible'] else "✗ NOT ACCESSIBLE"
        print(f"Status:         {status}")
        if test_result['error']:
            print(f"Error:          {test_result['error']}")


def main():
    """
    Main function to scan and display COM port information.
    """
    print("COM Port Scanner")
    print("=" * 60)
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Scan for available ports
    ports = scan_com_ports()
    
    if not ports:
        print("\nNo COM ports detected!")
        return
    
    print(f"\nFound {len(ports)} COM port(s):")
    
    # Display information for each port
    for port_info in ports:
        # Test port connectivity
        test_result = test_port_connectivity(port_info['device'])
        
        # Print port information
        print_port_info(port_info, test_result)
    
    print(f"\n{'='*60}")
    print("Scan completed.")


def list_ports_simple():
    """
    Simple function to just list port names - useful for quick checks.
    """
    ports = serial.tools.list_ports.comports()
    port_names = [port.device for port in ports]
    return port_names


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScan interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
