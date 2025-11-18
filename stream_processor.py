#!/usr/bin/env python3
"""
Stream Processor
Handles RTSP stream reading and processing
Two separate threads: one for direct display (no delays), one for background processing
"""

import cv2
import numpy as np
import logging
import threading
import queue
import time
import os
from typing import Optional, Tuple

# Set RTSP transport to TCP for more reliable connection (UDP can drop packets)
# This must be set before creating VideoCapture objects
if 'OPENCV_FFMPEG_CAPTURE_OPTIONS' not in os.environ:
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'


class StreamProcessor:
    """Manages RTSP stream with separate display and processing threads."""
    
    def __init__(self, rtsp_url: str = "rtsp://192.168.0.100:9079/vis"):
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_streaming = False
        
        # Display thread - direct frame access, no delays
        self.display_frame = None
        self.display_lock = threading.Lock()
        self.frame_ready = threading.Event()
        
        # Frame reading thread - continuously reads frames from RTSP
        self.read_thread = None
        self.last_read_time = 0
        
        # Processing thread - separate queue for background processing
        self.processing_queue = queue.Queue(maxsize=1)  # Small queue, drop frames if full
        self.processing_thread = None
        
    def start_stream(self) -> bool:
        """Start RTSP stream connection and processing thread."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            # Open RTSP stream with optimized settings for low latency
            # Use CAP_FFMPEG backend for better RTSP support
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to 1 frame to reduce latency (default is often 30+ frames)
            # This prevents frame buffering which causes delays
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # RTSP transport is set at module level (TCP for reliability)
            
            # Try to set other properties for better RTSP performance
            # Note: Some properties may not be supported by all streams
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            except:
                pass  # Ignore if not supported
            
            if not self.cap.isOpened():
                logging.error(f"Could not open RTSP stream: {self.rtsp_url}")
                return False
            
            self.is_streaming = True
            
            # Start frame reading thread - continuously reads and caches latest frame
            self.read_thread = threading.Thread(target=self._frame_reading_loop, daemon=True)
            self.read_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logging.info(f"Stream started: {self.rtsp_url}")
            return True
            
        except Exception as e:
            logging.error(f"Error starting stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop RTSP stream connection."""
        self.is_streaming = False
        
        # Wait a bit for threads to finish
        if self.read_thread is not None:
            self.read_thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear processing queue
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                break
        
        with self.display_lock:
            self.display_frame = None
            self.frame_ready.clear()
        
        logging.info("Stream stopped")
    
    def _frame_reading_loop(self):
        """
        Dedicated thread that continuously reads frames from RTSP stream.
        This prevents blocking when get_frame() is called frequently.
        """
        consecutive_errors = 0
        max_errors = 10
        
        while self.is_streaming:
            try:
                if self.cap is None:
                    time.sleep(0.01)
                    continue
                
                # Read frame from RTSP (this may block, but it's in a separate thread)
                ret, frame = self.cap.read()
                
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        logging.warning(f"Failed to read frame {consecutive_errors} times, stream may be disconnected")
                        consecutive_errors = 0
                    time.sleep(0.01)  # Small delay before retry
                    continue
                
                consecutive_errors = 0
                self.last_read_time = time.time()
                
                # Store latest frame for display (thread-safe, overwrites previous)
                with self.display_lock:
                    self.display_frame = frame.copy()
                    self.frame_ready.set()
                
                # Send copy to processing queue (non-blocking, drop if full)
                try:
                    self.processing_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Processing is slow, skip this frame - display is priority
                
                # Small sleep to prevent CPU spinning, but keep it responsive
                time.sleep(0.001)  # 1ms sleep
                
            except Exception as e:
                logging.error(f"Error in frame reading thread: {e}")
                time.sleep(0.1)  # Longer sleep on error
                continue
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame for display - NON-BLOCKING, returns cached latest frame.
        This is called frequently for smooth display.
        """
        if not self.is_streaming:
            return None
        
        # Return cached frame immediately (non-blocking)
        with self.display_lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
        
        return None
    
    def _processing_loop(self):
        """Background thread for processing frames - does not affect display."""
        while self.is_streaming:
            try:
                # Get frame from queue (with timeout to check is_streaming)
                try:
                    frame = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process frame through pipeline (for internal calculations only)
                processed = self._process_frame(frame)
                
                # Processed frame is for internal use only, not displayed
                # Store or use processed frame here if needed for calculations
                
            except Exception as e:
                logging.error(f"Error in processing thread: {e}")
                continue
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame through all processing functions.
        Currently passes through unchanged, but structure is ready for future processing.
        """
        # ============================================
        # REGISTRATION PROCESSING
        # ============================================
        processed = self._registration_processing(frame)
        
        # ============================================
        # FOCUS PROCESSING
        # ============================================
        processed = self._focus_processing(processed)
        
        # ============================================
        # GAIN PROCESSING
        # ============================================
        processed = self._gain_processing(processed)
        
        # ============================================
        # ZOOM PROCESSING
        # ============================================
        processed = self._zoom_processing(processed)
        
        return processed
    
    # ============================================
    # REGISTRATION PROCESSING FUNCTIONS
    # ============================================
    def _registration_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply registration processing to frame.
        Placeholder for future implementation.
        """
        # TODO: Implement registration processing
        return frame
    
    # ============================================
    # FOCUS PROCESSING FUNCTIONS
    # ============================================
    def _focus_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply focus processing to frame.
        Placeholder for future implementation.
        """
        # TODO: Implement focus processing
        return frame
    
    # ============================================
    # GAIN PROCESSING FUNCTIONS
    # ============================================
    def _gain_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gain processing to frame.
        Placeholder for future implementation.
        """
        # TODO: Implement gain processing
        return frame
    
    # ============================================
    # ZOOM PROCESSING FUNCTIONS
    # ============================================
    def _zoom_processing(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply zoom processing to frame.
        Placeholder for future implementation.
        """
        # TODO: Implement zoom processing
        return frame

