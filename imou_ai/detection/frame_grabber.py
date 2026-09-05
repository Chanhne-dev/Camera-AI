"""
frame_grabber.py - Thêm timestamp vào log để debug
"""

import os
import threading
import time
import cv2
from datetime import datetime

# Ep RTSP dung TCP
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|rw_timeout;5000000|max_delay;500000"
)

def _log(msg):
    """In log kèm timestamp (định dạng: YYYY-MM-DD HH:MM:SS.mmm)"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")

class FrameGrabber:
    def __init__(self, url, reconnect_delay=2.0, stale_timeout=8.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.stale_timeout = stale_timeout

        self.cap = None
        self.cap_lock = threading.Lock()
        self.frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0
        self.connected = False

        self.stop_flag = threading.Event()
        self.grab_thread = None
        self.watchdog_thread = None

    def start(self):
        self.stop_flag.clear()
        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.grab_thread.start()
        self.watchdog_thread.start()
        _log("[GRABBER] Started")

    def stop(self):
        self.stop_flag.set()
        if self.grab_thread is not None:
            self.grab_thread.join(timeout=5)
        if self.watchdog_thread is not None:
            self.watchdog_thread.join(timeout=5)
        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        _log("[GRABBER] Stopped")

    def _open(self):
        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(self.url)
            self.connected = self.cap.isOpened()
        if self.connected:
            _log("[GRABBER] Camera connected")
        else:
            _log("[GRABBER] Camera connection failed")
        return self.connected

    def _grab_loop(self):
        _log("[GRABBER] Grab loop started")
        self._open()
        while not self.stop_flag.is_set():
            with self.cap_lock:
                cap = self.cap
            if cap is None or not cap.isOpened():
                _log("[GRABBER] Cap is None or not opened, reconnecting...")
                time.sleep(self.reconnect_delay)
                self._open()
                continue
            try:
                ret, frame = cap.read()
            except Exception as e:
                _log(f"[GRABBER] Exception in cap.read(): {e}")
                time.sleep(0.5)
                continue
            if not ret or frame is None:
                _log("[GRABBER] cap.read() returned False or None frame")
                time.sleep(0.2)
                continue
            with self.frame_lock:
                self.frame = frame
                self.last_frame_time = time.time()
        _log("[GRABBER] Grab loop ended")

    def _watchdog_loop(self):
        _log("[GRABBER] Watchdog started")
        while not self.stop_flag.is_set():
            time.sleep(1.0)
            with self.frame_lock:
                last = self.last_frame_time
            if last == 0:
                continue
            elapsed = time.time() - last
            if elapsed > self.stale_timeout:
                _log(f"[GRABBER][WATCHDOG] No new frame for {elapsed:.1f}s (>{self.stale_timeout}s) - forcing reconnect...")
                with self.cap_lock:
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                with self.frame_lock:
                    self.last_frame_time = time.time()
        _log("[GRABBER] Watchdog ended")

    def get_latest(self, max_age=None):
        with self.frame_lock:
            if self.frame is None:
                return None
            if max_age is not None and (time.time() - self.last_frame_time) > max_age:
                return None
            return self.frame.copy()

    def is_healthy(self):
        with self.frame_lock:
            if self.last_frame_time == 0:
                return False
            return (time.time() - self.last_frame_time) <= self.stale_timeout