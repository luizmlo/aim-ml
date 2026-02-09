# recorder.py

import time
import threading
import os
from PIL import Image, ImageGrab
from dataset import Dataset
from typing import Optional

# --- Windows API Placeholders ---
# These are improved placeholders. If pywin32 is installed, they will attempt
# to capture the specific window. Otherwise, they fall back to capturing the
# entire screen.

def find_window_handle(window_title: str) -> int:
    """Finds a window handle by its title. Returns 0 for whole screen or if not found."""
    # Handle whole screen case
    if window_title == "Whole Screen" or not window_title:
        return 0
    
    try:
        import win32gui
        
        def enum_windows_callback(hwnd, target_title):
            if win32gui.IsWindowVisible(hwnd):
                current_title = win32gui.GetWindowText(hwnd)
                if current_title == target_title:
                    return hwnd
            return None
        
        # Try to find exact match first
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd != 0:
            return hwnd
        
        # If exact match failed, try enumeration (more reliable)
        found_hwnd = [0]
        def callback(hwnd, param):
            if win32gui.IsWindowVisible(hwnd):
                current_title = win32gui.GetWindowText(hwnd)
                if current_title == window_title:
                    found_hwnd[0] = hwnd
                    return False  # Stop enumeration
            return True
        
        win32gui.EnumWindows(callback, None)
        
        if found_hwnd[0] == 0:
            print(f"Warning: Window '{window_title}' not found. Capturing full screen.")
        
        return found_hwnd[0]
        
    except ImportError:
        print("Warning: pywin32 not installed. Capturing full screen.")
        return 0
    except Exception as e:
        print(f"An error occurred finding the window: {e}")
        return 0

def capture_window_frame(hwnd: int) -> Optional[Image.Image]:
    """Captures a single frame from the window handle using proper window capture."""
    try:
        if hwnd != 0:
            import win32gui, win32ui, win32con
            from ctypes import windll
            
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            if width <= 0 or height <= 0:
                return None
            
            # Get window device context
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)
            
            # Print window to bitmap (this captures the actual window content)
            # Try different PrintWindow flags for better compatibility
            result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)  # PW_RENDERFULLCONTENT = 3
            
            # If that fails, try with different flags
            if not result:
                result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)  # PW_CLIENTONLY = 2
            if not result:
                result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)  # Default
            
            # Convert to PIL Image
            if result:
                bmpinfo = save_bitmap.GetInfo()
                bmpstr = save_bitmap.GetBitmapBits(True)
                
                # Convert to PIL Image
                from PIL import Image
                img = Image.frombuffer(
                    'RGB',
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1
                )
                
                # Clean up
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
                
                return img
            else:
                # PrintWindow failed, fallback to screen capture
                print(f"PrintWindow failed for window {hwnd}, falling back to screen capture")
                
                # Clean up first
                win32gui.DeleteObject(save_bitmap.GetHandle())
                save_dc.DeleteDC()
                mfc_dc.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwnd_dc)
                
                # Fallback to screen region capture
                return ImageGrab.grab(bbox=(left, top, right, bottom))
                
        else:
            # Whole screen capture
            return ImageGrab.grab()
            
    except ImportError:
        print("Warning: pywin32 not installed. Using screen capture fallback.")
        if hwnd != 0:
            try:
                import win32gui
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                return ImageGrab.grab(bbox=(left, top, right, bottom))
            except:
                pass
        return ImageGrab.grab()
    except Exception as e:
        print(f"Error capturing window frame: {e}")
        # Fallback to screen capture if available
        if hwnd != 0:
            try:
                import win32gui
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                return ImageGrab.grab(bbox=(left, top, right, bottom))
            except:
                pass
        return ImageGrab.grab()

class Recorder:
    """Handles recording frames from a specified window."""
    def __init__(self, session_name: str, window_title: str, fps: int = 10):
        self.window_title = window_title
        self.fps = fps
        self.dataset = Dataset(session_name)
        
        self.recording = False
        self.paused = False
        self._thread: Optional[threading.Thread] = None
        self.frame_counter = 0

    def start(self):
        """Starts the recording thread."""
        if self.recording:
            return
        self.recording = True
        self.paused = False
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print(f"Recording started. Target FPS: {self.fps}")

    def pause(self):
        if self.recording and not self.paused:
            self.paused = True
            print("Recording paused.")

    def resume(self):
        if self.recording and self.paused:
            self.paused = False
            print("Recording resumed.")

    def stop(self):
        if self.recording:
            self.recording = False
            if self._thread:
                self._thread.join()
            print(f"Recording stopped. Total frames captured: {self.frame_counter}")

    @property
    def is_recording(self) -> bool:
        """Alias for recording state (main_app compatibility)."""
        return self.recording

    def stop_recording(self):
        """Alias for stop() (main_app compatibility)."""
        self.stop()

    def _record_loop(self):
        """The main loop for capturing frames in a separate thread."""
        hwnd = find_window_handle(self.window_title)
        
        existing_frames = os.listdir(self.dataset.raw_images_path)
        if existing_frames:
            try:
                last_frame_num = max(int(f.replace('frame', '').split('.')[0]) for f in existing_frames if f.startswith('frame'))
                self.frame_counter = last_frame_num + 1
            except (ValueError, IndexError):
                self.frame_counter = 0

        print(f"Starting frame count at: {self.frame_counter}")
        interval = 1.0 / self.fps

        while self.recording:
            if self.paused:
                time.sleep(0.1)
                continue

            start_time = time.time()
            
            try:
                frame = capture_window_frame(hwnd)
                if frame:
                    filename = f"frame{self.frame_counter:05d}.jpg"
                    save_path = os.path.join(self.dataset.raw_images_path, filename)
                    frame.convert('RGB').save(save_path, 'jpeg')
                    self.frame_counter += 1
            except Exception as e:
                print(f"Error in recording loop: {e}")

            elapsed = time.time() - start_time
            sleep_time = interval - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # *** THE CRITICAL FIX ***
                # If we are running late, still sleep for a tiny moment (1ms)
                # This yields control to the OS and allows the GUI thread to run.
                time.sleep(0.001)