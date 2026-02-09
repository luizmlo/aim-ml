# window_selector.py

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Optional
from recorder import Recorder
from recorder import find_window_handle, capture_window_frame
import threading
from PIL import Image, ImageTk
import numpy as np

class RecordingControl(tk.Toplevel):
    """Complete recording control center"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Recording Control Center")
        self.geometry("650x650")
        self.resizable(True, True)
        
        # Center the window
        self.transient(parent)
        self.grab_set()
        
        # Recording state
        self.recorder_instance = None
        self.recording = False
        
        # Preview state
        self.preview_running = False
        self.preview_thread = None
        self.current_preview_window = None
        
        # Result variables
        self.result = None
        
        self._setup_ui()
        self._refresh_windows()
        
        # Set up close protocol
        self.protocol("WM_DELETE_WINDOW", self._close)
        
        # Auto-focus
        self.focus_force()
        self.lift()
        
    def _setup_ui(self):
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        ttk.Label(main_frame, text="Recording Control Center", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 15))
        
        # Session Configuration Frame
        config_frame = ttk.LabelFrame(main_frame, text="Session Configuration", padding="10")
        config_frame.pack(fill="x", pady=(0, 10))
        
        # Session name
        session_frame = ttk.Frame(config_frame)
        session_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(session_frame, text="Session Name:", width=15).pack(side="left")
        self.session_var = tk.StringVar()
        session_entry = ttk.Entry(session_frame, textvariable=self.session_var, width=30)
        session_entry.pack(side="left", padx=(5, 0))
        
        # FPS setting
        fps_frame = ttk.Frame(config_frame)
        fps_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(fps_frame, text="FPS:", width=15).pack(side="left")
        self.fps_var = tk.IntVar(value=10)
        fps_spinbox = ttk.Spinbox(fps_frame, from_=1, to=60, textvariable=self.fps_var, width=10)
        fps_spinbox.pack(side="left", padx=(5, 0))
        
        # Window Selection Frame
        window_frame = ttk.LabelFrame(main_frame, text="Window Selection", padding="10")
        window_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(window_frame, text="Select window to capture:").pack(anchor="w")
        
        # Dropdown and refresh button frame
        dropdown_frame = ttk.Frame(window_frame)
        dropdown_frame.pack(fill="x", pady=(5, 0))
        
        self.window_var = tk.StringVar()
        self.window_combo = ttk.Combobox(dropdown_frame, textvariable=self.window_var, state="readonly")
        self.window_combo.pack(side="left", fill="x", expand=True)
        
        refresh_button = ttk.Button(dropdown_frame, text="Refresh", command=self._refresh_windows)
        refresh_button.pack(side="right", padx=(10, 0))
        
        # Bind selection change
        self.window_combo.bind("<<ComboboxSelected>>", self._on_window_selected)
        
        # Preview Frame
        preview_frame = ttk.LabelFrame(main_frame, text="Window Preview", padding="10")
        preview_frame.pack(fill="x", pady=(0, 10))
        preview_frame.pack_propagate(False)  # Don't shrink
        preview_frame.configure(height=170)  # Fixed height for preview
        
        self.preview_label = ttk.Label(preview_frame, text="Select a window to see preview", anchor="center", background="#f0f0f0")
        self.preview_label.pack(fill="both", expand=True)
        
        # Recording Controls Frame
        controls_frame = ttk.LabelFrame(main_frame, text="Recording Controls", padding="10")
        controls_frame.pack(fill="x", pady=(0, 10))
        
        # Status display
        self.status_var = tk.StringVar(value="Ready to record")
        status_label = ttk.Label(controls_frame, textvariable=self.status_var, font=("Arial", 10, "bold"))
        status_label.pack(anchor="w", pady=(0, 10))
        
        # Control buttons frame
        control_buttons_frame = ttk.Frame(controls_frame)
        control_buttons_frame.pack(fill="x")
        
        self.start_button = ttk.Button(control_buttons_frame, text="Start Recording", command=self._start_recording)
        self.start_button.pack(side="left", padx=(0, 10))
        
        self.pause_button = ttk.Button(control_buttons_frame, text="Pause", command=self._pause_recording, state="disabled")
        self.pause_button.pack(side="left", padx=(0, 10))
        
        self.stop_button = ttk.Button(control_buttons_frame, text="Stop", command=self._stop_recording, state="disabled")
        self.stop_button.pack(side="left")
        
        # Recording info frame
        info_frame = ttk.LabelFrame(main_frame, text="Recording Information", padding="10")
        info_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=6, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_frame = ttk.Frame(main_frame)
        close_frame.pack(fill="x")
        ttk.Button(close_frame, text="Close", command=self._close).pack(side="right")
        
        # Start periodic status updates
        self._update_status()
        
    def _get_windows_list(self) -> List[Tuple[str, str, str]]:
        """Get list of available windows (title, class_name, process)"""
        windows = []
        
        # Add whole screen option first
        windows.append(("Whole Screen", "DESKTOP", "System"))
        
        try:
            import win32gui
            
            def enum_windows_callback(hwnd, windows_list):
                if win32gui.IsWindowVisible(hwnd):
                    window_title = win32gui.GetWindowText(hwnd)
                    if window_title:  # Only include windows with titles
                        try:
                            class_name = win32gui.GetClassName(hwnd)
                            # Get process info
                            import win32process
                            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                            
                            try:
                                import psutil
                                process = psutil.Process(process_id)
                                process_name = process.name()
                            except:
                                process_name = f"PID:{process_id}"
                            
                            windows_list.append((window_title, class_name, process_name))
                        except:
                            windows_list.append((window_title, "Unknown", "Unknown"))
            
            win32gui.EnumWindows(enum_windows_callback, windows)
            
        except ImportError:
            print("Warning: pywin32 not installed. Only 'Whole Screen' option available.")
        except Exception as e:
            print(f"Error getting windows list: {e}")
        
        return windows
    
    def _refresh_windows(self):
        """Refresh the list of available windows"""
        windows = self._get_windows_list()
        
        # Create display strings for the dropdown
        self.windows_data = windows
        display_options = []
        
        for title, class_name, process in windows:
            if title == "Whole Screen":
                display_options.append("Whole Screen (Default)")
            else:
                display_options.append(f"{title} ({process})")
        
        self.window_combo['values'] = display_options
        
        # Select whole screen by default
        if display_options:
            self.window_combo.set(display_options[0])
            self._on_window_selected()
        else:
            # No windows available, stop preview
            self._stop_preview()
            self.preview_label.config(image="", text="No windows available")
    
    def _on_window_selected(self, event=None):
        """Handle window selection change"""
        self._update_info_display()
        self._start_preview()
    
    def _update_info_display(self):
        """Update the information display with current selection and recording status"""
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")
        
        # Window selection info
        selection = self.window_combo.get()
        if selection:
            selected_index = self.window_combo.current()
            if 0 <= selected_index < len(self.windows_data):
                title, class_name, process = self.windows_data[selected_index]
                
                info_text = f"Selected Window:\n"
                info_text += f"  Title: {title}\n"
                info_text += f"  Class: {class_name}\n"
                info_text += f"  Process: {process}\n\n"
                
                if title == "Whole Screen":
                    info_text += "Capture Mode: Entire screen\n"
                    info_text += "Method: Screen capture\n\n"
                else:
                    info_text += f"Capture Mode: Single window '{title}'\n"
                    info_text += "Method: Window content capture (bypasses overlays)\n\n"
        else:
            info_text = "No window selected\n\n"
        
        # Recording status info
        if self.recorder_instance:
            info_text += f"Session: {self.recorder_instance.dataset.session_name}\n"
            info_text += f"Target FPS: {self.recorder_instance.fps}\n"
            if self.recorder_instance.recording:
                state = "Paused" if self.recorder_instance.paused else "Recording"
                info_text += f"Status: {state}\n"
                info_text += f"Frames Captured: {self.recorder_instance.frame_counter}\n"
            else:
                info_text += "Status: Stopped\n"
        
        self.info_text.insert("1.0", info_text)
        self.info_text.config(state="disabled")
    
    def _start_recording(self):
        """Start recording with current settings"""
        # Validate inputs
        session_name = self.session_var.get().strip()
        if not session_name:
            self.status_var.set("Error: Please enter a session name")
            return
        
        if not session_name.isalnum():
            self.status_var.set("Error: Session name must be alphanumeric")
            return
        
        selected_index = self.window_combo.current()
        if selected_index < 0:
            self.status_var.set("Error: Please select a window")
            return
        
        # Get selected window
        window_title = self.windows_data[selected_index][0]
        fps = self.fps_var.get()
        
        # Create and start recorder
        try:
            self.recorder_instance = Recorder(session_name, window_title, fps)
            self.recorder_instance.start()
            
            # Update UI state
            self.start_button.config(state="disabled")
            self.pause_button.config(state="normal")
            self.stop_button.config(state="normal")
            self.status_var.set("Recording started")
            
            print(f"Recording started: {session_name} at {fps} FPS")
            
        except Exception as e:
            self.status_var.set(f"Error: Failed to start recording - {e}")
            print(f"Error starting recording: {e}")
    
    def _pause_recording(self):
        """Pause or resume recording"""
        if not self.recorder_instance:
            return
        
        if self.recorder_instance.paused:
            self.recorder_instance.resume()
            self.pause_button.config(text="Pause")
            self.status_var.set("Recording resumed")
        else:
            self.recorder_instance.pause()
            self.pause_button.config(text="Resume")
            self.status_var.set("Recording paused")
    
    def _stop_recording(self):
        """Stop recording"""
        if not self.recorder_instance:
            return
        
        try:
            self.recorder_instance.stop()
            frames_captured = self.recorder_instance.frame_counter
            
            # Update UI state
            self.start_button.config(state="normal")
            self.pause_button.config(state="disabled", text="Pause")
            self.stop_button.config(state="disabled")
            self.status_var.set(f"Recording stopped - {frames_captured} frames captured")
            
            print(f"Recording stopped. Total frames captured: {frames_captured}")
            
        except Exception as e:
            self.status_var.set(f"Error stopping recording: {e}")
            print(f"Error stopping recording: {e}")
    
    def _update_status(self):
        """Periodically update status and info display"""
        self._update_info_display()
        self.after(500, self._update_status)  # Update every 500ms
    
    def _start_preview(self):
        """Start live preview of selected window"""
        # Stop any existing preview
        self._stop_preview()
        
        selected_index = self.window_combo.current()
        if selected_index < 0 or selected_index >= len(self.windows_data):
            self.preview_label.config(image="", text="Select a window to see preview")
            return
        
        window_title = self.windows_data[selected_index][0]
        self.current_preview_window = window_title
        
        # Start preview thread
        self.preview_running = True
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()
    
    def _stop_preview(self):
        """Stop the live preview"""
        self.preview_running = False
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=1.0)
        self.current_preview_window = None
    
    def _preview_loop(self):
        """Preview loop running in separate thread"""
        import time
        
        while self.preview_running and self.current_preview_window:
            try:
                # Get window handle
                hwnd = find_window_handle(self.current_preview_window)
                
                # Capture frame
                frame = capture_window_frame(hwnd)
                
                if frame:
                    # Convert PIL image to numpy array if needed
                    if hasattr(frame, 'convert'):
                        frame_rgb = frame.convert('RGB')
                        frame_array = np.array(frame_rgb)
                    else:
                        frame_array = frame
                    
                    # Create thumbnail (max 300x200)
                    height, width = frame_array.shape[:2]
                    max_width, max_height = 300, 150
                    
                    if width > max_width or height > max_height:
                        ratio = min(max_width / width, max_height / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        
                        # Convert back to PIL for resizing
                        pil_image = Image.fromarray(frame_array)
                        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    else:
                        pil_image = Image.fromarray(frame_array)
                    
                    # Convert to PhotoImage and update label
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update UI from main thread
                    self.after(0, self._update_preview_image, photo)
                else:
                    # No frame captured
                    self.after(0, self._update_preview_text, "Cannot capture selected window")
                
                time.sleep(0.1)  # Update at ~10 FPS for preview
                
            except Exception as e:
                self.after(0, self._update_preview_text, f"Preview error: {str(e)[:50]}")
                time.sleep(0.5)  # Slower retry on error
    
    def _update_preview_image(self, photo):
        """Update preview with new image (called from main thread)"""
        if self.preview_running:
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # Keep a reference
    
    def _update_preview_text(self, text):
        """Update preview with text message (called from main thread)"""
        self.preview_label.config(image="", text=text)
        if hasattr(self.preview_label, 'image'):
            del self.preview_label.image
    
    def _close(self):
        """Handle close button or window close"""
        # Stop preview first
        self._stop_preview()
        
        if self.recorder_instance and self.recorder_instance.recording:
            self._stop_recording()
        self.result = "close"
        self.destroy()
    
    def get_result(self):
        """Get the result after dialog closes"""
        self.wait_window()  # Wait for dialog to close
        return self.result

# Alias for backwards compatibility
WindowSelector = RecordingControl
