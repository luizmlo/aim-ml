# main_app.py

import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
import os
import time
import threading
from datetime import datetime

# Import all the components we need
from recorder import Recorder
from dataset import Dataset
from window_selector import RecordingControl
from ml_components import (ModelTrainer, ModelMetadata, LiveTester, ARCHITECTURE_TYPES, 
                           ARCHITECTURE_SIZES, estimate_model_parameters, DEFAULT_INPUT_RESOLUTION,
                           DEFAULT_NUM_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE)
from labeler import LabelerApp
from PIL import Image, ImageTk
import mss
import numpy as np

# Window data class for storing window information
class WindowInfo:
    def __init__(self, title, class_name, process, left=0, top=0, width=1920, height=1080):
        self.title = title
        self.class_name = class_name
        self.process = process
        self.left = left
        self.top = top
        self.width = width
        self.height = height

def get_windows_list():
    """Get list of available windows (filtered to show only actual windows)"""
    windows = []
    
    # Add whole screen option first
    windows.append(WindowInfo("Whole Screen", "DESKTOP", "System", 0, 0, 1920, 1080))
    
    try:
        import win32gui
        import win32process
        import win32con
        
        def enum_windows_callback(hwnd, windows_list):
            # Must be visible
            if not win32gui.IsWindowVisible(hwnd):
                return True
            
            # Must have a window title
            window_title = win32gui.GetWindowText(hwnd)
            if not window_title or len(window_title.strip()) == 0:
                return True
            
            # Skip windows that are likely not user applications
            class_name = win32gui.GetClassName(hwnd)
            
            # Filter out system/shell windows and other non-application windows
            skip_classes = {
                'Progman',           # Desktop
                'WorkerW',           # Desktop worker
                'Shell_TrayWnd',     # Taskbar
                'DV2ControlHost',    # Windows shell
                'MsgrIMEWindowClass', # Input method
                'IME',               # Input method
                'MSCTFIME UI',       # Input method
                'Windows.UI.Core.CoreWindow',  # Modern UI windows (often invisible)
                'ApplicationFrameWindow',      # UWP container (we want the inner window)
                'ForegroundStaging', # Windows staging
                'SHELLDLL_DefView',  # Shell view
                'Button',            # System buttons
                'Static',            # Static controls
                'SysShadow',         # Window shadows
            }
            
            if class_name in skip_classes:
                return True
                
            # Skip windows with certain title patterns
            title_lower = window_title.lower()
            skip_title_patterns = [
                'program manager',
                'desktop window manager',
                'microsoft text input application',
                'windows input experience',
                'search',            # Often Windows search
                'cortana',
                'windows shell experience',
            ]
            
            for pattern in skip_title_patterns:
                if pattern in title_lower:
                    return True
            
            # Check if window has proper dimensions (not minimized or too small)
            try:
                rect = win32gui.GetWindowRect(hwnd)
                left, top, right, bottom = rect
                width = right - left
                height = bottom - top
                
                # Skip tiny windows (likely not real application windows)
                if width < 50 or height < 50:
                    return True
                    
                # Skip windows that are way off-screen
                if left < -2000 or top < -2000 or left > 5000 or top > 5000:
                    return True
                    
            except:
                # If we can't get rect, skip it
                return True
            
            # Check if window is cloaked (Windows 8+ feature for hidden windows)
            try:
                import ctypes
                from ctypes import wintypes
                
                # Define the DwmGetWindowAttribute function
                dwmapi = ctypes.windll.dwmapi
                DwmGetWindowAttribute = dwmapi.DwmGetWindowAttribute
                DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.POINTER(wintypes.BOOL), wintypes.DWORD]
                DwmGetWindowAttribute.restype = wintypes.LONG
                
                # DWMWA_CLOAKED = 14
                cloaked = wintypes.BOOL()
                result = DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
                
                if result == 0 and cloaked.value:
                    return True  # Skip cloaked windows
                    
            except:
                # If DWM API fails, continue anyway
                pass
            
            # Get process info
            try:
                _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                
                try:
                    import psutil
                    process = psutil.Process(process_id)
                    process_name = process.name()
                    
                    # Filter out some system processes that shouldn't be recorded
                    skip_processes = {
                        'dwm.exe',           # Desktop Window Manager
                        'winlogon.exe',      # Windows Logon
                        'csrss.exe',         # Client Server Runtime
                        'lsass.exe',         # Local Security Authority
                        'services.exe',      # Service Control Manager
                        'svchost.exe',       # Service Host (too generic)
                        'explorer.exe',      # Only if it's a background window
                        'searchui.exe',      # Windows Search
                        'cortana.exe',       # Cortana
                        'sihost.exe',        # Shell Infrastructure Host
                        'taskhostw.exe',     # Task Host Window
                        'dllhost.exe',       # COM Surrogate
                        'runtimebroker.exe', # Runtime Broker
                    }
                    
                    if process_name.lower() in skip_processes:
                        return True
                        
                except:
                    process_name = f"PID:{process_id}"
                
                # Add to list
                windows_list.append(WindowInfo(window_title, class_name, process_name, left, top, width, height))
                
            except Exception as e:
                # If we can't get process info, still add the window but with unknown process
                windows_list.append(WindowInfo(window_title, class_name, "Unknown", left, top, width, height))
                
            return True
        
        win32gui.EnumWindows(enum_windows_callback, windows)
        
    except ImportError:
        print("Warning: pywin32 not installed. Only 'Whole Screen' option available.")
    except Exception as e:
        print(f"Error getting windows list: {e}")
    
    return windows

# Add DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

class IntegratedApp(tk.Tk):
    """Main application with integrated tabbed interface"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Aim-ML - Screen-to-Coordinate Prediction")
        self.geometry("1400x900")
        
        # Maximize window by default
        try:
            self.state('zoomed')  # Windows maximized
        except:
            self.geometry("1600x1000")
            
        # Initialize variables for different components
        self.recorder_instance = None
        self.labeler_instance = None
        self.model_trainer = None
        self.live_tester = None
        self.model_metadata = ModelMetadata()
        
        # Create the main interface
        self._setup_main_interface()
        self._setup_recording_tab()
        self._setup_labeling_tab()
        self._setup_training_tab()
        self._setup_testing_tab()
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Auto-focus
        self.focus_force()
        self.lift()
    
    def _setup_main_interface(self):
        """Setup the main tabbed interface"""
        # Main container
        main_container = ttk.Frame(self, padding="10")
        main_container.pack(fill="both", expand=True)
        
        # Title bar
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(title_frame, text="Aim-ML", 
                 font=("Arial", 18, "bold")).pack(side="left")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(title_frame, textvariable=self.status_var, 
                                font=("Arial", 10), foreground="#666666")
        status_label.pack(side="right")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tab frames
        self.recording_frame = ttk.Frame(self.notebook, padding="20")
        self.labeling_frame = ttk.Frame(self.notebook, padding="20")
        self.training_frame = ttk.Frame(self.notebook, padding="20")
        self.testing_frame = ttk.Frame(self.notebook, padding="20")
        
        # Add tabs to notebook
        self.notebook.add(self.recording_frame, text="Recording")
        self.notebook.add(self.labeling_frame, text="Labeling")
        self.notebook.add(self.training_frame, text="Training")
        self.notebook.add(self.testing_frame, text="Testing")
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
    
    def _setup_recording_tab(self):
        """Setup the recording tab interface"""
        # Title
        ttk.Label(self.recording_frame, text="Recording Session", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Session Configuration
        config_frame = ttk.LabelFrame(self.recording_frame, text="Session Configuration", padding="15")
        config_frame.pack(fill="x", pady=(0, 15))
        
        # Session name
        session_frame = ttk.Frame(config_frame)
        session_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(session_frame, text="Session Name:", width=15).pack(side="left")
        self.session_name_var = tk.StringVar()
        ttk.Entry(session_frame, textvariable=self.session_name_var, width=30).pack(side="left", padx=(10, 0))
        
        # FPS setting
        fps_frame = ttk.Frame(config_frame)
        fps_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(fps_frame, text="Recording FPS:", width=15).pack(side="left")
        self.fps_var = tk.IntVar(value=10)
        ttk.Spinbox(fps_frame, from_=1, to=60, textvariable=self.fps_var, width=10).pack(side="left", padx=(10, 0))
        
        # Window selection
        window_frame = ttk.LabelFrame(self.recording_frame, text="Window Selection", padding="15")
        window_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(window_frame, text="Target Window:").pack(anchor="w")
        
        window_select_frame = ttk.Frame(window_frame)
        window_select_frame.pack(fill="x", pady=(10, 0))
        
        self.window_var = tk.StringVar()
        self.window_combo = ttk.Combobox(window_select_frame, textvariable=self.window_var, state="readonly")
        self.window_combo.pack(side="left", fill="x", expand=True)
        
        ttk.Button(window_select_frame, text="Refresh", 
                  command=self._refresh_windows).pack(side="right", padx=(10, 0))
        
        # Preview frame (simplified layout for better image display)
        preview_frame = ttk.LabelFrame(self.recording_frame, text="Window Preview", padding="10")
        preview_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Create preview label with proper settings
        self.recording_preview_label = tk.Label(preview_frame, 
                                               text="Select a window to see preview",
                                               anchor="center", 
                                               bg="#2b2b2b", 
                                               fg="white", 
                                               font=("Arial", 11),
                                               width=60,
                                               height=15)
        self.recording_preview_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Recording controls
        controls_frame = ttk.LabelFrame(self.recording_frame, text="Recording Controls", padding="15")
        controls_frame.pack(fill="x")
        
        # Status display
        self.recording_status_var = tk.StringVar(value="Ready to record")
        ttk.Label(controls_frame, textvariable=self.recording_status_var, 
                 font=("Arial", 10, "bold")).pack(pady=(0, 10))
        
        # Control buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack()
        
        self.start_rec_button = ttk.Button(button_frame, text="Start Recording", 
                                          command=self._start_recording)
        self.start_rec_button.pack(side="left", padx=(0, 10))
        
        self.pause_rec_button = ttk.Button(button_frame, text="Pause", 
                                          command=self._pause_recording, state="disabled")
        self.pause_rec_button.pack(side="left", padx=(0, 10))
        
        self.stop_rec_button = ttk.Button(button_frame, text="Stop", 
                                         command=self._stop_recording, state="disabled")
        self.stop_rec_button.pack(side="left")
        
        # Initialize preview variables
        self._recording_preview_running = False
        self._recording_preview_thread = None
        self.windows_data = []
        
        # Initialize window list
        self._refresh_windows()
        self.window_combo.bind("<<ComboboxSelected>>", self._on_recording_window_selected)
        
    def _setup_labeling_tab(self):
        """Setup the labeling tab interface"""
        # Title
        ttk.Label(self.labeling_frame, text="Dataset Labeling", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(self.labeling_frame, text="Dataset Selection", padding="15")
        dataset_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(dataset_frame, text="Select dataset to label:").pack(anchor="w")
        
        dataset_select_frame = ttk.Frame(dataset_frame)
        dataset_select_frame.pack(fill="x", pady=(10, 0))
        
        self.label_dataset_var = tk.StringVar()
        self.label_dataset_combo = ttk.Combobox(dataset_select_frame, textvariable=self.label_dataset_var, 
                                               state="readonly")
        self.label_dataset_combo.pack(side="left", fill="x", expand=True)
        
        ttk.Button(dataset_select_frame, text="Refresh", 
                  command=self._refresh_labeling_datasets).pack(side="right", padx=(10, 0))
        
        ttk.Button(dataset_select_frame, text="Start Labeling", 
                  command=self._start_labeling).pack(side="right", padx=(10, 0))
        
        # Dataset gallery preview with split layout
        gallery_frame = ttk.LabelFrame(self.labeling_frame, text="Dataset Preview", padding="10")
        gallery_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Split into left (thumbnails) and right (preview) panes
        gallery_split_frame = ttk.Frame(gallery_frame)
        gallery_split_frame.pack(fill="both", expand=True)
        
        # Left side: Gallery with thumbnails (50% width)
        gallery_left_frame = ttk.Frame(gallery_split_frame)
        gallery_left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Gallery content frame with scrollable area
        gallery_canvas = tk.Canvas(gallery_left_frame, bg="#f0f0f0")
        gallery_scrollbar = ttk.Scrollbar(gallery_left_frame, orient="vertical", command=gallery_canvas.yview)
        self.gallery_content_frame = ttk.Frame(gallery_canvas)
        
        # Configure scrolling
        self.gallery_content_frame.bind(
            "<Configure>",
            lambda e: gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))
        )
        
        gallery_canvas.create_window((0, 0), window=self.gallery_content_frame, anchor="nw")
        gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
        
        # Pack scrollable area
        gallery_canvas.pack(side="left", fill="both", expand=True)
        gallery_scrollbar.pack(side="right", fill="y")
        
        # Right side: Image preview panel (50% width)
        self.preview_frame = ttk.Frame(gallery_split_frame)
        self.preview_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Preview header
        preview_header = ttk.Frame(self.preview_frame)
        preview_header.pack(fill="x", pady=(0, 10))
        
        self.preview_title = tk.StringVar(value="Image Preview")
        ttk.Label(preview_header, textvariable=self.preview_title, 
                 font=("Arial", 11, "bold")).pack(side="left")
        
        # Preview image label with better container setup
        preview_container = ttk.Frame(self.preview_frame)
        preview_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure container to center content
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)
        
        self.preview_label = tk.Label(preview_container, 
                                      text="Select an image from the gallery\nto see a large preview here",
                                      bg="#2b2b2b", fg="white",
                                      font=("Arial", 12),
                                      justify="center",
                                      relief="sunken",
                                      borderwidth=2,
                                      padx=20, pady=20)
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Delete button (below preview)
        delete_frame = ttk.Frame(self.preview_frame)
        delete_frame.pack(fill="x", pady=(10, 0))
        
        self.delete_button = ttk.Button(delete_frame, 
                                       text="Delete Image", 
                                       command=self._delete_selected_image,
                                       state="disabled")
        self.delete_button.pack(side="right")
        
        # Initialize selected image tracking
        self.selected_image_path = None
        
        # Initial gallery content - placeholder
        self.gallery_placeholder_label = tk.Label(self.gallery_content_frame, 
                                                 text="Select a dataset to see preview images",
                                                 font=("Arial", 12),
                                                 fg="#666666")
        self.gallery_placeholder_label.pack(expand=True, pady=50)
        
        # Initialize dataset list
        self._refresh_labeling_datasets()
        
        # Bind dataset selection to gallery update
        self.label_dataset_combo.bind("<<ComboboxSelected>>", self._on_labeling_dataset_selected)
    
    def _setup_training_tab(self):
        """Setup the training tab interface"""
        # Title
        ttk.Label(self.training_frame, text="Model Training", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Two-column layout
        main_training_frame = ttk.Frame(self.training_frame)
        main_training_frame.pack(fill="both", expand=True)
        
        # Left column - Configuration
        left_column = ttk.Frame(main_training_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Dataset selection
        dataset_frame = ttk.LabelFrame(left_column, text="Dataset Selection", padding="15")
        dataset_frame.pack(fill="x", pady=(0, 15))
        
        self.train_dataset_var = tk.StringVar()
        self.train_dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.train_dataset_var, 
                                               state="readonly")
        self.train_dataset_combo.pack(fill="x", pady=(0, 10))
        
        ttk.Button(dataset_frame, text="Refresh Datasets", 
                  command=self._refresh_training_datasets).pack()
        
        # Model configuration
        model_config_frame = ttk.LabelFrame(left_column, text="Model Configuration", padding="15")
        model_config_frame.pack(fill="x", pady=(0, 15))
        
        # Model name
        name_frame = ttk.Frame(model_config_frame)
        name_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(name_frame, text="Model Name:", width=15).pack(side="left")
        self.model_name_var = tk.StringVar(value=f"model_{int(time.time())}")
        ttk.Entry(name_frame, textvariable=self.model_name_var, width=25).pack(side="left", padx=(10, 0))
        
        # Architecture selection
        arch_frame = ttk.Frame(model_config_frame)
        arch_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(arch_frame, text="Architecture:", width=15).pack(side="left")
        self.arch_type_var = tk.StringVar(value="CNN + LSTM")
        arch_combo = ttk.Combobox(arch_frame, textvariable=self.arch_type_var, 
                                 values=list(ARCHITECTURE_TYPES.values()), state="readonly")
        arch_combo.pack(side="left", padx=(10, 0))
        arch_combo.bind("<<ComboboxSelected>>", self._on_training_arch_change)
        
        # Size selection
        size_frame = ttk.Frame(model_config_frame)
        size_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(size_frame, text="Model Size:", width=15).pack(side="left")
        self.arch_size_var = tk.StringVar(value="Medium - Balanced performance and speed")
        size_combo = ttk.Combobox(size_frame, textvariable=self.arch_size_var, 
                                 values=[f"{v['name']} - {v['description']}" for v in ARCHITECTURE_SIZES.values()], 
                                 state="readonly")
        size_combo.pack(side="left", padx=(10, 0))
        size_combo.bind("<<ComboboxSelected>>", self._on_training_arch_change)
        
        # Model statistics
        self.training_model_stats_var = tk.StringVar()
        ttk.Label(model_config_frame, textvariable=self.training_model_stats_var, 
                 foreground="#666666", font=("Arial", 9)).pack(pady=(10, 0))
        
        # Training parameters
        train_params_frame = ttk.LabelFrame(left_column, text="Training Parameters", padding="15")
        train_params_frame.pack(fill="x")
        
        # Epochs
        epochs_frame = ttk.Frame(train_params_frame)
        epochs_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(epochs_frame, text="Epochs:", width=15).pack(side="left")
        self.epochs_var = tk.IntVar(value=DEFAULT_NUM_EPOCHS)
        ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=10).pack(side="left", padx=(10, 0))
        
        # Batch size
        batch_frame = ttk.Frame(train_params_frame)
        batch_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(batch_frame, text="Batch Size:", width=15).pack(side="left")
        self.batch_size_var = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        ttk.Entry(batch_frame, textvariable=self.batch_size_var, width=10).pack(side="left", padx=(10, 0))
        
        # Learning rate
        lr_frame = ttk.Frame(train_params_frame)
        lr_frame.pack(fill="x")
        ttk.Label(lr_frame, text="Learning Rate:", width=15).pack(side="left")
        self.learning_rate_var = tk.DoubleVar(value=DEFAULT_LEARNING_RATE)
        ttk.Entry(lr_frame, textvariable=self.learning_rate_var, width=10).pack(side="left", padx=(10, 0))
        
        # Right column - Progress and controls
        right_column = ttk.Frame(main_training_frame)
        right_column.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Training progress
        progress_frame = ttk.LabelFrame(right_column, text="Training Progress", padding="15")
        progress_frame.pack(fill="x", pady=(0, 15))
        
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(progress_frame, variable=self.training_progress_var, 
                                                    maximum=1.0)
        self.training_progress_bar.pack(fill="x", pady=(0, 10))
        
        self.training_status_var = tk.StringVar(value="Ready to train")
        ttk.Label(progress_frame, textvariable=self.training_status_var).pack(anchor="w")
        
        self.training_stats_var = tk.StringVar()
        ttk.Label(progress_frame, textvariable=self.training_stats_var, 
                 foreground="#666666", font=("Arial", 9)).pack(anchor="w", pady=(5, 0))
        
        # Training controls
        train_controls_frame = ttk.Frame(right_column)
        train_controls_frame.pack(fill="x")
        
        self.start_train_button = ttk.Button(train_controls_frame, text="Start Training", 
                                           command=self._start_training)
        self.start_train_button.pack(pady=10)
        
        # Initialize
        self._refresh_training_datasets()
        self._update_training_model_stats()
        
        # Map display names back to keys
        self._arch_type_map = {v: k for k, v in ARCHITECTURE_TYPES.items()}
        self._arch_size_map = {f"{v['name']} - {v['description']}": k for k, v in ARCHITECTURE_SIZES.items()}
    
    def _setup_testing_tab(self):
        """Setup the testing tab interface"""
        # Title
        ttk.Label(self.testing_frame, text="Model Testing", 
                 font=("Arial", 16, "bold")).pack(pady=(0, 20))
        
        # Two-column layout
        main_testing_frame = ttk.Frame(self.testing_frame)
        main_testing_frame.pack(fill="both", expand=True)
        
        # Left column - Model selection and info
        left_column = ttk.Frame(main_testing_frame)
        left_column.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Model selection
        model_frame = ttk.LabelFrame(left_column, text="Model Selection", padding="15")
        model_frame.pack(fill="x", pady=(0, 15))
        
        self.test_model_var = tk.StringVar()
        self.test_model_combo = ttk.Combobox(model_frame, textvariable=self.test_model_var, 
                                            state="readonly")
        self.test_model_combo.pack(fill="x", pady=(0, 10))
        self.test_model_combo.bind("<<ComboboxSelected>>", self._on_test_model_selected)
        
        model_buttons_frame = ttk.Frame(model_frame)
        model_buttons_frame.pack(fill="x")
        
        ttk.Button(model_buttons_frame, text="Refresh", 
                  command=self._refresh_test_models).pack(side="left")
        ttk.Button(model_buttons_frame, text="Load Model", 
                  command=self._load_test_model).pack(side="left", padx=(10, 0))
        
        # Model information
        info_frame = ttk.LabelFrame(left_column, text="Model Information", padding="15")
        info_frame.pack(fill="both", expand=True)
        
        self.test_info_text = tk.Text(info_frame, height=12, state="disabled", wrap="word")
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.test_info_text.yview)
        self.test_info_text.configure(yscrollcommand=scrollbar.set)
        
        self.test_info_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Right column - Testing controls
        right_column = ttk.Frame(main_testing_frame)
        right_column.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Monitor selection
        monitor_frame = ttk.LabelFrame(right_column, text="Monitor Selection", padding="15")
        monitor_frame.pack(fill="x", pady=(0, 15))
        
        self.test_monitor_var = tk.StringVar()
        self.test_monitor_combo = ttk.Combobox(monitor_frame, textvariable=self.test_monitor_var, 
                                              state="readonly")
        self.test_monitor_combo.pack(fill="x", pady=(0, 10))
        
        ttk.Button(monitor_frame, text="Refresh Monitors", 
                  command=self._refresh_test_monitors).pack()
        
        # Testing controls
        test_controls_frame = ttk.LabelFrame(right_column, text="Live Testing", padding="15")
        test_controls_frame.pack(fill="x", pady=(0, 15))
        
        self.test_status_var = tk.StringVar(value="No model loaded")
        ttk.Label(test_controls_frame, textvariable=self.test_status_var).pack(pady=(0, 10))
        
        test_buttons_frame = ttk.Frame(test_controls_frame)
        test_buttons_frame.pack()
        
        self.start_test_button = ttk.Button(test_buttons_frame, text="Start Live Test", 
                                          command=self._start_live_test, state="disabled")
        self.start_test_button.pack(pady=5)
        
        self.pause_test_button = ttk.Button(test_buttons_frame, text="Pause", 
                                          command=self._pause_live_test, state="disabled")
        self.pause_test_button.pack(pady=5)
        
        self.stop_test_button = ttk.Button(test_buttons_frame, text="Stop", 
                                         command=self._stop_live_test, state="disabled")
        self.stop_test_button.pack(pady=5)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(right_column, text="Instructions", padding="15")
        instructions_frame.pack(fill="both", expand=True)
        
        instructions_text = """1. Select and load a trained model
2. Choose the monitor to capture from
3. Start the live test
4. In the preview window:
   - Red dot: Current mouse position
   - Green dot: Model prediction
   - Press 'P' to pause/unpause mouse movement
   - Press 'Q' to quit the preview
5. Safety features:
   - Move mouse to screen corner to trigger failsafe stop
   - Use Stop button to end testing"""
        
        ttk.Label(instructions_frame, text=instructions_text, justify="left").pack(anchor="w")
        
        # Initialize
        self.live_tester = LiveTester(stop_callback=self._on_test_failsafe_stop)
        self._refresh_test_models()
        self._refresh_test_monitors()

    # Tab change event handler
    def _on_tab_changed(self, event):
        """Handle tab change events"""
        current_tab = self.notebook.select()
        tab_name = self.notebook.tab(current_tab, "text")
        
        # Update status bar
        self.status_var.set(f"Current module: {tab_name}")
        
        # Stop any running preview threads
        if tab_name != "Recording" and self._recording_preview_running:
            self._recording_preview_running = False
            if self._recording_preview_thread:
                self._recording_preview_thread.join(timeout=1.0)
        
        # For each tab, check if we need to initialize anything
        if tab_name == "Recording":
            # Start preview thread if needed
            selected_window = self.window_var.get()
            if selected_window and not self._recording_preview_running:
                self._start_recording_preview()
        elif tab_name == "Labeling":
            # Refresh dataset list
            self._refresh_labeling_datasets()
        elif tab_name == "Training":
            # Refresh dataset list
            self._refresh_training_datasets()
        elif tab_name == "Testing":
            # Refresh model list
            self._refresh_test_models()
    
    # Recording tab methods
    def _refresh_windows(self):
        """Refresh the list of available windows"""
        windows = get_windows_list()
        
        # Create display strings for the dropdown
        display_options = []
        for w in windows:
            if w.title == "Whole Screen":
                display_options.append("Whole Screen (Default)")
            else:
                display_options.append(f"{w.title} - {w.process}")
        
        self.window_combo['values'] = display_options
        self.windows_data = windows  # Store for later reference
        
        if windows and not self.window_var.get():
            self.window_combo.current(0)
            self._on_recording_window_selected(None)
        
        # Update status
        self.status_var.set(f"Found {len(windows)} windows")
    
    def _on_recording_window_selected(self, event):
        """Handle window selection in recording tab"""
        selected_window = self.window_var.get()
        
        if selected_window:
            # Start preview if not already running
            if not self._recording_preview_running:
                self._start_recording_preview()
            
            # Auto-generate session name if empty
            if not self.session_name_var.get():
                window_name = selected_window.split(' - ')[0].lower().replace(' ', '_')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.session_name_var.set(f"{window_name}_{timestamp}")
    
    def _start_recording_preview(self):
        """Start preview thread for recording tab"""
        if self._recording_preview_running:
            return
        
        self._recording_preview_running = True
        self._recording_preview_thread = threading.Thread(
            target=self._recording_preview_loop,
            daemon=True
        )
        self._recording_preview_thread.start()
    
    def _recording_preview_loop(self):
        """Preview loop for recording tab"""
        from recorder import find_window_handle, capture_window_frame
        
        while self._recording_preview_running:
            try:
                # Get the selected window
                selected_window = self.window_var.get()
                if not selected_window:
                    time.sleep(0.2)
                    continue
                
                # Find window coordinates from cached data
                selected_index = self.window_combo.current()
                if selected_index < 0 or selected_index >= len(getattr(self, 'windows_data', [])):
                    time.sleep(0.2)
                    continue
                
                target_window = self.windows_data[selected_index]
                window_title = target_window.title
                
                # Use proper window capture (same as recorder)
                hwnd = find_window_handle(window_title)
                img_pil = capture_window_frame(hwnd)
                
                if img_pil:
                    # Resize for preview to fit in preview container
                    max_width = 480
                    max_height = 270
                    
                    # Calculate aspect ratio preserving resize
                    width_ratio = max_width / img_pil.width
                    height_ratio = max_height / img_pil.height
                    resize_ratio = min(width_ratio, height_ratio)
                    
                    new_width = int(img_pil.width * resize_ratio)
                    new_height = int(img_pil.height * resize_ratio)
                    
                    img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Convert to Tkinter PhotoImage
                    img_tk = ImageTk.PhotoImage(img_pil)
                    
                    # Update label in main thread with proper reference handling
                    self.after_idle(lambda: self._update_preview_image(img_tk))
                else:
                    # Show error message if capture fails
                    self.after_idle(lambda: self._show_preview_error(f"Cannot capture: {window_title}"))
                
                # Sleep to limit updates (2 FPS for preview)
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Preview error: {e}")
                self.after_idle(lambda: self._show_preview_error(f"Preview error: {str(e)[:40]}"))
                time.sleep(1.0)
    
    def _update_preview_image(self, img_tk):
        """Update preview image in the recording tab"""
        try:
            if hasattr(self, 'recording_preview_label') and self.recording_preview_label.winfo_exists():
                self.recording_preview_label.configure(
                    image=img_tk, 
                    text="", 
                    compound="center",
                    relief="sunken",
                    borderwidth=1
                )
                # Store reference to prevent garbage collection
                self.recording_preview_label._image_ref = img_tk
        except Exception as e:
            print(f"Error updating preview image: {e}")
    
    def _show_preview_error(self, error_message):
        """Show error message in preview area"""
        try:
            if hasattr(self, 'recording_preview_label') and self.recording_preview_label.winfo_exists():
                self.recording_preview_label.configure(
                    image="", 
                    text=error_message, 
                    fg="#ff6666",
                    relief="flat",
                    compound="center"
                )
                # Clear any stored image reference
                if hasattr(self.recording_preview_label, '_image_ref'):
                    delattr(self.recording_preview_label, '_image_ref')
        except Exception as e:
            print(f"Error showing preview error: {e}")
    
    def _start_recording(self):
        """Start recording session"""
        # Validate inputs
        session_name = self.session_name_var.get()
        if not session_name:
            messagebox.showerror("Error", "Please enter a session name")
            return
        
        selected_window = self.window_var.get()
        if not selected_window:
            messagebox.showerror("Error", "Please select a window to record")
            return
        
        fps = self.fps_var.get()
        if fps < 1 or fps > 60:
            messagebox.showerror("Error", "FPS must be between 1 and 60")
            return
        
        # Find window coordinates from cached data
        selected_index = self.window_combo.current()
        if selected_index < 0 or selected_index >= len(getattr(self, 'windows_data', [])):
            messagebox.showerror("Error", "Selected window not found")
            return
        target_window = self.windows_data[selected_index]
        
        # Use the window title for the recorder
        window_title = target_window.title
        
        try:
            # Initialize recorder with window title (matching original Recorder interface)
            self.recorder_instance = Recorder(session_name, window_title, fps)
            self.recorder_instance.start()
            
            # Update UI
            self.recording_status_var.set("Recording...")
            self.start_rec_button.configure(state="disabled")
            self.pause_rec_button.configure(state="normal")
            self.stop_rec_button.configure(state="normal")
            self.status_var.set(f"Recording '{session_name}' at {fps} FPS")
            
        except Exception as e:
            messagebox.showerror("Recording Error", f"Failed to start recording: {str(e)}")
            print(f"Error starting recording: {e}")
    
    def _pause_recording(self):
        """Pause or resume recording session"""
        if not self.recorder_instance:
            return
        
        if self.recorder_instance.paused:
            # Resume
            self.recorder_instance.resume()
            self.recording_status_var.set("Recording...")
            self.pause_rec_button.configure(text="Pause")
            self.status_var.set("Recording resumed")
        else:
            # Pause
            self.recorder_instance.pause()
            self.recording_status_var.set("Paused")
            self.pause_rec_button.configure(text="Resume")
            self.status_var.set("Recording paused")
    
    def _stop_recording(self):
        """Stop recording session"""
        if not self.recorder_instance:
            return
        
        try:
            # Stop recording
            self.recorder_instance.stop()
            
            # Show stats
            frames_captured = self.recorder_instance.frame_counter
            session_name = self.recorder_instance.dataset.session_name
            
            # Reset state
            self.recorder_instance = None
            
            # Update UI
            self.recording_status_var.set("Ready to record")
            self.start_rec_button.configure(state="normal")
            self.pause_rec_button.configure(state="disabled", text="Pause")
            self.stop_rec_button.configure(state="disabled")
            
            # Show completion message
            messagebox.showinfo("Recording Complete", 
                               f"Recording '{session_name}' completed\n"
                               f"Captured {frames_captured} frames")
            
            # Ask if user wants to label the dataset
            if messagebox.askyesno("Label Dataset", 
                                  f"Do you want to label the dataset '{session_name}' now?"):
                # Switch to labeling tab
                self.notebook.select(self.labeling_frame)
                self.label_dataset_var.set(session_name)
                self._start_labeling()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")
            print(f"Error stopping recording: {e}")
    
    # Labeling tab methods
    def _refresh_labeling_datasets(self):
        """Refresh list of available datasets for labeling"""
        try:
            # Get datasets from datasets directory
            datasets = []
            datasets_path = os.path.abspath("datasets")
            if os.path.exists(datasets_path):
                for item in os.listdir(datasets_path):
                    item_path = os.path.join(datasets_path, item)
                    if os.path.isdir(item_path):
                        # Check if it has raw_images folder (indicating it's a valid dataset)
                        raw_images_path = os.path.join(item_path, "raw_images")
                        if os.path.exists(raw_images_path):
                            datasets.append(item)
            
            self.label_dataset_combo['values'] = datasets
            
            # Update status
            self.status_var.set(f"Found {len(datasets)} datasets")
            
            # Select first dataset if none selected
            if datasets and not self.label_dataset_var.get():
                self.label_dataset_combo.current(0)
                self._update_dataset_gallery()  # Update gallery for auto-selected dataset
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get datasets: {str(e)}")
    
    def _on_labeling_dataset_selected(self, event):
        """Handle dataset selection in labeling tab - update gallery"""
        self._update_dataset_gallery()
    
    def _update_dataset_gallery(self):
        """Update the gallery preview with images from selected dataset"""
        dataset_name = self.label_dataset_var.get()
        
        # Clear current gallery content
        for widget in self.gallery_content_frame.winfo_children():
            widget.destroy()
        
        # Reset preview panel
        self._reset_preview_panel()
        
        if not dataset_name:
            # Show placeholder
            placeholder = tk.Label(self.gallery_content_frame, 
                                 text="Select a dataset to see preview images",
                                 font=("Arial", 12),
                                 fg="#666666")
            placeholder.pack(expand=True, pady=50)
            return
        
        # Get dataset path
        dataset_path = os.path.join(os.path.abspath("datasets"), dataset_name)
        raw_images_path = os.path.join(dataset_path, "raw_images")
        
        if not os.path.exists(raw_images_path):
            # Show error message
            error_label = tk.Label(self.gallery_content_frame, 
                                 text=f"Dataset '{dataset_name}' not found or has no images",
                                 font=("Arial", 12),
                                 fg="#cc0000")
            error_label.pack(expand=True, pady=50)
            return
        
        try:
            # Get list of image files
            image_files = [f for f in os.listdir(raw_images_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                # Show no images message
                no_images_label = tk.Label(self.gallery_content_frame, 
                                          text=f"No images found in dataset '{dataset_name}'",
                                          font=("Arial", 12),
                                          fg="#666666")
                no_images_label.pack(expand=True, pady=50)
                return
            
            # Sort files and limit to first 20 for performance
            image_files.sort()
            display_images = image_files[:20]
            
            # Create header with info
            info_text = f"Dataset: {dataset_name} ({len(image_files)} images" + \
                       (f" - showing first {len(display_images)}" if len(image_files) > 20 else "")
            
            info_label = tk.Label(self.gallery_content_frame, 
                                text=info_text,
                                font=("Arial", 11, "bold"),
                                fg="#333333")
            info_label.pack(pady=(10, 20))
            
            # Create a responsive gallery grid that fills the full width
            # Use 4 images per row for optimal space utilization
            images_per_row = 4  # Maximize gallery width utilization
            
            for i in range(0, len(display_images), images_per_row):
                row_images = display_images[i:i + images_per_row]
                
                # Create row frame with uniform configuration
                row_frame = ttk.Frame(self.gallery_content_frame)
                row_frame.pack(fill="x", pady=2)
                
                # Configure row to distribute space evenly across all columns
                for col in range(images_per_row):
                    row_frame.columnconfigure(col, weight=1, uniform="gallery_cols")
                
                # Add images to this row
                for col, image_file in enumerate(row_images):
                    # Create thumbnail container with grid layout for better control
                    thumb_container = ttk.Frame(row_frame)
                    thumb_container.grid(row=0, column=col, sticky="nsew", padx=3, pady=3)
                    thumb_container.grid_propagate(False)  # Don't let content dictate size
                    
                    # Load and display thumbnail
                    self._add_gallery_thumbnail_grid(thumb_container, raw_images_path, image_file)
            
            # Add status info
            if len(image_files) > 20:
                more_label = tk.Label(self.gallery_content_frame, 
                                    text=f"... and {len(image_files) - 20} more images",
                                    font=("Arial", 10),
                                    fg="#666666")
                more_label.pack(pady=(20, 10))
                
        except Exception as e:
            # Show error message
            error_label = tk.Label(self.gallery_content_frame, 
                                 text=f"Error loading gallery: {str(e)}",
                                 font=("Arial", 12),
                                 fg="#cc0000")
            error_label.pack(expand=True, pady=50)
            print(f"Gallery error: {e}")
    
    def _add_gallery_thumbnail(self, parent_frame, images_path, image_file):
        """Add a thumbnail to the gallery"""
        try:
            # Create frame for this thumbnail with better space utilization
            thumb_frame = ttk.Frame(parent_frame)
            thumb_frame.pack(side="left", padx=8, pady=8, fill="both", expand=True)
            
            # Load image
            image_path = os.path.join(images_path, image_file)
            with Image.open(image_path) as img:
                # Create larger thumbnail (220x150 max size for better visibility)
                display_size = (220, 150)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Create label with image with better styling
                image_label = tk.Label(thumb_frame, 
                                     image=photo, 
                                     relief="ridge",
                                     borderwidth=2,
                                     bg="white",
                                     cursor="hand2",
                                     padx=5,
                                     pady=5)
                image_label.pack(fill="both", expand=True)
                
                # Store reference to prevent garbage collection
                image_label._image_ref = photo
                
                # Add click handler for selection
                image_label.bind("<Button-1>", 
                                lambda e, path=image_path, file=image_file: self._select_image(path, file))
                
                # Add filename label with better formatting
                filename_display = image_file[:25] + ("..." if len(image_file) > 25 else "")
                name_label = tk.Label(thumb_frame, 
                                    text=filename_display,
                                    font=("Arial", 9, "bold"),
                                    fg="#444444",
                                    cursor="hand2",
                                    bg="#f8f8f8",
                                    relief="flat",
                                    pady=3)
                name_label.pack(fill="x", pady=(3, 0))
                
                # Add click handler to filename too
                name_label.bind("<Button-1>", 
                               lambda e, path=image_path, file=image_file: self._select_image(path, file))
                
        except Exception as e:
            # If image fails to load, show placeholder
            placeholder_label = tk.Label(parent_frame, 
                                        text="[Image]\nError",
                                        width=18,
                                        height=6,
                                        relief="sunken",
                                        bg="#f0f0f0",
                                        fg="#999999",
                                        font=("Arial", 8))
            placeholder_label.pack(side="left", padx=10, pady=5)
            print(f"Thumbnail error for {image_file}: {e}")
    
    def _add_gallery_thumbnail_grid(self, parent_frame, images_path, image_file):
        """Add a thumbnail to the gallery using grid layout for better space utilization"""
        try:
            # Load image
            image_path = os.path.join(images_path, image_file)
            with Image.open(image_path) as img:
                # Optimize thumbnail size for 4-column layout (smaller but still clear)
                display_size = (140, 105)
                img.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Create container frame for the thumbnail
                thumb_inner = tk.Frame(parent_frame, bg="white", relief="ridge", bd=1)
                thumb_inner.pack(fill="both", expand=True, padx=2, pady=2)
                
                # Create label with image
                image_label = tk.Label(thumb_inner, 
                                     image=photo, 
                                     bg="white",
                                     cursor="hand2")
                image_label.pack(pady=3)
                
                # Store reference to prevent garbage collection
                image_label._image_ref = photo
                
                # Add click handler for selection
                image_label.bind("<Button-1>", 
                                lambda e, path=image_path, file=image_file: self._select_image(path, file))
                
                # Add filename label with compact formatting
                filename_display = image_file[:18] + ("..." if len(image_file) > 18 else "")
                name_label = tk.Label(thumb_inner, 
                                    text=filename_display,
                                    font=("Arial", 7),
                                    fg="#444444",
                                    cursor="hand2",
                                    bg="#f8f8f8",
                                    relief="flat")
                name_label.pack(fill="x", pady=(0, 2))
                
                # Add click handler to filename and container
                name_label.bind("<Button-1>", 
                               lambda e, path=image_path, file=image_file: self._select_image(path, file))
                thumb_inner.bind("<Button-1>", 
                               lambda e, path=image_path, file=image_file: self._select_image(path, file))
                
        except Exception as e:
            # If image fails to load, show placeholder
            placeholder_frame = tk.Frame(parent_frame, bg="#f0f0f0", relief="ridge", bd=1)
            placeholder_frame.pack(fill="both", expand=True, padx=2, pady=2)
            
            placeholder_label = tk.Label(placeholder_frame, 
                                        text="[Image]\nLoad Error",
                                        width=16,
                                        height=8,
                                        relief="flat",
                                        bg="#f0f0f0",
                                        fg="#999999",
                                        font=("Arial", 8),
                                        cursor="hand2")
            placeholder_label.pack(expand=True, fill="both")
            print(f"Thumbnail error for {image_file}: {e}")
    
    def _reset_preview_panel(self):
        """Reset the preview panel to default state"""
        self.selected_image_path = None
        self.preview_title.set("Image Preview")
        self.preview_label.configure(
            text="Select an image from the gallery\nto see a large preview here",
            image="",
            bg="#2b2b2b",
            fg="white",
            justify="center",
            padx=20,
            pady=20
        )
        # Clear image reference
        if hasattr(self.preview_label, '_image_ref'):
            del self.preview_label._image_ref
        self.delete_button.configure(state="disabled")
    
    def _select_image(self, image_path, image_file):
        """Handle image selection from gallery"""
        try:
            self.selected_image_path = image_path
            self.preview_title.set(f"Preview: {image_file}")
            
            # Load and display the image in preview
            with Image.open(image_path) as img:
                # Calculate size to fit in preview area - use larger size for better visibility
                preview_size = (450, 300)
                
                # Create a copy and resize maintaining aspect ratio
                img_copy = img.copy()
                img_copy.thumbnail(preview_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                preview_photo = ImageTk.PhotoImage(img_copy)
                
                # Update preview label with better styling
                self.preview_label.configure(
                    image=preview_photo,
                    text="",
                    bg="white",
                    relief="ridge",
                    borderwidth=3,
                    padx=5,
                    pady=5
                )
                
                # Store reference to prevent garbage collection
                self.preview_label._image_ref = preview_photo
                
            # Enable delete button
            self.delete_button.configure(state="normal")
            
        except Exception as e:
            print(f"Error selecting image {image_file}: {e}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _delete_selected_image(self):
        """Delete the currently selected image permanently"""
        if not self.selected_image_path or not os.path.exists(self.selected_image_path):
            return
        
        # Get filename for confirmation
        image_file = os.path.basename(self.selected_image_path)
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to permanently delete '{image_file}'?\n\n"
                              "This action cannot be undone."):
            try:
                # Delete the file
                os.remove(self.selected_image_path)
                
                # Reset preview panel
                self._reset_preview_panel()
                
                # Refresh gallery to update the display
                self._update_dataset_gallery()
                
                messagebox.showinfo("Success", f"Image '{image_file}' has been deleted.")
                
            except Exception as e:
                print(f"Error deleting image: {e}")
                messagebox.showerror("Error", f"Failed to delete image: {str(e)}")
    
    def _start_labeling(self):
        """Start labeling session"""
        dataset_name = self.label_dataset_var.get()
        if not dataset_name:
            messagebox.showerror("Error", "Please select a dataset to label")
            return
        
        # Get the full session path
        session_path = os.path.join(os.path.abspath("datasets"), dataset_name)
        if not os.path.exists(session_path):
            messagebox.showerror("Error", f"Dataset '{dataset_name}' not found")
            return
        
        try:
            # Create labeler instance (it will open its own window)
            self.labeler_instance = LabelerApp(self, session_path, on_complete_callback=self._on_labeling_complete)
            
            # Update status
            self.status_var.set(f"Labeling dataset: {dataset_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start labeling: {str(e)}")
    
    def _on_labeling_complete(self):
        """Handle labeling completion"""
        # Reset labeler
        self.labeler_instance = None
        
        # Refresh the gallery to show updated dataset
        self._update_dataset_gallery()
        
        # Update status
        self.status_var.set("Labeling complete")
        
        # Ask if user wants to train a model
        dataset_name = self.label_dataset_var.get()
        if messagebox.askyesno("Train Model", 
                              f"Do you want to train a model using dataset '{dataset_name}' now?"):
            # Switch to training tab
            self.notebook.select(self.training_frame)
            self.train_dataset_var.set(dataset_name)
    
    # Training tab methods
    def _refresh_training_datasets(self):
        """Refresh list of available datasets for training"""
        try:
            # Get datasets that have labels (check for labels.csv)
            datasets = []
            datasets_path = os.path.abspath("datasets")
            if os.path.exists(datasets_path):
                for item in os.listdir(datasets_path):
                    item_path = os.path.join(datasets_path, item)
                    if os.path.isdir(item_path):
                        # Check if it has labels.csv (indicating it's labeled)
                        labels_path = os.path.join(item_path, "labels.csv")
                        if os.path.exists(labels_path):
                            datasets.append(item)
            
            self.train_dataset_combo['values'] = datasets
            
            # Update status
            self.status_var.set(f"Found {len(datasets)} labeled datasets")
            
            # Select first dataset if none selected
            if datasets and not self.train_dataset_var.get():
                self.train_dataset_combo.current(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get datasets: {str(e)}")
    
    def _on_training_arch_change(self, event):
        """Handle architecture or size change in training tab"""
        self._update_training_model_stats()
    
    def _update_training_model_stats(self):
        """Update model statistics display in training tab"""
        try:
            arch_type_display = self.arch_type_var.get()
            arch_size_display = self.arch_size_var.get()
            
            # Map display names back to keys
            arch_type = self._arch_type_map.get(arch_type_display)
            arch_size = self._arch_size_map.get(arch_size_display)
            
            if not arch_type or not arch_size:
                return
            
            # Get estimated parameters using new dynamic sizing
            params = estimate_model_parameters(arch_type, DEFAULT_INPUT_RESOLUTION, arch_size)
            
            # Update stats display
            stats_text = (f"Estimated model size: {params['total_parameters']:,} parameters\n"
                         f"Input resolution: {DEFAULT_INPUT_RESOLUTION[0]}×{DEFAULT_INPUT_RESOLUTION[1]}×3\n"
                         f"Memory usage: ~{params['model_size_mb']:.1f} MB\n"
                         f"Training speed: {params['relative_speed']:.2f}x relative")
            
            self.training_model_stats_var.set(stats_text)
        except Exception as e:
            print(f"Error updating model stats: {e}")
    
    def _start_training(self):
        """Start model training process"""
        # Validate inputs
        dataset_name = self.train_dataset_var.get()
        if not dataset_name:
            messagebox.showerror("Error", "Please select a dataset to train on")
            return
        
        model_name = self.model_name_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return
        
        # Map display names back to keys
        arch_type_display = self.arch_type_var.get()
        arch_size_display = self.arch_size_var.get()
        
        arch_type = self._arch_type_map.get(arch_type_display)
        arch_size = self._arch_size_map.get(arch_size_display)
        
        if not arch_type or not arch_size:
            messagebox.showerror("Error", "Invalid architecture selection")
            return
        
        # Get training parameters
        try:
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            learning_rate = self.learning_rate_var.get()
        except:
            messagebox.showerror("Error", "Please enter valid numeric values for training parameters")
            return
        
        # Initialize trainer
        self.model_trainer = ModelTrainer(progress_callback=self._on_training_progress)
        
        # Create training configuration
        self.training_config = {
            "architecture_type": arch_type,
            "architecture_size": arch_size,
            "input_resolution": DEFAULT_INPUT_RESOLUTION,
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        
        self.training_model_name = model_name
        self.training_dataset_name = dataset_name
        
        # Update UI
        self.start_train_button.configure(state="disabled")
        self.training_status_var.set("Initializing training...")
        self.training_progress_var.set(0.0)
        self.training_stats_var.set("")
        self.status_var.set(f"Training model '{model_name}' on dataset '{dataset_name}'")
        
        # Start training in a separate thread
        threading.Thread(target=self._training_thread, daemon=True).start()
    
    def _training_thread(self):
        """Background thread for model training"""
        try:
            # Get session path
            session_path = os.path.join(os.path.abspath("datasets"), self.training_dataset_name)
            
            # Start training
            success = self.model_trainer.train_model(session_path, self.training_model_name, self.training_config)
            
            # Handle completion
            self.after(0, lambda: self._on_training_complete(success=success))
        except Exception as e:
            # Handle errors
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.after(0, lambda: self._on_training_complete(success=False))
    
    def _on_training_progress(self, message, progress):
        """Handle training progress updates"""
        self.training_progress_var.set(progress)
        self.training_status_var.set(message)
        
        # Extract additional info from message if available
        if "Loss:" in message:
            self.training_stats_var.set(message)
        else:
            self.training_stats_var.set("")
    
    def _on_training_complete(self, success=True):
        """Handle training completion"""
        if success:
            model_name = self.model_name_var.get()
            
            # Refresh model metadata to ensure new model is available
            self.model_metadata.refresh_metadata()
            
            # Show single unified popup with success message and test option
            response = messagebox.askyesnocancel(
                "Training Complete", 
                f"Model '{model_name}' trained successfully!\n\n"
                f"Would you like to test this model now?\n\n"
                f"• Yes: Switch to Testing tab and load the model\n"
                f"• No: Stay on Training tab\n"
                f"• Cancel: Close this dialog",
                icon="question"
            )
            
            if response:  # Yes - switch to testing
                # Switch to testing tab
                self.notebook.select(self.testing_frame)
                self._refresh_test_models()
                self.test_model_var.set(model_name)
                self._load_test_model()
            # No or Cancel - just stay where we are
            
        else:
            self.training_status_var.set("Training failed")
            messagebox.showerror("Training Failed", 
                               "Model training failed. Please check the console for error details.")
        
        # Reset UI
        self.start_train_button.configure(state="normal")
        self.model_trainer = None
    
    # Testing tab methods
    def _refresh_test_models(self):
        """Refresh list of available trained models"""
        models = list(self.model_metadata.get_all_models().keys())
        self.test_model_combo['values'] = models
        
        # Update status
        self.status_var.set(f"Found {len(models)} trained models")
    
    def _refresh_test_monitors(self):
        """Refresh list of available monitors for testing"""
        with mss.mss() as sct:
            monitors = [f"Monitor {i+1}: {m['width']}x{m['height']}" 
                       for i, m in enumerate(sct.monitors[1:])]  # Skip the "all monitors" entry
            
            self.test_monitor_combo['values'] = monitors
            
            # Select first monitor if none selected
            if monitors and not self.test_monitor_var.get():
                self.test_monitor_combo.current(0)
    
    def _on_test_model_selected(self, event):
        """Handle model selection in testing tab"""
        model_name = self.test_model_var.get()
        if model_name:
            # Display model info
            model_info = self.model_metadata.get_model_info(model_name)
            if model_info:
                # Enable text widget for editing
                self.test_info_text.configure(state="normal")
                # Clear and insert new info
                self.test_info_text.delete(1.0, tk.END)
                
                # Extract model configuration and training info
                model_config = model_info.get('model_config', {})
                training_info = model_info.get('training_info', {})
                
                # Get architecture display names
                arch_type = model_config.get('architecture_type', 'Unknown')
                arch_type_display = ARCHITECTURE_TYPES.get(arch_type, arch_type)
                
                arch_size = model_config.get('architecture_size', 'Unknown')
                arch_size_display = ARCHITECTURE_SIZES.get(arch_size, {}).get('name', arch_size.title())
                
                # Get input resolution
                input_res = model_config.get('input_resolution', [0, 0])
                input_width = input_res[0] if len(input_res) > 0 else 0
                input_height = input_res[1] if len(input_res) > 1 else 0
                
                # Extract dataset name from session path
                session_path = model_info.get('session_path', 'Unknown')
                dataset_name = os.path.basename(session_path) if session_path != 'Unknown' else 'Unknown'
                
                # Format creation date
                created_date = model_info.get('created_date', 'Unknown')
                if created_date != 'Unknown':
                    try:
                        from datetime import datetime
                        date_obj = datetime.fromisoformat(created_date.replace('T', ' ').split('.')[0])
                        created_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass  # Keep original if parsing fails
                
                # Training time in a readable format
                training_time = training_info.get('training_time_minutes', 0)
                if training_time >= 60:
                    hours = int(training_time // 60)
                    minutes = int(training_time % 60)
                    time_str = f"{hours}h {minutes}m"
                else:
                    time_str = f"{training_time:.1f} minutes"
                
                info_str = (f"Model: {model_name}\n"
                           f"Architecture: {arch_type_display}\n"
                           f"Size: {arch_size_display}\n"
                           f"Input Resolution: {input_width}×{input_height}\n"
                           f"Training Dataset: {dataset_name}\n"
                           f"Dataset Size: {training_info.get('dataset_size', 0):,} samples\n"
                           f"Date Created: {created_date}\n\n"
                           f"Training Configuration:\n"
                           f"  Epochs: {model_config.get('num_epochs', 0)}\n"
                           f"  Batch Size: {model_config.get('batch_size', 0)}\n"
                           f"  Learning Rate: {model_config.get('learning_rate', 0)}\n\n"
                           f"Training Results:\n"
                           f"  Final Loss: {training_info.get('final_loss', 0):.6f}\n"
                           f"  Training Time: {time_str}\n"
                           f"  Model File: {os.path.basename(model_info.get('file_path', 'Unknown'))}")
                
                self.test_info_text.insert(tk.END, info_str)
                # Disable editing again
                self.test_info_text.configure(state="disabled")
            else:
                # Enable text widget for editing
                self.test_info_text.configure(state="normal")
                # Clear and show error message
                self.test_info_text.delete(1.0, tk.END)
                self.test_info_text.insert(tk.END, f"No information available for model: {model_name}")
                # Disable editing again
                self.test_info_text.configure(state="disabled")
    
    def _load_test_model(self):
        """Load selected model for testing"""
        model_name = self.test_model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please select a model to load")
            return
        
        # Update UI
        self.test_status_var.set("Loading model...")
        self.start_test_button.configure(state="disabled")
        
        # Load model in background thread
        threading.Thread(target=self._load_model_thread, args=(model_name,), daemon=True).start()
    
    def _load_model_thread(self, model_name):
        """Background thread for model loading"""
        try:
            # Load the model and check return value
            success = self.live_tester.load_model(model_name)
            
            # Update UI in main thread
            self.after(0, lambda: self._on_model_loaded(success))
        except Exception as e:
            # Handle errors in main thread
            print(f"Model loading thread error: {e}")
            self.after(0, lambda: messagebox.showerror("Model Loading Error", str(e)))
            self.after(0, lambda: self._on_model_loaded(False))
    
    def _on_model_loaded(self, success):
        """Handle model loading completion"""
        if success:
            self.test_status_var.set("Model loaded successfully")
            self.start_test_button.configure(state="normal")
        else:
            self.test_status_var.set("Failed to load model")
    
    def _start_live_test(self):
        """Start live testing with loaded model"""
        model_name = self.test_model_var.get()
        monitor_str = self.test_monitor_var.get()
        
        if not model_name:
            messagebox.showerror("Error", "No model loaded")
            return
        
        if not monitor_str:
            messagebox.showerror("Error", "Please select a monitor")
            return
        
        # Parse monitor index from string (format: "Monitor X: WIDTHxHEIGHT")
        # Note: Monitor display starts at 1 but monitors array includes 'all monitors' at index 0
        try:
            display_index = int(monitor_str.split(":")[0].replace("Monitor ", ""))
            monitor_index = display_index  # Display index maps directly to monitors[index]
        except:
            messagebox.showerror("Error", "Invalid monitor selection")
            return
        
        # Update UI
        self.start_test_button.configure(state="disabled")
        self.pause_test_button.configure(state="normal")
        self.stop_test_button.configure(state="normal")
        self.test_status_var.set("Testing in progress...")
        
        # Start testing
        success = self.live_tester.start_live_test(monitor_index)
        if not success:
            messagebox.showerror("Error", "Failed to start live testing")
            self._on_testing_complete()
    
    @property
    def is_testing(self):
        """Check if live tester is currently running"""
        return self.live_tester and self.live_tester.running
    
    @property
    def is_paused(self):
        """Check if live tester is currently paused"""
        return self.live_tester and self.live_tester.paused
    
    def _pause_live_test(self):
        """Pause or resume live testing"""
        if not self.is_testing:
            return
        
        if self.is_paused:
            # Resume testing
            self.live_tester.paused = False
            self.test_status_var.set("Testing resumed")
            self.pause_test_button.configure(text="Pause")
        else:
            # Pause testing
            self.live_tester.paused = True
            self.test_status_var.set("Testing paused")
            self.pause_test_button.configure(text="Resume")
    
    def _stop_live_test(self):
        """Stop live testing"""
        if not self.is_testing:
            return
        
        # Stop testing
        self.live_tester.stop_live_test()
        self._on_testing_complete()
    
    def _on_test_failsafe_stop(self):
        """Handle failsafe trigger during testing"""
        # Update UI in main thread
        self.after(0, lambda: messagebox.showwarning(
            "Testing Stopped", 
            "Failsafe triggered: Mouse moved to corner of screen"
        ))
        self.after(0, self._on_testing_complete)
    
    def _on_testing_complete(self):
        """Handle testing completion"""
        # Update UI
        self.start_test_button.configure(state="normal")
        self.pause_test_button.configure(state="disabled", text="Pause")
        self.stop_test_button.configure(state="disabled")
        self.test_status_var.set("Ready to test")
    
    # Application closing handler
    def _on_closing(self):
        """Handle application closing"""
        # Stop any running threads and processes
        self._recording_preview_running = False
        
        if self.recorder_instance and self.recorder_instance.is_recording:
            if messagebox.askyesno("Exit", "Recording in progress. Stop and exit?"):
                self.recorder_instance.stop_recording()
            else:
                return
        
        if self.model_trainer and hasattr(self.model_trainer, 'is_training') and self.model_trainer.is_training:
            if messagebox.askyesno("Exit", "Training in progress. Stop and exit?"):
                # No clean way to stop training, just proceed with exit
                pass
            else:
                return
        
        if self.is_testing:
            self.live_tester.stop_live_test()
        
        # Wait for threads to finish
        if self._recording_preview_thread:
            self._recording_preview_thread.join(timeout=1.0)
        
        # Destroy the window
        self.destroy()

# Main entry point
if __name__ == "__main__":
    app = IntegratedApp()
    app.mainloop()
