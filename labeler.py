# labeler.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from typing import Optional
import threading
import queue
import shutil

from dataset import Dataset

class LabelerApp(tk.Toplevel):
    """
    A Tkinter GUI application for labeling captured frames.
    """
    def __init__(self, parent, session_path: str, on_complete_callback=None):
        super().__init__(parent)
        self.withdraw() # Start with the main window hidden completely.
        self.on_complete_callback = on_complete_callback

        self.session_path = session_path
        self.dataset = Dataset(os.path.basename(session_path))
        
        self.raw_images = sorted([f for f in os.listdir(self.dataset.raw_images_path) if f.lower().endswith('.jpg')])
        if not self.raw_images:
            print("Error: No raw images found in this session.")
            self.after(10, self.destroy)
            return

        # --- NEW ATTRIBUTES for managing display scaling ---
        self.original_img: Optional[Image.Image] = None # Holds the original small image
        self.display_photo_size = (1, 1) # (width, height), avoids division by zero
        self.image_offsets = (0, 0) # (x, y) offsets for centering the image

        self.processed_images = []
        self.current_frame_index = 0
        self.downscale_factor = 1.0
        self.undo_stack = []

        self._prompt_for_downscale()

    def _prompt_for_downscale(self):
        dialog = tk.Toplevel(self)
        dialog.title("Set Processing Resolution")
        dialog.geometry("800x700")
        dialog.resizable(True, True)
        
        # Make maximized window
        try:
            dialog.state('zoomed')  # Windows maximized
        except:
            # Fallback for non-Windows systems
            dialog.geometry("1000x800")
        
        # Main frame with padding (following pattern from other screens)
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        ttk.Label(main_frame, text="Set Processing Resolution", font=("Arial", 16, "bold")).pack(pady=(0, 15))
        
        # Instructions frame
        instructions_frame = ttk.LabelFrame(main_frame, text="Information", padding="10")
        instructions_frame.pack(fill="x", pady=(0, 15))
        
        instructions = "Configure the processing scale for your images. Lower values result in smaller file sizes and faster processing."
        ttk.Label(instructions_frame, text=instructions, wraplength=700, justify="left").pack()
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Scale Configuration", padding="10")
        config_frame.pack(fill="x", pady=(0, 15))
        
        # Scale input frame
        scale_input_frame = ttk.Frame(config_frame)
        scale_input_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(scale_input_frame, text="Scale Factor:", width=15).pack(side="left")
        
        # Numeric input with validation
        vcmd = (dialog.register(self._validate_scale), '%P')
        self.scale_var = tk.StringVar(value="1.0")
        scale_entry = ttk.Entry(scale_input_frame, textvariable=self.scale_var, width=10, 
                               validate='key', validatecommand=vcmd, justify='center')
        scale_entry.pack(side="left", padx=(5, 15))
        scale_entry.bind('<KeyRelease>', lambda e: self._update_preview_numeric(dialog, preview_label, size_label))
        
        ttk.Label(scale_input_frame, text="(0.05 to 2.0)").pack(side="left")
        
        # Preset buttons frame
        presets_frame = ttk.Frame(config_frame)
        presets_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Label(presets_frame, text="Quick presets:", width=15).pack(side="left")
        
        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(side="left", padx=(5, 0))
        
        presets = [("25%", "0.25"), ("50%", "0.5"), ("75%", "0.75"), ("100%", "1.0")]
        for label, value in presets:
            btn = ttk.Button(preset_buttons_frame, text=label, width=8,
                           command=lambda v=value: self._set_preset(v, dialog, preview_label, size_label))
            btn.pack(side="left", padx=2)
        
        # Preview frame (simplified layout for better image display)
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Create preview label with proper settings for image display
        preview_label = tk.Label(preview_frame, 
                                text="Loading preview...", 
                                anchor="center", 
                                relief="sunken", 
                                background="#f0f0f0",
                                width=60,
                                height=20)
        preview_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Size information frame
        info_frame = ttk.LabelFrame(main_frame, text="Size Information", padding="10")
        info_frame.pack(fill="x", pady=(0, 15))
        
        size_label = ttk.Label(info_frame, text="", justify="left")
        size_label.pack(anchor="w")
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill="x")
        
        ttk.Button(buttons_frame, text="Cancel", command=lambda: self._cancel_scale(dialog)).pack(side="right", padx=(10, 0))
        ttk.Button(buttons_frame, text="Confirm and Start Labeling", command=lambda: self._confirm_scale(dialog)).pack(side="right")
        
        # Set up dialog properties
        dialog.bind('<Escape>', lambda e: self._cancel_scale(dialog))
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._cancel_scale(dialog))
        dialog.transient(self.master)
        
        # Auto-focus and initial preview
        dialog.focus_force()
        dialog.lift()
        scale_entry.focus_set()
        
        # Initial preview update
        dialog.after(100, lambda: self._update_preview_numeric(dialog, preview_label, size_label))
        
        dialog.wait_window(dialog)
    
    def _validate_scale(self, value):
        """Validate scale input - allow empty, decimal numbers between 0.1 and 2.0"""
        if value == "":
            return True
        try:
            num = float(value)
            return 0.05 <= num <= 2.0
        except ValueError:
            return False
    
    def _set_preset(self, value, dialog, preview_label, size_label):
        """Set a preset scale value"""
        self.scale_var.set(value)
        self._update_preview_numeric(dialog, preview_label, size_label)
    
    def _update_preview_numeric(self, dialog, preview_label, size_label):
        """Update preview based on numeric input"""
        try:
            scale_text = self.scale_var.get()
            if not scale_text:
                preview_label.config(
                    image="", 
                    text="Enter a scale value", 
                    compound="center",
                    relief="flat"
                )
                size_label.config(text="")
                return
            
            scale = float(scale_text)
            if not (0.05 <= scale <= 2.0):
                preview_label.config(
                    image="", 
                    text="Scale must be between 0.05 and 2.0", 
                    compound="center",
                    relief="flat"
                )
                size_label.config(text="")
                return
            
            img_path = os.path.join(self.dataset.raw_images_path, self.raw_images[0])
            with Image.open(img_path) as img:
                original_size = (img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                
                # Create downscaled image
                downscaled_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Scale for display (fit within available preview area)
                display_img = downscaled_img.copy()
                # Get current preview container size or use reasonable defaults
                max_width = 800
                max_height = 400
                display_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                
                # Create PhotoImage
                dialog.preview_photo = ImageTk.PhotoImage(display_img)
                preview_label.config(
                    image=dialog.preview_photo, 
                    text="", 
                    compound="center",
                    relief="sunken",
                    borderwidth=1
                )
                
                # Update size information with better formatting
                size_info = f"Original Size: {original_size[0]} × {original_size[1]} pixels\n"
                size_info += f"New Size: {new_size[0]} × {new_size[1]} pixels\n"
                size_info += f"Scale Factor: {scale:.3f} ({scale*100:.1f}%)\n"
                size_info += f"Estimated File Size: {(scale**2)*100:.1f}% of original"
                size_label.config(text=size_info)
                
        except (ValueError, FileNotFoundError, Exception) as e:
            preview_label.config(
                image="", 
                text=f"Preview error: {str(e)[:50]}...", 
                compound="center",
                relief="flat"
            )
            size_label.config(text="Error loading preview")
    
    def _confirm_scale(self, dialog):
        """Confirm the scale selection"""
        try:
            scale_text = self.scale_var.get()
            scale = float(scale_text) if scale_text else 1.0
            if 0.05 <= scale <= 2.0:
                self.downscale_factor = scale
                dialog.destroy()
                self._initialize_main_app()
            else:
                # Invalid scale, reset to 1.0
                self.scale_var.set("1.0")
        except ValueError:
            self.scale_var.set("1.0")
    
    def _cancel_scale(self, dialog):
        """Cancel scale selection"""
        self.downscale_factor = -1
        dialog.destroy()
        self.destroy()

    def _initialize_main_app(self):
        self.title(f"Aim-ML - Labeling: {self.dataset.session_name}")
        self.geometry("1000x800")
        self.state('zoomed')
        self._start_image_processing()
        
    def _start_image_processing(self):
        self.deiconify()
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', mode='determinate', length=300)
        self.progress_bar.pack(pady=20)
        self.progress_label = ttk.Label(self, text="Preparing to process images...")
        self.progress_label.pack()
        self.progress_queue = queue.Queue()
        self.progress_bar['maximum'] = len(self.raw_images)
        self.processing_thread = threading.Thread(target=self._threaded_process_images, args=(self.progress_queue,), daemon=True)
        self.processing_thread.start()
        self.after(100, self._check_progress_queue)

    def _threaded_process_images(self, q: queue.Queue):
        total = len(self.raw_images)
        for i, img_name in enumerate(self.raw_images):
            try:
                raw_path = os.path.join(self.dataset.raw_images_path, img_name)
                processed_path = os.path.join(self.dataset.processed_images_path, img_name)
                if self.downscale_factor < 1.0:
                    with Image.open(raw_path) as img:
                        new_size = (int(img.width * self.downscale_factor), int(img.height * self.downscale_factor))
                        resized_img = img.resize(new_size, Image.Resampling.BILINEAR)
                        resized_img.save(processed_path, "jpeg")
                else:
                    shutil.copy(raw_path, processed_path)
                q.put({'progress': i + 1, 'status': f"Processing image {i + 1}/{total}..."})
            except Exception as e:
                q.put({'error': f"Failed on {img_name}: {e}"})
        q.put({'done': True})

    def _check_progress_queue(self):
        try:
            while not self.progress_queue.empty():
                message = self.progress_queue.get_nowait()
                if 'progress' in message:
                    self.progress_bar['value'] = message['progress']
                    self.progress_label.config(text=message['status'])
                elif 'error' in message:
                    print(message['error'])
                elif 'done' in message:
                    self._on_processing_complete()
                    return
        finally:
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.after(100, self._check_progress_queue)

    def _on_processing_complete(self):
        self.progress_bar.destroy()
        self.progress_label.destroy()
        self.dataset.load_labels()
        self.processed_images = self.dataset.get_all_processed_images()
        if not self.processed_images:
            print("Error: Failed to process any images. Check console for errors.")
            self.destroy()
            return
        self._build_main_ui()
        self.load_frame()

    def _build_main_ui(self):
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill='x', padx=10, pady=5)
        self.canvas = tk.Canvas(self, cursor="cross", bg="gray")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor='w')
        status_bar.pack(side='bottom', fill='x', padx=10, pady=2)
        self.flag_var = tk.StringVar(value=self.dataset.action_flags[0])
        ttk.Label(controls_frame, text="Action Flag:").pack(side='left', padx=(0, 5))
        for flag in self.dataset.action_flags:
            rb = ttk.Radiobutton(controls_frame, text=flag.capitalize(), variable=self.flag_var, value=flag)
            rb.pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Skip Frame (Space)", command=self.skip_frame).pack(side='right', padx=5)
        ttk.Button(controls_frame, text="Undo (U)", command=self.undo).pack(side='right')
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # NEW: Bind to the Configure event to handle window resizing
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.bind("<KeyPress-u>", lambda event: self.undo())
        self.bind("<KeyPress-U>", lambda event: self.undo())
        self.bind("<space>", lambda event: self.skip_frame())
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_canvas_resize(self, event):
        """Called whenever the canvas is resized to redraw the image."""
        self._display_current_image()

    def _display_current_image(self):
        """Scales and displays the loaded self.original_img to fit the canvas."""
        if self.original_img is None:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 2 or canvas_h < 2: return # Avoid drawing on a tiny canvas

        orig_w, orig_h = self.original_img.size
        ratio = min(canvas_w / orig_w, canvas_h / orig_h)
        self.display_photo_size = (int(orig_w * ratio), int(orig_h * ratio))

        display_img = self.original_img.resize(self.display_photo_size, Image.Resampling.NEAREST)
        self.photo = ImageTk.PhotoImage(display_img)

        self.image_offsets = ((canvas_w - self.display_photo_size[0]) // 2, (canvas_h - self.display_photo_size[1]) // 2)

        self.canvas.delete("all")
        self.canvas.create_image(self.image_offsets[0], self.image_offsets[1], anchor='nw', image=self.photo)

    def load_frame(self):
        """Loads the current frame from disk and triggers a redraw."""
        if not (0 <= self.current_frame_index < len(self.processed_images)):
            self.display_completion_message()
            return
        image_name = self.processed_images[self.current_frame_index]
        image_path = os.path.join(self.dataset.processed_images_path, image_name)
        try:
            self.original_img = Image.open(image_path)
            self._display_current_image()
        except IOError as e:
            print(f"Error: Could not load image: {image_path}\n{e}")
            self.original_img = None
            self.canvas.delete("all")
            self.skip_frame()
            return
        self.update_status()

    def update_status(self, coords: Optional[tuple] = None, mapped_coords: Optional[tuple] = None):
        """Updates the status bar text, now showing mapped coordinates."""
        coord_str = "(-, -)"
        if coords and mapped_coords:
            coord_str = f"Clicked: ({coords[0]}, {coords[1]}) -> Mapped: ({mapped_coords[0]}, {mapped_coords[1]})"
        elif coords:
            coord_str = f"Coords: ({coords[0]}, {coords[1]})"

        total_frames = len(self.processed_images)
        percentage = ((self.current_frame_index + 1) / total_frames * 100) if total_frames > 0 else 0
        frame_str = f"({percentage:.1f}%) Frame: {self.current_frame_index + 1}/{total_frames}"
        flag_str = f"Flag: {self.flag_var.get().capitalize()}"
        self.status_var.set(f"{frame_str} | {flag_str} | {coord_str}")

    def on_canvas_click(self, event):
        """Handles mouse clicks, mapping coordinates from display to original image size."""
        if self.original_img is None: return

        click_x = event.x - self.image_offsets[0]
        click_y = event.y - self.image_offsets[1]

        if not (0 <= click_x < self.display_photo_size[0] and 0 <= click_y < self.display_photo_size[1]): return

        orig_w, orig_h = self.original_img.size
        display_w, display_h = self.display_photo_size
        if display_w == 0 or display_h == 0: return

        ratio_w = orig_w / display_w
        ratio_h = orig_h / display_h

        mapped_x = int(click_x * ratio_w)
        mapped_y = int(click_y * ratio_h)

        image_name = self.processed_images[self.current_frame_index]
        flag = self.flag_var.get()
        self.undo_stack.append({'index': self.current_frame_index, 'label': self.dataset.get_label_for_image(image_name)})
        self.dataset.add_or_update_label(image_name, mapped_x, mapped_y, flag)
        self.update_status(coords=(event.x, event.y), mapped_coords=(mapped_x, mapped_y))
        self.next_frame()

    def next_frame(self):
        self.current_frame_index += 1
        if self.dataset.action_flags:
            self.flag_var.set(self.dataset.action_flags[0])
        self.load_frame()

    def skip_frame(self):
        image_name = self.processed_images[self.current_frame_index]
        self.undo_stack.append({'index': self.current_frame_index, 'label': self.dataset.get_label_for_image(image_name)})
        rel_path = os.path.join("images", image_name).replace("\\", "/")
        self.dataset.labels = [lbl for lbl in self.dataset.labels if lbl['image_path'] != rel_path]
        self.next_frame()
        
    def undo(self):
        if not self.undo_stack:
            print("Info: Nothing to undo.")
            return
        last_action = self.undo_stack.pop()
        self.current_frame_index = last_action['index']
        prev_label = last_action['label']
        image_name = self.processed_images[self.current_frame_index]
        rel_path = os.path.join("images", image_name).replace("\\", "/")
        self.dataset.labels = [lbl for lbl in self.dataset.labels if lbl['image_path'] != rel_path]
        if prev_label:
            self.dataset.labels.append(prev_label)
        self.load_frame()

    def display_completion_message(self):
        self.dataset.save_labels()
        print("Complete: All frames have been processed. Dataset exported.")
        
        # Show completion message briefly, then auto-close
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, 
                               text="Labeling Complete!\nDataset saved.\nClosing...", font=("Arial", 16))
        
        # Auto-close after 2 seconds and notify parent
        self.after(2000, self._auto_close)

    def _auto_close(self):
        """Auto-close after completion"""
        try:
            if self.on_complete_callback:
                self.on_complete_callback()
        except Exception as e:
            print(f"Error in completion callback: {e}")
        finally:
            self.destroy()
    
    def on_closing(self):
        # Always save progress on close
        self.dataset.save_labels()
        print("Progress saved.")
        
        # Call completion callback if available
        try:
            if self.on_complete_callback:
                self.on_complete_callback()
        except Exception as e:
            print(f"Error in completion callback: {e}")
        finally:
            self.destroy()
