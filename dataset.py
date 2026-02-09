# dataset.py

import os
import csv
from typing import List, Dict, Optional

class Dataset:
    """
    Manages a labeling dataset, including file structure and CSV operations.
    """
    def __init__(self, session_name: str, base_path: str = "datasets"):
        self.session_name = session_name
        self.base_path = os.path.abspath(base_path)
        self.session_path = os.path.join(self.base_path, self.session_name)
        self.raw_images_path = os.path.join(self.session_path, "raw_images")
        self.processed_images_path = os.path.join(self.session_path, "images")
        self.labels_csv_path = os.path.join(self.session_path, "labels.csv")

        self.labels: List[Dict] = []
        self.action_flags = ['walk', 'attack', 'teleport'] # Extensible list of flags

        self._create_dirs()

    def _create_dirs(self):
        """Creates the necessary directories for the session if they don't exist."""
        os.makedirs(self.raw_images_path, exist_ok=True)
        os.makedirs(self.processed_images_path, exist_ok=True)

    def load_labels(self):
        """Loads labels from the CSV file into memory."""
        self.labels = []
        if not os.path.exists(self.labels_csv_path):
            return

        try:
            with open(self.labels_csv_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric strings back to numbers
                    processed_row = {k: (int(v) if v.isdigit() else v) for k, v in row.items()}
                    self.labels.append(processed_row)
        except (IOError, csv.Error) as e:
            print(f"Error loading labels CSV: {e}")

    def save_labels(self):
        """Saves the current in-memory labels to the CSV file."""
        if not self.labels:
            # If labels are empty, we might want to remove the csv or leave it empty.
            # For this implementation, we will not write an empty file if it doesnt exist.
            if not os.path.exists(self.labels_csv_path):
                 return
        
        headers = ['image_path', 'x', 'y'] + self.action_flags
        
        try:
            with open(self.labels_csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                # Filter out any potential malformed entries before writing
                valid_labels = [row for row in self.labels if all(k in row for k in headers)]
                writer.writerows(valid_labels)
        except IOError as e:
            print(f"Error saving labels CSV: {e}")

    def add_or_update_label(self, image_name: str, x: int, y: int, flag: str):
        """
        Adds a new label or updates an existing one for a given image.
        Ensures flag exclusivity.
        """
        # Create the label dictionary with all flags set to 0
        new_label = {f: 0 for f in self.action_flags}
        new_label['image_path'] = os.path.join("images", image_name).replace("\\", "/")
        new_label['x'] = x
        new_label['y'] = y
        
        # Set the active flag to 1
        if flag in new_label:
            new_label[flag] = 1
        else:
            raise ValueError(f"Unknown flag: {flag}")

        # Check if a label for this image already exists and update it
        for i, label in enumerate(self.labels):
            if label['image_path'] == new_label['image_path']:
                self.labels[i] = new_label
                return
        
        # If not found, append it
        self.labels.append(new_label)

    def get_label_for_image(self, image_name: str) -> Optional[Dict]:
        """Retrieves a label for a specific image name."""
        rel_path = os.path.join("images", image_name).replace("\\", "/")
        for label in self.labels:
            if label['image_path'] == rel_path:
                return label
        return None

    def get_all_processed_images(self) -> List[str]:
        """Returns a sorted list of all processed image filenames."""
        if not os.path.exists(self.processed_images_path):
            return []
        try:
            images = [f for f in os.listdir(self.processed_images_path) if f.lower().endswith('.jpg')]
            return sorted(images)
        except OSError:
            return []