# ml_components.py

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import mss
from pynput import mouse
import pydirectinput
import json
import threading
import queue
from datetime import datetime
from typing import Optional, List, Dict, Callable

# --- Architecture Configurations ---
DEFAULT_INPUT_RESOLUTION = (256, 144)  # (width, height) - common 16:9 ratio

# Architecture Types
ARCHITECTURE_TYPES = {
    "CNN_ONLY": "CNN Only",
    "RNN_ONLY": "RNN Only", 
    "CNN_RNN": "CNN + RNN",
    "CNN_LSTM": "CNN + LSTM",
    "CNN_GRU": "CNN + GRU",
    "TRANSFORMER": "Vision Transformer"
}

# Architecture Size Scaling Factors
ARCHITECTURE_SIZES = {
    "small": {
        "name": "Small",
        "description": "Fast training, low memory usage",
        "base_filters": 8,
        "filter_multiplier": 1.0,
        "dense_ratio": 0.05,  # Much smaller ratio to keep dense units reasonable
        "rnn_ratio": 0.5,     # RNN units as ratio of dense units
        "transformer_dim_ratio": 0.125,  # Transformer dim as ratio of input area
        "transformer_heads": 2,
        "transformer_layers": 2
    },
    "medium": {
        "name": "Medium",
        "description": "Balanced performance and speed",
        "base_filters": 16,
        "filter_multiplier": 1.5,
        "dense_ratio": 0.1,  # Smaller ratio
        "rnn_ratio": 0.5,
        "transformer_dim_ratio": 0.25,
        "transformer_heads": 4,
        "transformer_layers": 3
    },
    "large": {
        "name": "Large", 
        "description": "Better accuracy, slower training",
        "base_filters": 32,
        "filter_multiplier": 2.0,
        "dense_ratio": 0.15,  # Smaller ratio
        "rnn_ratio": 0.5,
        "transformer_dim_ratio": 0.5,
        "transformer_heads": 8,
        "transformer_layers": 4
    },
    "huge": {
        "name": "Huge",
        "description": "Maximum accuracy, requires powerful GPU",
        "base_filters": 64,
        "filter_multiplier": 3.0,
        "dense_ratio": 1.0,
        "rnn_ratio": 0.5,
        "transformer_dim_ratio": 1.0,
        "transformer_heads": 12,
        "transformer_layers": 6
    }
}

# Legacy defaults for backwards compatibility
DEFAULT_CONV_FILTERS = [16, 32, 64]
DEFAULT_DENSE_UNITS = [128]
DEFAULT_RNN_UNITS = 64

# Training parameters
DEFAULT_NUM_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.001

# Live Test Parameters
DEFAULT_TARGET_FPS = 30                # Target frame rate for live testing
DEFAULT_BLOCK_SIZE = 4                 # Size of each "big pixel" in the live preview

def calculate_dynamic_architecture(input_res: tuple, size_key: str) -> Dict:
    """Calculate dynamic architecture parameters based on input resolution"""
    width, height = input_res
    input_area = width * height
    
    # Get size configuration
    size_config = ARCHITECTURE_SIZES[size_key]
    
    # Calculate number of conv layers based on input size
    # Larger inputs get more layers to properly downsample
    min_layers = 3
    if input_area > 200000:  # > 500x400
        num_conv_layers = 5
    elif input_area > 100000:  # > 350x285
        num_conv_layers = 4
    else:
        num_conv_layers = min_layers
    
    # Calculate conv filters dynamically
    base_filters = size_config["base_filters"]
    multiplier = size_config["filter_multiplier"]
    
    # Start with base filters, scale based on input size
    input_scale_factor = max(1.0, (input_area / 36864) ** 0.3)  # 36864 = 256*144
    scaled_base = max(4, int(base_filters * input_scale_factor))
    
    conv_filters = []
    for i in range(num_conv_layers):
        filters = int(scaled_base * (multiplier ** i))
        # Ensure even numbers and reasonable bounds
        filters = max(4, min(512, filters + filters % 2))
        conv_filters.append(filters)
    
    # Calculate feature map size after conv layers (assuming 2x2 pooling each layer)
    final_width = width // (2 ** num_conv_layers)
    final_height = height // (2 ** num_conv_layers)
    final_features = conv_filters[-1] * final_width * final_height
    
    # Calculate dense units based on final conv features with better bounds
    base_dense = int(final_features * size_config["dense_ratio"])
    # Apply reasonable bounds similar to legacy defaults
    min_dense = 64   # Minimum reasonable size
    max_dense = 512  # Much more reasonable maximum
    dense_units = max(min_dense, min(max_dense, base_dense))
    # Ensure even numbers
    dense_units = dense_units + dense_units % 2
    
    # Calculate RNN units
    rnn_units = max(16, int(dense_units * size_config["rnn_ratio"]))
    rnn_units = min(512, rnn_units + rnn_units % 2)
    
    # Calculate transformer dimensions
    transformer_dim = max(32, int(input_area * size_config["transformer_dim_ratio"] / 1000))
    # Ensure divisible by number of heads
    heads = size_config["transformer_heads"]
    transformer_dim = ((transformer_dim // heads) * heads)
    transformer_dim = max(heads * 4, min(768, transformer_dim))  # Reasonable bounds
    
    return {
        "conv_filters": conv_filters,
        "dense_units": [dense_units],
        "rnn_units": rnn_units,
        "transformer_dim": transformer_dim,
        "transformer_heads": size_config["transformer_heads"],
        "transformer_layers": size_config["transformer_layers"],
        "input_area": input_area,
        "final_conv_features": final_features,
        "num_conv_layers": num_conv_layers
    }

class ModelMetadata:
    """Manages model metadata and storage"""
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.metadata_file = os.path.join(models_dir, "models.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load model metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading model metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save model metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving model metadata: {e}")

    def save_model_info(self, model_name: str, session_path: str, model_config: Dict, training_info: Dict):
        """Save information about a trained model"""
        model_info = {
            "name": model_name,
            "session_path": session_path,
            "created_date": datetime.now().isoformat(),
            "model_config": model_config,
            "training_info": training_info,
            "file_path": os.path.join(self.models_dir, f"{model_name}.pth")
        }
        
        self.metadata[model_name] = model_info
        self._save_metadata()

    def refresh_metadata(self):
        """Refresh metadata from disk"""
        self.metadata = self._load_metadata()
    
    def get_all_models(self, refresh: bool = True) -> Dict:
        """Get all available models"""
        if refresh:
            self.refresh_metadata()
        return self.metadata

    def get_model_info(self, model_name: str, refresh: bool = True) -> Optional[Dict]:
        """Get information about a specific model"""
        if refresh:
            self.refresh_metadata()
        return self.metadata.get(model_name)

    def delete_model(self, model_name: str):
        """Delete a model and its metadata"""
        if model_name in self.metadata:
            model_path = self.metadata[model_name]["file_path"]
            if os.path.exists(model_path):
                os.remove(model_path)
            del self.metadata[model_name]
            self._save_metadata()

class FrameLabelDataset(Dataset):
    """
    Loads image frames and normalized labels from the Tkinter labeling tool's output.
    """
    def __init__(self, session_path, input_res):
        self.session_path = session_path
        self.input_res = input_res  # (width, height)
        
        labels_csv_path = os.path.join(session_path, 'labels.csv')
        self.labels_df = pd.read_csv(labels_csv_path)

        print(f"Found {len(self.labels_df)} labeled frames in {session_path}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Construct full image path
        img_path = os.path.join(self.session_path, row['image_path'])
        frame = cv2.imread(img_path)
        
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Get original image dimensions for normalization
        original_h, original_w, _ = frame.shape
        
        # Preprocess the image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.input_res)
        frame = frame / 255.0  # Normalize to [0, 1]
        frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW for PyTorch

        # Normalize labels to be [0, 1] based on original image dimensions
        x_label = row['x'] / original_w
        y_label = row['y'] / original_h
        
        labels = np.array([x_label, y_label], dtype=np.float32)

        return torch.FloatTensor(frame), torch.FloatTensor(labels)

class DynamicCNN(nn.Module):
    """Simplified Dynamic CNN backbone"""
    def __init__(self, input_res, conv_filters):
        super(DynamicCNN, self).__init__()
        
        self.input_res = input_res
        w, h = input_res
        
        # Build standard conv layers with pooling after each
        layers = []
        in_channels = 3
        current_w, current_h = w, h
        
        for out_channels in conv_filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # Pool after each conv layer
            ])
            in_channels = out_channels
            current_w = current_w // 2
            current_h = current_h // 2
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate final feature map size
        self.final_width = current_w
        self.final_height = current_h
        self.final_channels = conv_filters[-1]
        self.total_features = self.final_channels * self.final_width * self.final_height
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        # Flatten for dense layers
        x = x.contiguous().view(x.size(0), -1)
        return x

class SpatialCoordinateHead(nn.Module):
    """Simple coordinate prediction head"""
    def __init__(self, input_features, input_res):
        super(SpatialCoordinateHead, self).__init__()
        
        self.input_res = input_res
        
        # Better coordinate prediction without sigmoid saturation
        self.coord_predictor = nn.Sequential(
            nn.Linear(input_features, 2),
            nn.Tanh()  # Use tanh to avoid saturation, then scale to [0,1]
        )
        
        # Initialize weights properly for coordinate prediction
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable coordinate prediction"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use very small initialization to keep tanh in linear region
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very small gain
                if m.bias is not None:
                    # Initialize bias to 0 for tanh (will center around 0.5 after scaling)
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, features):
        # Get tanh output in [-1, 1] range
        tanh_output = self.coord_predictor(features)
        # Scale to [0, 1] range: (tanh + 1) / 2
        coords = (tanh_output + 1.0) * 0.5
        return coords

class CNNRNN(nn.Module):
    """
    Dynamic CNN-RNN model that adapts to input size
    """
    def __init__(self, input_res, conv_filters, dense_units, rnn_units):
        super(CNNRNN, self).__init__()
        
        # Dynamic CNN backbone
        self.cnn = DynamicCNN(input_res, conv_filters)
        
        # Dense layer
        self.fc1 = nn.Linear(self.cnn.total_features, dense_units[0])
        
        # RNN layer
        self.rnn = nn.RNN(input_size=dense_units[0], hidden_size=rnn_units, batch_first=True)
        
        # Spatial coordinate head
        self.coord_head = SpatialCoordinateHead(rnn_units, input_res)

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)
        features = torch.relu(self.fc1(features))
        
        # RNN processing (treat single frame as sequence of length 1)
        features = features.unsqueeze(1)
        features, _ = self.rnn(features)
        features = features.squeeze(1)
        
        # Spatial coordinate prediction
        coords = self.coord_head(features)
        return coords

class CNNOnly(nn.Module):
    """Dynamic CNN-only architecture with spatial coordinate prediction"""
    def __init__(self, input_res, conv_filters, dense_units):
        super(CNNOnly, self).__init__()
        
        # Dynamic CNN backbone
        self.cnn = DynamicCNN(input_res, conv_filters)
        
        # Feature processing
        self.fc1 = nn.Linear(self.cnn.total_features, dense_units[0])
        self.dropout = nn.Dropout(0.2)
        
        # Spatial coordinate head
        self.coord_head = SpatialCoordinateHead(dense_units[0], input_res)
        
        # Initialize FC layer weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for the fully connected layer"""
        for m in [self.fc1]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)
        
        # Feature processing
        features = torch.relu(self.fc1(features))
        features = self.dropout(features)
        
        # Spatial coordinate prediction
        coords = self.coord_head(features)
        
        return coords

class RNNOnly(nn.Module):
    """RNN-only architecture using flattened image as sequence"""
    def __init__(self, input_res, rnn_units):
        super(RNNOnly, self).__init__()
        
        w, h = input_res
        self.input_size = w * h * 3  # Flattened RGB image
        self.hidden_size = rnn_units
        
        # Reduce dimensionality first
        self.input_projection = nn.Linear(self.input_size, rnn_units)
        self.rnn = nn.RNN(rnn_units, rnn_units, batch_first=True, num_layers=2)
        self.output_projection = nn.Linear(rnn_units, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        
        x = torch.relu(self.input_projection(x))
        x = x.unsqueeze(1)  # Add sequence dimension
        
        x, _ = self.rnn(x)
        x = x.squeeze(1)  # Remove sequence dimension
        x = self.dropout(x)
        return torch.sigmoid(self.output_projection(x))

class CNNLSTM(nn.Module):
    """Dynamic CNN + LSTM architecture"""
    def __init__(self, input_res, conv_filters, dense_units, lstm_units):
        super(CNNLSTM, self).__init__()
        
        # Dynamic CNN backbone
        self.cnn = DynamicCNN(input_res, conv_filters)
        
        # Dense and LSTM layers
        self.fc1 = nn.Linear(self.cnn.total_features, dense_units[0])
        self.lstm = nn.LSTM(dense_units[0], lstm_units, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.2)
        
        # Spatial coordinate head
        self.coord_head = SpatialCoordinateHead(lstm_units, input_res)
    
    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)
        features = torch.relu(self.fc1(features))
        
        # LSTM processing
        features = features.unsqueeze(1)  # Add sequence dimension
        features, _ = self.lstm(features)
        features = features.squeeze(1)  # Remove sequence dimension
        
        features = self.dropout(features)
        
        # Spatial coordinate prediction
        coords = self.coord_head(features)
        
        return coords

class CNNGRU(nn.Module):
    """Dynamic CNN + GRU architecture"""
    def __init__(self, input_res, conv_filters, dense_units, gru_units):
        super(CNNGRU, self).__init__()
        
        # Dynamic CNN backbone
        self.cnn = DynamicCNN(input_res, conv_filters)
        
        # Dense and GRU layers
        self.fc1 = nn.Linear(self.cnn.total_features, dense_units[0])
        self.gru = nn.GRU(dense_units[0], gru_units, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.2)
        
        # Spatial coordinate head
        self.coord_head = SpatialCoordinateHead(gru_units, input_res)
    
    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)
        features = torch.relu(self.fc1(features))
        
        # GRU processing
        features = features.unsqueeze(1)  # Add sequence dimension
        features, _ = self.gru(features)
        features = features.squeeze(1)  # Remove sequence dimension
        
        features = self.dropout(features)
        
        # Spatial coordinate prediction
        coords = self.coord_head(features)
        
        return coords

class VisionTransformer(nn.Module):
    """Dynamic Vision Transformer that adapts patch size to input resolution"""
    def __init__(self, input_res, transformer_dim, num_heads, num_layers):
        super(VisionTransformer, self).__init__()
        
        w, h = input_res
        
        # Dynamic patch size based on input resolution - must divide both dimensions
        if w * h < 50000:  # < ~224x224
            preferred_patch = 8
        elif w * h < 100000:  # < ~316x316
            preferred_patch = 12
        else:
            preferred_patch = 16
        
        # Ensure patch_size divides both width and height
        self.patch_size = self._find_valid_patch_size(w, h, preferred_patch)
        self.num_patches = (w // self.patch_size) * (h // self.patch_size)
        patch_dim = 3 * self.patch_size * self.patch_size
        
        self.patch_embedding = nn.Linear(patch_dim, transformer_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, transformer_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(transformer_dim, 2)
    
    def _find_valid_patch_size(self, w: int, h: int, preferred: int) -> int:
        """Find patch size that divides both dimensions (max 4 patches per dim)."""
        preferred = min(preferred, w // 4, h // 4)
        preferred = max(4, preferred)
        # Try preferred and divisors
        for ps in [preferred, 8, 16, 12, 6, 4]:
            if w % ps == 0 and h % ps == 0 and ps <= w // 4 and ps <= h // 4:
                return ps
        # Fallback: use gcd of common divisors
        return 4
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)
        
        # Embed patches
        x = self.patch_embedding(patches)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x += self.pos_embedding
        
        # Transform
        x = self.transformer(x)
        
        # Use class token for prediction
        x = x[:, 0]  # Take class token
        return torch.sigmoid(self.fc(x))

def create_model(architecture_type: str, input_res: tuple, size_key: str):
    """Factory function to create different model architectures with dynamic sizing"""
    # Calculate dynamic architecture parameters
    arch_config = calculate_dynamic_architecture(input_res, size_key)
    
    if architecture_type == "CNN_ONLY":
        return CNNOnly(input_res, arch_config["conv_filters"], arch_config["dense_units"])
    elif architecture_type == "RNN_ONLY":
        return RNNOnly(input_res, arch_config["rnn_units"])
    elif architecture_type == "CNN_RNN":
        return CNNRNN(input_res, arch_config["conv_filters"], arch_config["dense_units"], arch_config["rnn_units"])
    elif architecture_type == "CNN_LSTM":
        return CNNLSTM(input_res, arch_config["conv_filters"], arch_config["dense_units"], arch_config["rnn_units"])
    elif architecture_type == "CNN_GRU":
        return CNNGRU(input_res, arch_config["conv_filters"], arch_config["dense_units"], arch_config["rnn_units"])
    elif architecture_type == "TRANSFORMER":
        return VisionTransformer(input_res, arch_config["transformer_dim"], 
                               arch_config["transformer_heads"], arch_config["transformer_layers"])
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

def estimate_model_parameters(architecture_type: str, input_res: tuple, size_key: str) -> dict:
    """Estimate model parameters and performance characteristics with dynamic sizing"""
    # Get dynamic architecture config
    arch_config = calculate_dynamic_architecture(input_res, size_key)
    
    # Create model for parameter estimation
    model = create_model(architecture_type, input_res, size_key)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    # Estimate relative training speed (smaller = faster)
    # Adjust speed based on actual model complexity
    base_speed = {
        "CNN_ONLY": 1.0,
        "RNN_ONLY": 0.8, 
        "CNN_RNN": 0.7,
        "CNN_LSTM": 0.5,
        "CNN_GRU": 0.6,
        "TRANSFORMER": 0.3
    }.get(architecture_type, 0.5)
    
    # Adjust speed based on model size (larger models are slower)
    size_factor = max(0.1, min(2.0, 1.0 / (total_params / 100000) ** 0.2))
    speed_factor = base_speed * size_factor
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params, 
        "model_size_mb": param_size_mb,
        "relative_speed": speed_factor,
        "estimated_training_time_factor": 1.0 / speed_factor,
        "architecture_details": {
            "conv_layers": len(arch_config.get("conv_filters", [])),
            "conv_filters": arch_config.get("conv_filters", []),
            "dense_units": arch_config.get("dense_units", []),
            "rnn_units": arch_config.get("rnn_units", 0),
            "transformer_dim": arch_config.get("transformer_dim", 0),
            "input_area": arch_config.get("input_area", 0)
        }
    }

class ModelTrainer:
    """Handles model training with progress reporting"""
    def __init__(self, progress_callback: Callable[[str, float], None] = None):
        self.progress_callback = progress_callback
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_metadata = ModelMetadata()
        
    def train_model(self, session_path: str, model_name: str, config: Dict = None) -> bool:
        """Train a model on the given session data"""
        if config is None:
            config = {
                "architecture_type": "CNN_RNN",
                "architecture_size": "medium", 
                "input_resolution": DEFAULT_INPUT_RESOLUTION,
                "num_epochs": DEFAULT_NUM_EPOCHS,
                "batch_size": DEFAULT_BATCH_SIZE,
                "learning_rate": DEFAULT_LEARNING_RATE
            }
        
        try:
            if self.progress_callback:
                self.progress_callback("Initializing training...", 0.0)
            
            # Get architecture configuration
            architecture_type = config.get("architecture_type", "CNN_RNN")
            architecture_size = config.get("architecture_size", "medium")
            
            # Create model using factory function with dynamic sizing
            model = create_model(architecture_type, config["input_resolution"], architecture_size).to(self.device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

            dataset = FrameLabelDataset(session_path, config["input_resolution"])
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

            if self.progress_callback:
                self.progress_callback(f"Training on {len(dataset)} samples...", 0.0)
            
            training_start_time = time.time()
            
            for epoch in range(config["num_epochs"]):
                epoch_start_time = time.time()
                model.train()
                running_loss = 0.0
                
                for batch_idx, (frames, labels) in enumerate(loader):
                    frames, labels = frames.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(loader)
                progress = (epoch + 1) / config["num_epochs"]
                
                # Calculate ETA
                elapsed_time = time.time() - training_start_time
                epochs_completed = epoch + 1
                avg_time_per_epoch = elapsed_time / epochs_completed
                remaining_epochs = config["num_epochs"] - epochs_completed
                eta_seconds = remaining_epochs * avg_time_per_epoch
                
                # Format ETA
                if eta_seconds > 3600:
                    eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                elif eta_seconds > 60:
                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                else:
                    eta_str = f"{int(eta_seconds)}s"
                
                if self.progress_callback:
                    message = f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.6f}"
                    if remaining_epochs > 0:
                        message += f" | ETA: {eta_str}"
                    else:
                        message += " | Training complete!"
                    
                    self.progress_callback(message, progress)
            
            training_time = time.time() - training_start_time
            
            # Save model
            model_path = os.path.join(self.model_metadata.models_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            
            # Save metadata
            training_info = {
                "training_time_minutes": training_time / 60,
                "final_loss": avg_loss,
                "dataset_size": len(dataset)
            }
            
            self.model_metadata.save_model_info(model_name, session_path, config, training_info)
            
            if self.progress_callback:
                self.progress_callback(f"Training complete! Model saved as '{model_name}'", 1.0)
            
            return True
            
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"Training failed: {str(e)}", 0.0)
            print(f"Training error: {e}")
            return False

class LiveTester:
    """Handles live testing of trained models"""
    def __init__(self, stop_callback: Callable[[], None] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_metadata = ModelMetadata()
        self.model = None
        self.model_config = None
        self.running = False
        self.paused = False
        self.thread = None
        self.stop_callback = stop_callback
        
    def load_model(self, model_name: str) -> bool:
        """Load a trained model for testing"""
        try:
            print(f"Loading model '{model_name}'...")
            model_info = self.model_metadata.get_model_info(model_name)
            if not model_info:
                print(f"Error: Model '{model_name}' not found in metadata")
                return False
            
            model_path = model_info["file_path"]
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at path: {model_path}")
                return False
            
            self.model_config = model_info["model_config"]
            print(f"Model config: {self.model_config}")
            
            # Get architecture configuration 
            architecture_type = self.model_config.get("architecture_type", "CNN_RNN")
            architecture_size = self.model_config.get("architecture_size", "medium")
            
            if architecture_size not in ARCHITECTURE_SIZES:
                print(f"Error: Unknown architecture size '{architecture_size}'")
                return False
                
            print(f"Using architecture: {architecture_type} ({architecture_size})")
            
            # Normalize input_resolution to tuple (JSON may store as list)
            input_res = self.model_config["input_resolution"]
            input_res = tuple(input_res) if not isinstance(input_res, tuple) else input_res
            
            # Create model using factory function with dynamic sizing
            self.model = create_model(
                architecture_type, 
                input_res, 
                architecture_size
            ).to(self.device)
            
            # Load model weights (weights_only for security when available)
            print(f"Loading weights from {model_path}...")
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"Successfully loaded model '{model_name}' for testing")
            return True
            
        except RuntimeError as e:
            err_str = str(e)
            if "state_dict" in err_str and ("Missing key" in err_str or "Unexpected key" in err_str):
                print(f"Error loading model '{model_name}': Architecture mismatch. "
                      "This model may have been trained with an older version. Consider retraining.")
            else:
                print(f"Error loading model '{model_name}': {e}")
            return False
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_live_test(self, monitor_index: int = 1) -> bool:
        """Start live testing in a separate thread"""
        if self.model is None:
            print("Error: No model loaded for testing")
            return False
        
        if self.running:
            print("Warning: Live test already running")
            return False
        
        try:
            print(f"Starting live test on monitor {monitor_index}")
            self.monitor = self._get_monitor_config(monitor_index)
            print(f"Monitor config: {self.monitor}")
            
            # Test if we can capture from this monitor
            with mss.mss() as sct:
                test_capture = self._capture_frame(sct, self.monitor)
                if test_capture is None:
                    print("Error: Failed to capture test frame from selected monitor")
                    return False
            
            self.running = True
            self.paused = False
            self.thread = threading.Thread(target=self._live_test_loop, daemon=True)
            self.thread.start()
            print("Live test started successfully")
            return True
        except Exception as e:
            print(f"Error starting live test: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop_live_test(self):
        """Stop live testing"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        cv2.destroyAllWindows()
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        return self.paused
    
    def _get_monitor_config(self, monitor_index):
        """Get the monitor configuration for the specified index."""
        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor_index < 0 or monitor_index >= len(monitors):
                raise ValueError(f"Invalid monitor index {monitor_index}. Available monitors: {len(monitors)-1}")
            return monitors[monitor_index]
    
    def _capture_frame(self, sct, monitor):
        """Capture and process a frame from the specified monitor region."""
        try:
            screen = np.array(sct.grab(monitor))
            frame_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
            return frame_rgb
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def _create_big_pixel_preview(self, frame, preview_res, block_size):
        """Create a big pixel preview of the frame."""
        downscaled = cv2.resize(frame, (preview_res[0] // block_size, preview_res[1] // block_size), interpolation=cv2.INTER_AREA)
        preview = downscaled.repeat(block_size, axis=0).repeat(block_size, axis=1)
        return cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    
    def _draw_mouse_dot(self, frame, mouse_x, mouse_y, monitor):
        """Draw a red dot at the current mouse position on the frame."""
        height, width, _ = frame.shape
        norm_x = (mouse_x - monitor["left"]) / monitor["width"]
        norm_y = (mouse_y - monitor["top"]) / monitor["height"]
        preview_x = int(norm_x * width)
        preview_y = int(norm_y * height)
        dot_radius = max(2, DEFAULT_BLOCK_SIZE)
        cv2.circle(frame, (preview_x, preview_y), dot_radius, (0, 0, 255), -1)  # Red for actual mouse
        return frame
    
    def _display_preview(self, frame, pred_x=None, pred_y=None):
        """Display the preview with overlays."""
        cv2.putText(frame, f"Actions: {'Paused (P)' if self.paused else 'Running (P)'}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if pred_x is not None and pred_y is not None:
            cv2.circle(frame, (pred_x, pred_y), max(2, DEFAULT_BLOCK_SIZE), (0, 255, 0), -1)  # Green for predicted
        cv2.imshow("Live Prediction Preview", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False, False
        elif key == ord("p"):
            return True, True
        return True, False
    
    def _live_test_loop(self):
        """Main loop for live testing"""
        mouse_controller = mouse.Controller()
        input_res = self.model_config["input_resolution"]
        preview_size = (input_res[0] * DEFAULT_BLOCK_SIZE, input_res[1] * DEFAULT_BLOCK_SIZE)
        
        with mss.mss() as sct:
            while self.running:
                try:
                    start_time = time.time()
                    
                    frame_rgb = self._capture_frame(sct, self.monitor)
                    if frame_rgb is None:
                        continue
                    
                    # Prepare frame for the model
                    frame_resized = cv2.resize(frame_rgb, input_res)
                    frame_normalized = frame_resized / 255.0
                    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
                    frame_tensor = torch.FloatTensor(frame_transposed).unsqueeze(0).to(self.device)

                    # Get model prediction
                    with torch.no_grad():
                        prediction = self.model(frame_tensor)
                    
                    pred_norm_x, pred_norm_y = prediction.cpu().numpy()[0]
                    
                    # Move mouse if not paused
                    if not self.paused:
                        try:
                            target_x = int(pred_norm_x * self.monitor["width"] + self.monitor["left"])
                            target_y = int(pred_norm_y * self.monitor["height"] + self.monitor["top"])
                            pydirectinput.moveTo(target_x, target_y)
                        except Exception as move_error:
                            # Handle pyautogui failsafe and other mouse movement errors
                            error_msg = str(move_error).lower()
                            if 'failsafe' in error_msg or 'corner' in error_msg:
                                print("PyAutoGUI failsafe triggered - stopping live test for safety")
                                if self.stop_callback:
                                    self.stop_callback()
                                break
                            else:
                                print(f"Mouse movement error: {move_error}")
                                # Continue running even if mouse movement fails

                    # Create and display preview
                    preview = self._create_big_pixel_preview(frame_rgb, preview_size, DEFAULT_BLOCK_SIZE)
                    mouse_x, mouse_y = mouse_controller.position
                    preview = self._draw_mouse_dot(preview, mouse_x, mouse_y, self.monitor)
                    
                    pred_preview_x = int(pred_norm_x * preview_size[0])
                    pred_preview_y = int(pred_norm_y * preview_size[1])
                    
                    continue_display, paused_toggled = self._display_preview(preview, pred_preview_x, pred_preview_y)
                    if not continue_display:
                        break
                    if paused_toggled:
                        self.paused = not self.paused

                    # Maintain target FPS
                    elapsed = time.time() - start_time
                    sleep_time = max(0, 1.0 / DEFAULT_TARGET_FPS - elapsed)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"Error in live test loop: {e}")
                    break
        
        self.running = False
        cv2.destroyAllWindows()
