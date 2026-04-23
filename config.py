"""
config.py — Centralized Configuration
======================================
All hyperparameters, paths, and settings in one place using Python dataclasses.
Demonstrates: dataclass usage, device auto-detection, path management.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DataConfig:
    """Configuration for data loading and augmentation."""
    data_dir: str = "./data"
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    # Image properties (CIFAR-10)
    image_size: Tuple[int, int] = (32, 32)
    num_channels: int = 3
    num_classes: int = 10
    class_names: Tuple[str, ...] = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )
    # Augmentation
    random_crop_padding: int = 4
    horizontal_flip_prob: float = 0.5
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    # Normalization (CIFAR-10 mean/std)
    normalize_mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    normalize_std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Backbone selection: 'custom_cnn' or 'resnet18'
    backbone_type: str = "resnet18"
    # Whether to use pretrained weights for transfer learning
    pretrained: bool = True
    # Whether to freeze backbone layers initially
    freeze_backbone: bool = False
    # Feature dimension output from backbone
    feature_dim: int = 512  # ResNet-18 outputs 512-d features
    # Classifier head
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    classifier_dropout: float = 0.3
    # Decoder head (autoencoder reconstruction)
    decoder_hidden_dim: int = 256
    # Custom CNN backbone settings
    custom_cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    # Basic training
    epochs: int = 30
    # Optimizer: 'adam', 'sgd', 'adamw'
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    # Learning rate scheduler: 'cosine', 'plateau', 'none'
    scheduler: str = "cosine"
    scheduler_patience: int = 5  # For ReduceLROnPlateau
    scheduler_min_lr: float = 1e-6
    # Multi-task loss weights
    classification_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.5
    # Learnable loss weights (uncertainty weighting)
    use_learnable_loss_weights: bool = False
    # Mixed precision training (AMP)
    use_amp: bool = True
    # Gradient clipping
    gradient_clip_value: float = 1.0
    use_gradient_clipping: bool = True
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    # Reproducibility
    seed: int = 42


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpoints."""
    # Checkpoint directory
    checkpoint_dir: str = "./checkpoints"
    # TensorBoard log directory
    tensorboard_dir: str = "./runs"
    # How often to log (in batches)
    log_interval: int = 50
    # Save visualizations
    output_dir: str = "./outputs"
    # Verbosity
    verbose: bool = True


@dataclass
class ProjectConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Auto-detect device and create directories."""
        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU (no CUDA GPU detected)")
            # Adjust settings for CPU
            self.training.use_amp = False
            self.data.batch_size = min(self.data.batch_size, 32)
            self.data.num_workers = 0

        # Create necessary directories
        for dir_path in [
            self.data.data_dir,
            self.logging.checkpoint_dir,
            self.logging.tensorboard_dir,
            self.logging.output_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    def summary(self) -> str:
        """Print a human-readable summary of the configuration."""
        lines = [
            "=" * 60,
            "  PROJECT CONFIGURATION SUMMARY",
            "=" * 60,
            f"  Device:              {self.device}",
            f"  Backbone:            {self.model.backbone_type}",
            f"  Pretrained:          {self.model.pretrained}",
            f"  Batch Size:          {self.data.batch_size}",
            f"  Epochs:              {self.training.epochs}",
            f"  Optimizer:           {self.training.optimizer}",
            f"  Learning Rate:       {self.training.learning_rate}",
            f"  LR Scheduler:        {self.training.scheduler}",
            f"  Weight Decay:        {self.training.weight_decay}",
            f"  AMP Enabled:         {self.training.use_amp}",
            f"  Gradient Clipping:   {self.training.use_gradient_clipping}",
            f"  Early Stopping:      patience={self.training.early_stopping_patience}",
            f"  Loss Weights:        cls={self.training.classification_loss_weight}, "
            f"recon={self.training.reconstruction_loss_weight}",
            "=" * 60,
        ]
        return "\n".join(lines)
