"""
dataset.py — Custom Dataset & Data Pipeline
=============================================
Demonstrates:
  - Custom torch.utils.data.Dataset
  - Data augmentation with torchvision.transforms
  - DataLoader with num_workers, pin_memory
  - Tensor operations in custom transforms
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from typing import Tuple, Dict

from config import DataConfig


# ──────────────────────────────────────────────
# Custom Transform (demonstrates tensor operations)
# ──────────────────────────────────────────────
class AddGaussianNoise:
    """
    Custom transform that adds Gaussian noise to a tensor.
    Demonstrates writing custom transforms using tensor operations.
    """
    def __init__(self, mean: float = 0.0, std: float = 0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Tensor operation: create noise with same shape and add it
        noise = torch.randn_like(tensor) * self.std + self.mean
        noisy_tensor = tensor + noise
        # Clamp to valid range [0, 1] before normalization
        return torch.clamp(noisy_tensor, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


# ──────────────────────────────────────────────
# Custom Dataset for Multi-Task Learning
# ──────────────────────────────────────────────
class CIFAR10MultiTaskDataset(Dataset):
    """
    Custom Dataset wrapping CIFAR-10 for multi-task learning.

    Returns:
        augmented_image: Augmented image tensor for the model input
        label: Class label (0-9) for classification task
        clean_image: Clean (non-augmented, normalized) image for reconstruction target

    This separation lets the model learn to reconstruct the clean image
    from augmented inputs — a form of denoising autoencoder.
    """

    def __init__(self, config: DataConfig, train: bool = True):
        """
        Args:
            config: Data configuration dataclass
            train: If True, load training set with augmentation; else test set
        """
        self.config = config
        self.train = train

        # ── Build transforms ──
        # Clean transform: only ToTensor + Normalize (for reconstruction target)
        self.clean_transform = T.Compose([
            T.ToTensor(),  # Converts PIL [0,255] → Tensor [0.0, 1.0]
        ])

        # Augmented transform: full augmentation pipeline (for model input)
        if train:
            self.augmented_transform = T.Compose([
                T.RandomCrop(
                    config.image_size[0],
                    padding=config.random_crop_padding
                ),
                T.RandomHorizontalFlip(p=config.horizontal_flip_prob),
                T.ColorJitter(
                    brightness=config.color_jitter_brightness,
                    contrast=config.color_jitter_contrast,
                ),
                T.ToTensor(),
                AddGaussianNoise(mean=0.0, std=0.02),  # Custom transform!
                T.Normalize(config.normalize_mean, config.normalize_std),
            ])
        else:
            # Test set: no augmentation, only normalize
            self.augmented_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(config.normalize_mean, config.normalize_std),
            ])

        # ── Download and load CIFAR-10 ──
        # torchvision handles downloading automatically
        self.dataset = torchvision.datasets.CIFAR10(
            root=config.data_dir,
            train=train,
            download=True,
        )

        print(f"📦 Loaded CIFAR-10 {'train' if train else 'test'} set: "
              f"{len(self.dataset)} samples")

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            augmented_image: [3, 32, 32] augmented & normalized tensor
            label: integer class label
            clean_image: [3, 32, 32] clean tensor (for reconstruction)
        """
        # Get raw PIL image and label
        pil_image, label = self.dataset[idx]

        # Apply transforms
        augmented_image = self.augmented_transform(pil_image)
        clean_image = self.clean_transform(pil_image)

        return augmented_image, label, clean_image

    def get_class_name(self, label: int) -> str:
        """Convert numeric label to human-readable class name."""
        return self.config.class_names[label]


# ──────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────
def create_data_loaders(config: DataConfig) -> Dict[str, DataLoader]:
    """
    Create train and test DataLoaders.

    Demonstrates:
        - DataLoader with num_workers for parallel data loading
        - pin_memory for faster GPU transfer
        - drop_last for consistent batch sizes during training

    Args:
        config: Data configuration

    Returns:
        Dictionary with 'train' and 'test' DataLoader instances
    """
    # Create datasets
    train_dataset = CIFAR10MultiTaskDataset(config, train=True)
    test_dataset = CIFAR10MultiTaskDataset(config, train=False)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,          # Shuffle training data each epoch
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,   # Faster CPU→GPU transfer
        drop_last=True,        # Drop incomplete last batch
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,         # No need to shuffle test data
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,       # Keep all test samples
    )

    print(f"📊 Train batches: {len(train_loader)} | "
          f"Test batches: {len(test_loader)} | "
          f"Batch size: {config.batch_size}")

    return {"train": train_loader, "test": test_loader}
