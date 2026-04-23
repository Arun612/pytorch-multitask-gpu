"""
backbone.py — Feature Extractor Backbones
==========================================
Demonstrates:
  - Building custom CNN layers with nn.Module
  - Transfer learning with pretrained ResNet-18
  - Freezing/unfreezing parameters (requires_grad)
  - BatchNorm, ReLU, MaxPool, AdaptiveAvgPool
  - Sequential containers and layer composition
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class CustomCNNBackbone(nn.Module):
    """
    A hand-built CNN backbone for feature extraction.

    Architecture:
        Input (3, 32, 32)
        → [Conv → BN → ReLU → Conv → BN → ReLU → MaxPool] × 4 blocks
        → AdaptiveAvgPool → Flatten

    Demonstrates:
        - Manual layer construction with nn.Conv2d, nn.BatchNorm2d
        - nn.Sequential for organizing layer groups
        - Forward pass design
    """

    def __init__(self, in_channels: int = 3, channel_list: List[int] = None):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            channel_list: List of output channels per block, e.g. [32, 64, 128, 256]
        """
        super().__init__()

        if channel_list is None:
            channel_list = [32, 64, 128, 256]

        self.feature_dim = channel_list[-1]

        # Build convolutional blocks dynamically
        blocks = []
        current_channels = in_channels

        for out_channels in channel_list:
            blocks.append(self._make_block(current_channels, out_channels))
            current_channels = out_channels

        self.features = nn.Sequential(*blocks)

        # Global Average Pooling → collapses spatial dims to 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize weights using Kaiming initialization
        self._initialize_weights()

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Create a convolutional block:
            Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool2x2

        This is similar to VGG-style blocks.
        """
        return nn.Sequential(
            # First conv layer
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # Second conv layer (deepens the block)
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            # Spatial downsampling
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def _initialize_weights(self):
        """
        Kaiming initialization — best practice for ReLU-based networks.
        Demonstrates manual weight initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features from input images.

        Args:
            x: Input tensor of shape [B, 3, 32, 32]

        Returns:
            Feature tensor of shape [B, feature_dim]
        """
        x = self.features(x)        # [B, 256, 2, 2] after 4 pooling ops on 32×32
        x = self.global_pool(x)     # [B, 256, 1, 1]
        x = torch.flatten(x, 1)    # [B, 256]
        return x


class PretrainedBackbone(nn.Module):
    """
    Transfer Learning backbone using pretrained ResNet-18.

    Demonstrates:
        - Loading pretrained weights from torchvision
        - Removing the final classification layer
        - Freezing/unfreezing layers for fine-tuning
        - Adapting pretrained models to new input sizes

    ResNet-18 was pretrained on ImageNet (1000 classes, 224×224 images).
    We adapt it for CIFAR-10 (10 classes, 32×32 images).
    """

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        """
        Args:
            pretrained: Whether to load ImageNet pretrained weights
            freeze: Whether to freeze backbone weights initially
        """
        super().__init__()

        # Load ResNet-18 with or without pretrained weights
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            print("🔄 Loading pretrained ResNet-18 (ImageNet weights)")
        else:
            weights = None
            print("🆕 Initializing ResNet-18 from scratch")

        resnet = models.resnet18(weights=weights)

        # ── Adapt for CIFAR-10's 32×32 images ──
        # Original ResNet uses 7×7 conv + maxpool (designed for 224×224)
        # We replace with a 3×3 conv and remove the maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # If pretrained, partially transfer conv1 weights
        if pretrained:
            # Average the 7×7 pretrained weights to initialize 3×3
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.data
                # Center-crop the 7×7 kernel to 3×3
                self.conv1.weight.data = pretrained_weight[:, :, 2:5, 2:5]

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # Skip resnet.maxpool — would reduce 32×32 too aggressively

        # Copy residual layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension
        self.feature_dim = 512

        # ── Freeze backbone if requested ──
        if freeze:
            self.freeze()

    def freeze(self):
        """
        Freeze all backbone parameters.
        Useful for transfer learning: train only the task heads first.
        """
        for param in self.parameters():
            param.requires_grad = False
        print("❄️  Backbone frozen — only task heads will be trained")

    def unfreeze(self):
        """
        Unfreeze all backbone parameters for full fine-tuning.
        Usually done after initial head training.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen — full model will be trained")

    def unfreeze_from_layer(self, layer_num: int):
        """
        Gradually unfreeze from a specific layer onwards.
        Demonstrates progressive fine-tuning.

        Args:
            layer_num: Unfreeze from this layer (1-4) onwards
        """
        layers = {1: self.layer1, 2: self.layer2,
                  3: self.layer3, 4: self.layer4}

        for num, layer in layers.items():
            for param in layer.parameters():
                param.requires_grad = (num >= layer_num)

        print(f"🔓 Unfrozen layers {layer_num}-4, frozen layers 1-{layer_num - 1}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapted ResNet backbone.

        Args:
            x: Input tensor [B, 3, 32, 32]

        Returns:
            Feature tensor [B, 512]
        """
        x = self.conv1(x)        # [B, 64, 32, 32]
        x = self.bn1(x)
        x = self.relu(x)
        # No maxpool here (unlike standard ResNet)

        x = self.layer1(x)       # [B, 64, 32, 32]
        x = self.layer2(x)       # [B, 128, 16, 16]
        x = self.layer3(x)       # [B, 256, 8, 8]
        x = self.layer4(x)       # [B, 512, 4, 4]

        x = self.avgpool(x)      # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]

        return x
