"""
decoder_head.py — Image Reconstruction (Autoencoder Decoder) Head
==================================================================
Demonstrates:
  - ConvTranspose2d (transposed/deconvolution) for upsampling
  - Building decoder/generator networks
  - Sigmoid activation for pixel value output [0, 1]
  - Reshaping feature vectors into spatial feature maps
"""

import torch
import torch.nn as nn


class DecoderHead(nn.Module):
    """
    Reconstruction head: maps backbone features → reconstructed image.

    Architecture:
        Feature vector [B, feature_dim]
        → FC → Reshape to [B, C, 2, 2]
        → [ConvTranspose2d → BN → ReLU] × 4 upsampling blocks
        → Conv2d → Sigmoid → [B, 3, 32, 32]

    The decoder mirrors the encoder — upsampling at each stage to
    recover spatial resolution. This is the decoder half of an autoencoder.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        """
        Args:
            feature_dim: Input feature dimension from backbone
            hidden_dim: Base channel count for the decoder
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # ── Project feature vector into spatial feature map ──
        # feature_dim → hidden_dim * 2 * 2 (to reshape into [B, hidden_dim, 2, 2])
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2 * 2),
            nn.ReLU(inplace=True),
        )

        # ── Upsampling decoder using transposed convolutions ──
        # Each ConvTranspose2d doubles the spatial resolution
        self.decoder = nn.Sequential(
            # [B, 256, 2, 2] → [B, 128, 4, 4]
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # [B, 128, 4, 4] → [B, 64, 8, 8]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # [B, 64, 8, 8] → [B, 32, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # [B, 32, 16, 16] → [B, 16, 32, 32]
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final conv to get 3 channels (RGB)
            # [B, 16, 32, 32] → [B, 3, 32, 32]
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output pixels in [0, 1] range
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: decode features into a reconstructed image.

        Args:
            features: Backbone features [B, feature_dim]

        Returns:
            Reconstructed image [B, 3, 32, 32] with values in [0, 1]
        """
        # Project to spatial dimensions
        x = self.fc(features)                            # [B, hidden_dim * 4]
        x = x.view(-1, self.hidden_dim, 2, 2)          # [B, hidden_dim, 2, 2]

        # Upsample through decoder
        x = self.decoder(x)                              # [B, 3, 32, 32]

        return x
