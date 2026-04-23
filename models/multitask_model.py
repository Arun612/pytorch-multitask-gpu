"""
multitask_model.py — Combined Multi-Task Model
================================================
Demonstrates:
  - Multi-task learning architecture (shared backbone + task heads)
  - Forward pass returning multiple outputs as a dictionary
  - Model composition: combining separately defined modules
  - Parameter counting and model summary
"""

import torch
import torch.nn as nn
from typing import Dict

from .backbone import CustomCNNBackbone, PretrainedBackbone
from .classifier_head import ClassifierHead
from .decoder_head import DecoderHead


class MultiTaskModel(nn.Module):
    """
    Multi-Task Image Intelligence Model.

    Architecture:
        Input Image [B, 3, 32, 32]
            │
            ▼
        ┌──────────────────┐
        │   Shared Backbone │  (CNN or ResNet-18)
        │   [B, 512]       │
        └────────┬─────────┘
                 │
          ┌──────┴──────┐
          ▼              ▼
    ┌───────────┐  ┌──────────────┐
    │ Classifier │  │   Decoder    │
    │   Head     │  │    Head      │
    │ [B, 10]    │  │ [B, 3,32,32]│
    └───────────┘  └──────────────┘
      (logits)      (reconstruction)

    The shared backbone learns features useful for BOTH tasks.
    This is the core idea of multi-task learning:
        - Classification forces high-level semantic understanding
        - Reconstruction forces low-level detail preservation
        - The backbone must learn both → richer representations
    """

    def __init__(
        self,
        backbone_type: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        num_classes: int = 10,
        classifier_hidden_dims=None,
        classifier_dropout: float = 0.3,
        decoder_hidden_dim: int = 256,
        custom_cnn_channels=None,
    ):
        """
        Args:
            backbone_type: 'resnet18' or 'custom_cnn'
            pretrained: Use pretrained weights (for resnet18)
            freeze_backbone: Freeze backbone parameters
            feature_dim: Feature vector dimension from backbone
            num_classes: Number of classification classes
            classifier_hidden_dims: Hidden layer dims for classifier
            classifier_dropout: Dropout rate for classifier
            decoder_hidden_dim: Base channels for decoder
            custom_cnn_channels: Channel list for custom CNN backbone
        """
        super().__init__()

        # ── Build backbone ──
        if backbone_type == "resnet18":
            self.backbone = PretrainedBackbone(
                pretrained=pretrained,
                freeze=freeze_backbone,
            )
            actual_feature_dim = self.backbone.feature_dim
        elif backbone_type == "custom_cnn":
            self.backbone = CustomCNNBackbone(
                in_channels=3,
                channel_list=custom_cnn_channels,
            )
            actual_feature_dim = self.backbone.feature_dim
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # ── Build task heads ──
        self.classifier_head = ClassifierHead(
            feature_dim=actual_feature_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
        )

        self.decoder_head = DecoderHead(
            feature_dim=actual_feature_dim,
            hidden_dim=decoder_hidden_dim,
        )

        # Store config for reference
        self.backbone_type = backbone_type

        # Print model summary
        self._print_summary()

    def _print_summary(self):
        """Print a summary of model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\n🧠 MultiTaskModel Summary:")
        print(f"   Backbone:    {self.backbone_type}")
        print(f"   Total params:     {total:>10,}")
        print(f"   Trainable params: {trainable:>10,}")
        print(f"   Frozen params:    {frozen:>10,}")
        print(f"   Size:             {total * 4 / 1024 / 1024:.1f} MB (float32)\n")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task model.

        Args:
            x: Input images [B, 3, 32, 32]

        Returns:
            Dictionary with:
                'classification': class logits [B, num_classes]
                'reconstruction': reconstructed image [B, 3, 32, 32]
                'features': backbone features [B, feature_dim] (for analysis)
        """
        # Shared feature extraction
        features = self.backbone(x)

        # Task-specific heads
        class_logits = self.classifier_head(features)
        reconstructed = self.decoder_head(features)

        return {
            "classification": class_logits,
            "reconstruction": reconstructed,
            "features": features,
        }

    @classmethod
    def from_config(cls, model_config, data_config) -> "MultiTaskModel":
        """
        Factory method to create model from config objects.
        Demonstrates the @classmethod pattern (like in your OOP notebook!).
        """
        return cls(
            backbone_type=model_config.backbone_type,
            pretrained=model_config.pretrained,
            freeze_backbone=model_config.freeze_backbone,
            feature_dim=model_config.feature_dim,
            num_classes=data_config.num_classes,
            classifier_hidden_dims=model_config.classifier_hidden_dims,
            classifier_dropout=model_config.classifier_dropout,
            decoder_hidden_dim=model_config.decoder_hidden_dim,
            custom_cnn_channels=model_config.custom_cnn_channels,
        )
