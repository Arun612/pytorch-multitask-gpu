"""
classifier_head.py — Classification Task Head
===============================================
Demonstrates:
  - Fully connected layers with nn.Linear
  - Dropout regularization
  - ReLU activation
  - Building task-specific heads for multi-task learning
"""

import torch
import torch.nn as nn
from typing import List


class ClassifierHead(nn.Module):
    """
    Classification head: maps backbone features → class logits.

    Architecture:
        Feature vector [B, feature_dim]
        → FC → BN → ReLU → Dropout
        → FC → BN → ReLU → Dropout
        → FC → logits [B, num_classes]

    The raw logits are returned (no softmax) because:
        - CrossEntropyLoss expects raw logits (it applies log-softmax internally)
        - This is numerically more stable
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 10,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            feature_dim: Input feature dimension from backbone
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build layers dynamically from hidden_dims list
        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),   # BN stabilizes training
                nn.ReLU(),
                nn.Dropout(p=dropout),         # Dropout prevents overfitting
            ])
            in_dim = hidden_dim

        # Final classification layer (no activation — raw logits)
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Initialize the final layer with smaller weights
        self._initialize_final_layer()

    def _initialize_final_layer(self):
        """Initialize the final linear layer with Xavier uniform."""
        final_layer = self.classifier[-1]
        nn.init.xavier_uniform_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Backbone features [B, feature_dim]

        Returns:
            Class logits [B, num_classes]
        """
        return self.classifier(features)
