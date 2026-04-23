"""
losses.py — Custom Loss Functions & Gradient Analysis
=======================================================
Demonstrates:
  - Custom loss functions inheriting nn.Module
  - Combining multiple losses with learnable weights
  - Autograd: backward hooks for gradient inspection
  - Computational graph analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from collections import defaultdict


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Total Loss = α * Classification_Loss + β * Reconstruction_Loss

    Two modes:
        1. Fixed weights: α and β are hyperparameters
        2. Learnable weights: Uses uncertainty weighting (Kendall et al., 2018)
           to automatically balance task losses

    The learnable approach models each task's homoscedastic uncertainty
    and optimizes:  L = (1/2σ²₁) * L₁ + (1/2σ²₂) * L₂ + log(σ₁) + log(σ₂)
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        recon_weight: float = 0.5,
        use_learnable_weights: bool = False,
    ):
        """
        Args:
            cls_weight: Fixed weight for classification loss
            recon_weight: Fixed weight for reconstruction loss
            use_learnable_weights: If True, learn the loss weights during training
        """
        super().__init__()

        self.use_learnable_weights = use_learnable_weights

        if use_learnable_weights:
            # Learnable log-variance parameters (one per task)
            # We learn log(σ²) for numerical stability
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_recon = nn.Parameter(torch.zeros(1))
            print("📊 Using learnable loss weights (uncertainty weighting)")
        else:
            self.cls_weight = cls_weight
            self.recon_weight = recon_weight
            print(f"📊 Using fixed loss weights: "
                  f"cls={cls_weight}, recon={recon_weight}")

        # Individual loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.reconstruction_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Model outputs dict with 'classification' and 'reconstruction'
            targets: Ground truth dict with 'labels' and 'images'

        Returns:
            Dictionary with:
                'total': weighted combined loss (for backward())
                'classification': raw classification loss
                'reconstruction': raw reconstruction loss
        """
        # Individual task losses
        cls_loss = self.classification_loss(
            predictions["classification"],
            targets["labels"],
        )

        recon_loss = self.reconstruction_loss(
            predictions["reconstruction"],
            targets["images"],
        )

        # Combine losses
        if self.use_learnable_weights:
            # Uncertainty weighting: L = (1/2σ²) * L_task + log(σ)
            # Since we learn log(σ²), precision = exp(-log_var)
            cls_precision = torch.exp(-self.log_var_cls)
            recon_precision = torch.exp(-self.log_var_recon)

            total_loss = (
                cls_precision * cls_loss + self.log_var_cls +
                recon_precision * recon_loss + self.log_var_recon
            )
        else:
            total_loss = (
                self.cls_weight * cls_loss +
                self.recon_weight * recon_loss
            )

        return {
            "total": total_loss,
            "classification": cls_loss.detach(),
            "reconstruction": recon_loss.detach(),
        }


class GradientAnalyzer:
    """
    Utility to inspect gradients flowing through the network.

    Demonstrates:
        - register_backward_hook (now register_full_backward_hook)
        - Autograd and the computational graph
        - Gradient magnitude monitoring (for detecting vanishing/exploding gradients)

    Usage:
        analyzer = GradientAnalyzer(model)
        output = model(input)
        loss.backward()
        stats = analyzer.get_gradient_stats()
        analyzer.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        """
        Register hooks on all layers to capture gradients during backward pass.
        """
        self.gradient_stats = defaultdict(list)
        self.hooks = []

        for name, module in model.named_modules():
            # Only hook into "interesting" layers
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                hook = module.register_full_backward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """
        Create a backward hook function for a specific layer.

        Backward hooks receive:
            module: the layer
            grad_input: gradients w.r.t. the layer's input
            grad_output: gradients w.r.t. the layer's output
        """
        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0].detach()
                self.gradient_stats[layer_name].append({
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.abs().max().item(),
                    "min": grad.abs().min().item(),
                    "norm": grad.norm().item(),
                })
        return hook_fn

    def get_gradient_stats(self) -> Dict:
        """Return captured gradient statistics for all hooked layers."""
        return dict(self.gradient_stats)

    def get_latest_stats(self) -> Dict:
        """Return only the latest gradient stats."""
        return {
            name: stats[-1] if stats else None
            for name, stats in self.gradient_stats.items()
        }

    def check_gradient_health(self) -> Dict[str, str]:
        """
        Analyze gradient health — detect vanishing or exploding gradients.

        Returns:
            Dict mapping layer names to health status strings
        """
        health = {}
        for name, stats_list in self.gradient_stats.items():
            if not stats_list:
                health[name] = "no_data"
                continue

            latest = stats_list[-1]
            norm = latest["norm"]

            if norm < 1e-7:
                health[name] = "⚠️ VANISHING"
            elif norm > 1e3:
                health[name] = "🔥 EXPLODING"
            else:
                health[name] = "✅ healthy"

        return health

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
