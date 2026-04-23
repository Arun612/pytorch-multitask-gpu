"""
visualize.py — TensorBoard & Matplotlib Visualization
=======================================================
Demonstrates: TensorBoard SummaryWriter, image grids, loss curves, gradient flow plots
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
from typing import Dict, List, Optional


class TBLogger:
    """TensorBoard logger wrapper for structured logging."""

    def __init__(self, log_dir: str = "./runs"):
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📈 TensorBoard logging to: {log_dir}")
        print(f"   Run: tensorboard --logdir={log_dir}")

    def log_scalars(self, tag_value_dict: Dict[str, float], step: int):
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(tag, value, step)

    def log_learning_rate(self, lr: float, step: int):
        self.writer.add_scalar("Training/LearningRate", lr, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int, nrow: int = 8):
        """Log a grid of images. images: [B, C, H, W] tensor."""
        grid = torchvision.utils.make_grid(images[:16], nrow=nrow, normalize=True)
        self.writer.add_image(tag, grid, step)

    def log_reconstruction(self, originals: torch.Tensor, reconstructed: torch.Tensor, step: int):
        """Log original vs reconstructed images side by side."""
        n = min(8, originals.size(0))
        comparison = torch.cat([originals[:n], reconstructed[:n]], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=n, normalize=True)
        self.writer.add_image("Reconstruction/comparison", grid, step)

    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        try:
            self.writer.add_graph(model, input_tensor)
            print("📊 Model graph logged to TensorBoard")
        except Exception as e:
            print(f"⚠️ Could not log model graph: {e}")

    def close(self):
        self.writer.close()


def plot_training_history(history: Dict[str, List[float]], output_dir: str = "./outputs"):
    """Plot training curves and save to disk."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # Total loss
    ax = axes[0, 0]
    if "train_total_loss" in history:
        ax.plot(history["train_total_loss"], label="Train", linewidth=2)
    if "val_total_loss" in history:
        ax.plot(history["val_total_loss"], label="Val", linewidth=2)
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Classification loss
    ax = axes[0, 1]
    if "train_cls_loss" in history:
        ax.plot(history["train_cls_loss"], label="Train", linewidth=2)
    if "val_cls_loss" in history:
        ax.plot(history["val_cls_loss"], label="Val", linewidth=2)
    ax.set_title("Classification Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction loss
    ax = axes[1, 0]
    if "train_recon_loss" in history:
        ax.plot(history["train_recon_loss"], label="Train", linewidth=2)
    if "val_recon_loss" in history:
        ax.plot(history["val_recon_loss"], label="Val", linewidth=2)
    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1, 1]
    if "train_accuracy" in history:
        ax.plot(history["train_accuracy"], label="Train", linewidth=2)
    if "val_accuracy" in history:
        ax.plot(history["val_accuracy"], label="Val", linewidth=2)
    ax.set_title("Classification Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training history plot saved: {path}")


def plot_gradient_flow(named_parameters, output_dir: str = "./outputs"):
    """Plot gradient magnitudes per layer to detect vanishing/exploding gradients."""
    os.makedirs(output_dir, exist_ok=True)
    ave_grads, max_grads, layers = [], [], []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n.replace(".weight", ""))
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    if not layers:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.3), 6))
    ax.bar(range(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c", label="Max")
    ax.bar(range(len(ave_grads)), ave_grads, alpha=0.7, lw=1, color="b", label="Mean")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_xlabel("Layers")
    ax.set_ylabel("Gradient Magnitude")
    ax.set_title("Gradient Flow")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "gradient_flow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
