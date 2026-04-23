"""
evaluate.py — Model Evaluation & Metrics
==========================================
Demonstrates: per-class accuracy, confusion matrix, classification report,
              reconstruction quality visualization
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, classification_report
import os
from typing import Dict, List, Tuple


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    class_names: Tuple[str, ...],
    output_dir: str = "./outputs",
    use_amp: bool = False,
) -> Dict:
    """
    Comprehensive model evaluation on the test set.

    Returns:
        Dictionary with overall accuracy, per-class accuracy,
        confusion matrix, and classification report.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_preds = []
    all_labels = []
    all_originals = []
    all_reconstructed = []
    correct = 0
    total = 0

    for images, labels, clean_images in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_images = clean_images.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            predictions = model(images)

        _, predicted = predictions["classification"].max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Store first batch for reconstruction visualization
        if len(all_originals) == 0:
            all_originals = clean_images[:16].cpu()
            all_reconstructed = predictions["reconstruction"][:16].cpu()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall_accuracy = 100. * correct / total

    # ── Per-class accuracy ──
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = 100. * (all_preds[mask] == i).sum() / mask.sum()

    # ── Confusion matrix ──
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names, output_dir)

    # ── Classification report ──
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # ── Reconstruction visualization ──
    if len(all_originals) > 0:
        _plot_reconstruction(all_originals, all_reconstructed, output_dir)

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"\n  Per-Class Accuracy:")
    for name, acc in per_class_acc.items():
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"    {name:>12s}: {bar} {acc:.1f}%")
    print(f"\n{report}")
    print(f"{'='*60}\n")

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def _plot_confusion_matrix(cm: np.ndarray, class_names: Tuple[str, ...], output_dir: str):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved: {path}")


def _plot_reconstruction(originals: torch.Tensor, reconstructed: torch.Tensor, output_dir: str):
    """Plot original vs reconstructed images."""
    n = min(8, originals.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    fig.suptitle("Original (top) vs Reconstructed (bottom)", fontsize=14)

    for i in range(n):
        # Original
        img = originals[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis("off")

        # Reconstructed
        rec = reconstructed[i].permute(1, 2, 0).numpy()
        rec = np.clip(rec, 0, 1)
        axes[1, i].imshow(rec)
        axes[1, i].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "reconstruction_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Reconstruction comparison saved: {path}")
