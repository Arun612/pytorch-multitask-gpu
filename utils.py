"""
utils.py — Utility Functions
==============================
Demonstrates: checkpointing, early stopping, reproducibility, parameter counting
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed set to {seed}")


class ModelCheckpoint:
    """Save and load model checkpoints using state_dict pattern."""

    def __init__(self, checkpoint_dir: str = "./checkpoints", verbose: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, metrics, scheduler=None, filename="checkpoint.pth"):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        if self.verbose:
            print(f"💾 Checkpoint saved: {path} (epoch {epoch})")

    def load(self, model, optimizer=None, scheduler=None, filename="checkpoint.pth", device=torch.device("cpu")):
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.verbose:
            print(f"📂 Checkpoint loaded: {path} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint

    def save_best(self, model, optimizer, epoch, metrics, scheduler=None):
        self.save(model, optimizer, epoch, metrics, scheduler, filename="best_model.pth")


class EarlyStopping:
    """Early stopping to prevent overfitting. Monitors validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"   📉 Val loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"   ⏳ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("   🛑 Early stopping triggered!")
                return True
        return False

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.should_stop = False


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


class AverageMeter:
    """Computes and stores the running average."""
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
