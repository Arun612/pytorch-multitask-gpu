"""
trainer.py — Full-Featured Training Engine
============================================
Demonstrates:
  - Complete training loop with train/val phases
  - Optimizer selection (Adam, SGD, AdamW)
  - Learning rate schedulers (CosineAnnealing, ReduceLROnPlateau)
  - Mixed precision training (AMP) with GradScaler
  - Gradient clipping (clip_grad_norm_)
  - Early stopping integration
  - model.train() / model.eval() / torch.no_grad()
  - TensorBoard logging during training
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict

from config import ProjectConfig
from losses import MultiTaskLoss
from utils import (
    ModelCheckpoint, EarlyStopping, AverageMeter, get_lr
)
from visualize import TBLogger, plot_gradient_flow


class Trainer:
    """
    Multi-task model trainer with all modern training techniques.

    Usage:
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader)
    """

    def __init__(self, model: nn.Module, config: ProjectConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device

        # ── Loss function ──
        self.criterion = MultiTaskLoss(
            cls_weight=config.training.classification_loss_weight,
            recon_weight=config.training.reconstruction_loss_weight,
            use_learnable_weights=config.training.use_learnable_loss_weights,
        ).to(self.device)

        # ── Optimizer ──
        self.optimizer = self._create_optimizer()

        # ── LR Scheduler ──
        self.scheduler = self._create_scheduler()

        # ── Mixed Precision (AMP) ──
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── Early Stopping ──
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
        )

        # ── Checkpointing ──
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.logging.checkpoint_dir,
        )

        # ── TensorBoard ──
        self.tb_logger = TBLogger(log_dir=config.logging.tensorboard_dir)

        # ── Training history ──
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.best_val_loss = float("inf")

        print(f"⚙️ Optimizer: {config.training.optimizer}")
        print(f"⚙️ Scheduler: {config.training.scheduler}")
        print(f"⚙️ AMP: {'enabled' if self.use_amp else 'disabled'}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config. Demonstrates optimizer selection."""
        cfg = self.config.training

        # Collect parameters — include loss parameters if learnable
        params = list(self.model.parameters())
        if cfg.use_learnable_loss_weights:
            params += list(self.criterion.parameters())

        if cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _create_scheduler(self):
        """Create LR scheduler based on config."""
        cfg = self.config.training
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=cfg.scheduler_min_lr,
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=cfg.scheduler_patience,
                min_lr=cfg.scheduler_min_lr, factor=0.5,
            )
        elif cfg.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    def fit(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """
        Main training loop.

        For each epoch:
            1. Train phase (model.train, forward, backward, optimizer.step)
            2. Validation phase (model.eval, torch.no_grad)
            3. LR scheduler step
            4. Early stopping check
            5. Checkpoint best model
            6. Log to TensorBoard
        """
        # Log model graph to TensorBoard
        sample = next(iter(train_loader))[0][:2].to(self.device)
        self.tb_logger.log_model_graph(self.model, sample)

        print(f"\n{'='*60}")
        print(f"  Starting Training — {self.config.training.epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, self.config.training.epochs + 1):
            # ── Train phase ──
            train_metrics = self._train_epoch(train_loader, epoch)

            # ── Validation phase ──
            val_metrics = self._validate_epoch(val_loader, epoch)

            # ── Log metrics ──
            self._log_epoch(epoch, train_metrics, val_metrics)

            # ── LR Scheduler step ──
            current_lr = get_lr(self.optimizer)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["total_loss"])
                else:
                    self.scheduler.step()

            self.tb_logger.log_learning_rate(current_lr, epoch)

            # ── Save best model ──
            if val_metrics["total_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total_loss"]
                self.checkpoint.save_best(
                    self.model, self.optimizer, epoch,
                    {"val_loss": val_metrics["total_loss"], "val_acc": val_metrics["accuracy"]},
                    self.scheduler,
                )

            # ── Early stopping ──
            if self.early_stopping(val_metrics["total_loss"]):
                print(f"\n🛑 Training stopped early at epoch {epoch}")
                break

            # ── Periodic checkpoint ──
            if epoch % 5 == 0:
                self.checkpoint.save(
                    self.model, self.optimizer, epoch,
                    {"val_loss": val_metrics["total_loss"]},
                    self.scheduler,
                    filename=f"checkpoint_epoch_{epoch}.pth",
                )

        self.tb_logger.close()
        return dict(self.history)

    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()  # Enable dropout, batch norm in training mode

        meters = {
            "total": AverageMeter("total"),
            "cls": AverageMeter("cls"),
            "recon": AverageMeter("recon"),
            "correct": 0,
            "total_samples": 0,
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, (images, labels, clean_images) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            clean_images = clean_images.to(self.device, non_blocking=True)

            # ── Forward pass with AMP ──
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                predictions = self.model(images)
                loss_dict = self.criterion(
                    predictions,
                    {"labels": labels, "images": clean_images},
                )

            # ── Backward pass ──
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict["total"]).backward()

            # ── Gradient clipping ──
            if self.config.training.use_gradient_clipping:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_value,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Track metrics ──
            batch_size = images.size(0)
            meters["total"].update(loss_dict["total"].item(), batch_size)
            meters["cls"].update(loss_dict["classification"].item(), batch_size)
            meters["recon"].update(loss_dict["reconstruction"].item(), batch_size)

            _, predicted = predictions["classification"].max(1)
            meters["correct"] += predicted.eq(labels).sum().item()
            meters["total_samples"] += batch_size

            pbar.set_postfix({
                "loss": f"{meters['total'].avg:.4f}",
                "acc": f"{100. * meters['correct'] / meters['total_samples']:.1f}%",
            })

            # ── TensorBoard batch logging ──
            global_step = (epoch - 1) * len(train_loader) + batch_idx
            if batch_idx % self.config.logging.log_interval == 0:
                self.tb_logger.log_scalars({
                    "Batch/train_total_loss": loss_dict["total"].item(),
                    "Batch/train_cls_loss": loss_dict["classification"].item(),
                }, global_step)

        accuracy = 100. * meters["correct"] / meters["total_samples"]
        return {
            "total_loss": meters["total"].avg,
            "cls_loss": meters["cls"].avg,
            "recon_loss": meters["recon"].avg,
            "accuracy": accuracy,
        }

    @torch.no_grad()
    def _validate_epoch(self, val_loader, epoch: int) -> Dict[str, float]:
        """Single validation epoch — no gradients computed."""
        self.model.eval()  # Disable dropout, use running stats for batch norm

        meters = {
            "total": AverageMeter("total"),
            "cls": AverageMeter("cls"),
            "recon": AverageMeter("recon"),
            "correct": 0,
            "total_samples": 0,
        }

        first_batch_logged = False

        for images, labels, clean_images in val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            clean_images = clean_images.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                predictions = self.model(images)
                loss_dict = self.criterion(
                    predictions, {"labels": labels, "images": clean_images},
                )

            batch_size = images.size(0)
            meters["total"].update(loss_dict["total"].item(), batch_size)
            meters["cls"].update(loss_dict["classification"].item(), batch_size)
            meters["recon"].update(loss_dict["reconstruction"].item(), batch_size)

            _, predicted = predictions["classification"].max(1)
            meters["correct"] += predicted.eq(labels).sum().item()
            meters["total_samples"] += batch_size

            # Log reconstruction comparison (first batch only)
            if not first_batch_logged:
                self.tb_logger.log_reconstruction(
                    clean_images, predictions["reconstruction"], epoch,
                )
                first_batch_logged = True

        accuracy = 100. * meters["correct"] / meters["total_samples"]
        return {
            "total_loss": meters["total"].avg,
            "cls_loss": meters["cls"].avg,
            "recon_loss": meters["recon"].avg,
            "accuracy": accuracy,
        }

    def _log_epoch(self, epoch, train_m, val_m):
        """Log epoch metrics to console, history, and TensorBoard."""
        # Console
        print(f"Epoch {epoch:>3d} | "
              f"Train Loss: {train_m['total_loss']:.4f} Acc: {train_m['accuracy']:.1f}% | "
              f"Val Loss: {val_m['total_loss']:.4f} Acc: {val_m['accuracy']:.1f}% | "
              f"LR: {get_lr(self.optimizer):.6f}")

        # History
        for key in ["total_loss", "cls_loss", "recon_loss", "accuracy"]:
            self.history[f"train_{key}"].append(train_m[key])
            self.history[f"val_{key}"].append(val_m[key])

        # TensorBoard
        self.tb_logger.log_scalars({
            "Epoch/train_total_loss": train_m["total_loss"],
            "Epoch/val_total_loss": val_m["total_loss"],
            "Epoch/train_cls_loss": train_m["cls_loss"],
            "Epoch/val_cls_loss": val_m["cls_loss"],
            "Epoch/train_recon_loss": train_m["recon_loss"],
            "Epoch/val_recon_loss": val_m["recon_loss"],
            "Epoch/train_accuracy": train_m["accuracy"],
            "Epoch/val_accuracy": val_m["accuracy"],
        }, epoch)
