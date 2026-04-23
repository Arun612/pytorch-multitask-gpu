"""
inference.py — Single Image Inference Pipeline
================================================
Demonstrates: loading a saved model, preprocessing, inference with no_grad, displaying results
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from typing import Tuple

from config import ProjectConfig
from models import MultiTaskModel


class InferencePipeline:
    """
    End-to-end inference pipeline: load model → preprocess → predict.

    Usage:
        pipeline = InferencePipeline.from_checkpoint("checkpoints/best_model.pth", config)
        result = pipeline.predict(image_path_or_tensor)
    """

    def __init__(self, model: nn.Module, config: ProjectConfig):
        self.model = model.to(config.device)
        self.model.eval()  # Always in eval mode for inference
        self.device = config.device
        self.config = config

        # Preprocessing transform (same as test-time transform)
        self.transform = T.Compose([
            T.Resize(config.data.image_size),
            T.ToTensor(),
            T.Normalize(config.data.normalize_mean, config.data.normalize_std),
        ])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: ProjectConfig) -> "InferencePipeline":
        """Load model from a saved checkpoint."""
        model = MultiTaskModel.from_config(config.model, config.data)

        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"📂 Loaded model from: {checkpoint_path}")
        print(f"   Epoch: {checkpoint.get('epoch', '?')}, "
              f"Metrics: {checkpoint.get('metrics', {})}")

        return cls(model, config)

    @torch.no_grad()
    def predict(self, image) -> dict:
        """
        Run inference on a single image.

        Args:
            image: PIL Image, file path string, or tensor [C, H, W]

        Returns:
            Dict with predicted class, probabilities, and reconstructed image
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dim
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                input_tensor = image.unsqueeze(0)
            else:
                input_tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        input_tensor = input_tensor.to(self.device)

        # Forward pass
        predictions = self.model(input_tensor)

        # Process classification output
        logits = predictions["classification"][0]
        probabilities = torch.softmax(logits, dim=0)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()

        # Get top-5 predictions
        top5_probs, top5_indices = probabilities.topk(5)
        top5 = [
            (self.config.data.class_names[idx.item()], prob.item())
            for idx, prob in zip(top5_indices, top5_probs)
        ]

        return {
            "predicted_class": self.config.data.class_names[predicted_class],
            "confidence": confidence,
            "top5": top5,
            "probabilities": probabilities.cpu().numpy(),
            "reconstructed": predictions["reconstruction"][0].cpu(),
        }

    def predict_and_visualize(self, image, output_path: str = None):
        """Predict and create a visualization of the result."""
        # Get original image for display
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = None

        result = self.predict(image)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle("Inference Result", fontsize=14, fontweight="bold")

        # Original image
        if pil_image:
            axes[0].imshow(pil_image)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Reconstructed image
        recon = result["reconstructed"].permute(1, 2, 0).numpy()
        recon = np.clip(recon, 0, 1)
        axes[1].imshow(recon)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")

        # Top-5 predictions bar chart
        names = [t[0] for t in result["top5"]]
        probs = [t[1] * 100 for t in result["top5"]]
        colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(names))]
        axes[2].barh(names[::-1], probs[::-1], color=colors[::-1])
        axes[2].set_xlabel("Confidence (%)")
        axes[2].set_title(f"Predicted: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
        axes[2].set_xlim(0, 100)

        plt.tight_layout()

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"📊 Inference visualization saved: {output_path}")
        plt.close()

        return result
