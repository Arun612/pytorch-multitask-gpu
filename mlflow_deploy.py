"""
mlflow_deploy.py — MLflow Integration for MultiTask Model
===========================================================
Usage:
    1. Train + Log:    python mlflow_deploy.py --mode train
    2. Launch UI:      mlflow ui --port 5000
    3. Serve Model:    python mlflow_deploy.py --mode serve
"""

import os
import sys
import argparse
import torch
import numpy as np
import mlflow
import mlflow.pytorch
import mlflow.pyfunc

from config import ProjectConfig
from models import MultiTaskModel
from dataset import create_data_loaders
from trainer import Trainer
from evaluate import evaluate_model
from utils import seed_everything


class MultiTaskModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps our MultiTaskModel for MLflow serving.
    Converts HTTP input → tensor → model prediction → JSON output.
    """
    
    def load_context(self, context):
        """Called once when the model is loaded for serving."""
        import torch
        # Load the PyTorch model from the artifacts
        self.device = torch.device("cpu")  # Serve on CPU (or GPU if available)
        self.model = mlflow.pytorch.load_model(
            context.artifacts["pytorch_model"]
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = (
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        )
    
    def predict(self, context, model_input):
        """
        Called for each prediction request.
        
        Args:
            model_input: pandas DataFrame with image data
                         (flattened pixel values or base64 encoded)
        
        Returns:
            Dictionary with predictions
        """
        import torch
        import numpy as np
        
        # Convert input to tensor [B, 3, 32, 32]
        if hasattr(model_input, 'values'):
            input_array = model_input.values
        else:
            input_array = np.array(model_input)
        
        input_tensor = torch.tensor(
            input_array.reshape(-1, 3, 32, 32), 
            dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Process classification results
        probs = torch.softmax(output["classification"], dim=1)
        predicted_classes = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        
        results = []
        for i in range(len(predicted_classes)):
            results.append({
                "predicted_class": self.class_names[predicted_classes[i].item()],
                "confidence": round(confidences[i].item(), 4),
                "all_probabilities": {
                    name: round(probs[i][j].item(), 4) 
                    for j, name in enumerate(self.class_names)
                },
            })
        
        return results

def train_and_log():
    """Train the model and log everything to MLflow."""
    
    config = ProjectConfig()
    seed_everything(config.training.seed)
    
    # ── MLflow Setup ──
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("multitask-cifar10")
    
    with mlflow.start_run(run_name=f"{config.model.backbone_type}-run"):
        
        # Log all hyperparameters
        mlflow.log_params({
            "backbone": config.model.backbone_type,
            "pretrained": config.model.pretrained,
            "optimizer": config.training.optimizer,
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "batch_size": config.data.batch_size,
            "epochs": config.training.epochs,
            "scheduler": config.training.scheduler,
            "use_amp": config.training.use_amp,
            "cls_weight": config.training.classification_loss_weight,
            "recon_weight": config.training.reconstruction_loss_weight,
        })
        
        # Load data
        data_loaders = create_data_loaders(config.data)
        train_loader = data_loaders["train"]
        test_loader = data_loaders["test"]
        
        # Build model
        model = MultiTaskModel.from_config(config.model, config.data)
        model.to(config.device)
        
        # Train
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, test_loader)
        
        # Log per-epoch metrics
        for epoch_idx in range(len(history["train_total_loss"])):
            mlflow.log_metrics({
                "train_loss": history["train_total_loss"][epoch_idx],
                "val_loss": history["val_total_loss"][epoch_idx],
                "train_acc": history["train_accuracy"][epoch_idx],
                "val_acc": history["val_accuracy"][epoch_idx],
            }, step=epoch_idx + 1)
        
        # Load best model
        best_path = os.path.join(config.logging.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=config.device,
            class_names=config.data.class_names,
            output_dir=config.logging.output_dir,
            use_amp=config.training.use_amp,
        )
        
        mlflow.log_metric("test_accuracy", results["overall_accuracy"])
        
        # Log output artifacts (plots)
        if os.path.exists(config.logging.output_dir):
            mlflow.log_artifacts(config.logging.output_dir, artifact_path="plots")
        
        # Save PyTorch model temporarily
        pytorch_model_path = "mlruns/temp_model"
        mlflow.pytorch.save_model(model, pytorch_model_path)

        # Log the PythonModel wrapper
        mlflow.pyfunc.log_model(
            artifact_path="multitask-served",
            python_model=MultiTaskModelWrapper(),
            artifacts={"pytorch_model": pytorch_model_path},
            registered_model_name="MultiTaskCIFAR10",
            pip_requirements=[
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "numpy>=1.24.0",
                "mlflow>=2.10.0",
                "pandas",
            ],
        )
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n{'='*60}")
        print(f"  ✅ MLflow Run Complete!")
        print(f"  📊 Run ID: {run_id}")
        print(f"  🎯 Test Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"  🌐 View at: mlflow ui --port 5000")
        print(f"{'='*60}")


def serve_model():
    """Serve the latest registered model."""
    print("🚀 Starting MLflow Model Server...")
    print("   Send requests to: http://localhost:5001/invocations")
    os.system(
        'mlflow models serve '
        '--model-uri "models:/MultiTaskCIFAR10/latest" '
        '--port 5001 '
        '--no-conda'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "serve"], default="train")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_and_log()
    elif args.mode == "serve":
        serve_model()
