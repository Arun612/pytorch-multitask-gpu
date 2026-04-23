"""
main.py — Entry Point: Orchestrates the Entire Pipeline
=========================================================
This is the master script that ties everything together:
    1. Configure → 2. Load Data → 3. Build Model → 4. Train
    → 5. Evaluate → 6. Inference Demo → 7. Visualize

Run:  python main.py

PyTorch Concepts Demonstrated Across This Project:
───────────────────────────────────────────────────
 1. Tensors & Operations        — dataset.py (custom transforms)
 2. Autograd & Comp. Graph      — losses.py (GradientAnalyzer, backward hooks)
 3. nn.Module & Custom Layers   — all model files
 4. Custom Dataset & DataLoader — dataset.py
 5. CNN Architectures           — models/backbone.py (CustomCNNBackbone)
 6. Transfer Learning           — models/backbone.py (PretrainedBackbone)
 7. Custom Loss Functions       — losses.py (MultiTaskLoss)
 8. Optimizers (Adam/SGD/AdamW) — trainer.py
 9. LR Schedulers               — trainer.py (Cosine, Plateau)
10. Regularization              — models/ (Dropout, BatchNorm, WeightDecay)
11. Model Checkpointing         — utils.py (save/load state_dict)
12. TensorBoard Integration     — visualize.py, trainer.py
13. Mixed Precision (AMP)       — trainer.py (GradScaler, autocast)
14. Gradient Clipping           — trainer.py (clip_grad_norm_)
15. Early Stopping              — utils.py, trainer.py
16. Multi-Task Learning         — models/multitask_model.py
17. Model Evaluation & Metrics  — evaluate.py
18. Inference Pipeline          — inference.py
"""

import torch
import sys
import os

from config import ProjectConfig
from dataset import create_data_loaders
from models import MultiTaskModel
from trainer import Trainer
from evaluate import evaluate_model
from inference import InferencePipeline
from visualize import plot_training_history, plot_gradient_flow
from losses import GradientAnalyzer
from utils import seed_everything, count_parameters


def main():
    """Main entry point — runs the complete pipeline."""

    # ╔══════════════════════════════════════════╗
    # ║  STEP 1: Configuration                  ║
    # ╚══════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("  🔥 PyTorch Multi-Task Image Intelligence System")
    print("=" * 60)

    config = ProjectConfig()
    print(config.summary())

    # Set random seed for reproducibility
    seed_everything(config.training.seed)

    # ╔══════════════════════════════════════════╗
    # ║  STEP 2: Load Data                      ║
    # ╚══════════════════════════════════════════╝
    print("\n📦 STEP 2: Loading Data...")
    data_loaders = create_data_loaders(config.data)
    train_loader = data_loaders["train"]
    test_loader = data_loaders["test"]

    # Quick data inspection
    sample_images, sample_labels, sample_clean = next(iter(train_loader))
    print(f"   Sample batch shape: {sample_images.shape}")
    print(f"   Labels shape: {sample_labels.shape}")
    print(f"   Clean images shape: {sample_clean.shape}")
    print(f"   Label range: [{sample_labels.min()}, {sample_labels.max()}]")
    print(f"   Image value range: [{sample_images.min():.2f}, {sample_images.max():.2f}]")

    # ╔══════════════════════════════════════════╗
    # ║  STEP 3: Build Model                    ║
    # ╚══════════════════════════════════════════╝
    print("\n🧠 STEP 3: Building Model...")
    model = MultiTaskModel.from_config(config.model, config.data)

    # Verify model with a forward pass
    model.to(config.device)
    test_input = sample_images[:2].to(config.device)
    test_output = model(test_input)
    print(f"   ✅ Forward pass OK:")
    print(f"      Classification output: {test_output['classification'].shape}")
    print(f"      Reconstruction output: {test_output['reconstruction'].shape}")
    print(f"      Features shape:        {test_output['features'].shape}")

    print(f"\n   Total parameters:     {count_parameters(model, trainable_only=False):>10,}")
    print(f"   Trainable parameters: {count_parameters(model, trainable_only=True):>10,}")

    # ╔══════════════════════════════════════════╗
    # ║  STEP 4: Train the Model                ║
    # ╚══════════════════════════════════════════╝
    print("\n🏋️ STEP 4: Training...")
    trainer = Trainer(model, config)
    history = trainer.fit(train_loader, test_loader)

    # ── Plot training history ──
    plot_training_history(history, config.logging.output_dir)

    # ── Gradient flow analysis (one final pass) ──
    print("\n🔬 Analyzing gradient flow...")
    model.train()
    analyzer = GradientAnalyzer(model)
    sample_input = sample_images[:4].to(config.device)
    sample_labels_dev = sample_labels[:4].to(config.device)
    sample_clean_dev = sample_clean[:4].to(config.device)

    output = model(sample_input)
    from losses import MultiTaskLoss
    temp_loss_fn = MultiTaskLoss().to(config.device)
    loss = temp_loss_fn(output, {"labels": sample_labels_dev, "images": sample_clean_dev})
    loss["total"].backward()

    health = analyzer.check_gradient_health()
    print("   Gradient Health Check:")
    for layer, status in list(health.items())[:10]:
        print(f"      {layer:>30s}: {status}")
    analyzer.remove_hooks()

    plot_gradient_flow(model.named_parameters(), config.logging.output_dir)

    # ╔══════════════════════════════════════════╗
    # ║  STEP 5: Evaluate on Test Set           ║
    # ╚══════════════════════════════════════════╝
    print("\n📊 STEP 5: Evaluating...")

    # Load best model for evaluation
    best_path = os.path.join(config.logging.checkpoint_dir, "best_model.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   Loaded best model from epoch {checkpoint.get('epoch', '?')}")

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=config.device,
        class_names=config.data.class_names,
        output_dir=config.logging.output_dir,
        use_amp=config.training.use_amp,
    )

    # ╔══════════════════════════════════════════╗
    # ║  STEP 6: Inference Demo                 ║
    # ╚══════════════════════════════════════════╝
    print("\n🔮 STEP 6: Inference Demo...")

    pipeline = InferencePipeline(model, config)

    # Run inference on a few test samples
    test_dataset = test_loader.dataset
    for i in range(3):
        image, label, _ = test_dataset[i]
        result = pipeline.predict(image)
        true_class = config.data.class_names[label]
        pred_class = result["predicted_class"]
        conf = result["confidence"] * 100
        status = "✅" if true_class == pred_class else "❌"
        print(f"   Sample {i+1}: True={true_class:>10s} | "
              f"Pred={pred_class:>10s} ({conf:.1f}%) {status}")

    # Visualize one inference
    image, label, _ = test_dataset[0]
    pipeline.predict_and_visualize(
        image,
        output_path=os.path.join(config.logging.output_dir, "inference_demo.png"),
    )

    # ╔══════════════════════════════════════════╗
    # ║  STEP 7: Summary                        ║
    # ╚══════════════════════════════════════════╝
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  📊 Test Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"  💾 Best model:    {best_path}")
    print(f"  📈 TensorBoard:   tensorboard --logdir={config.logging.tensorboard_dir}")
    print(f"  📁 Outputs:       {config.logging.output_dir}/")
    print(f"\n  Generated files:")
    print(f"    • training_history.png")
    print(f"    • confusion_matrix.png")
    print(f"    • reconstruction_comparison.png")
    print(f"    • gradient_flow.png")
    print(f"    • inference_demo.png")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
