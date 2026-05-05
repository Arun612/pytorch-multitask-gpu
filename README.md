# PyTorch Multi-Task Image Intelligence System

A production-structured PyTorch project that trains a **multi-task CNN** on the CIFAR-10 dataset, simultaneously performing **image classification** and **image reconstruction** (autoencoder). The repo is built as an educational showcase of 18 core PyTorch concepts, covering everything from raw tensor operations to mixed-precision training and MLflow deployment.

---

## Features

- **Dual backbone support** — choose between a custom CNN built from scratch or a pretrained ResNet-18 (transfer learning)
- **Multi-task learning** — shared feature extraction feeds both a classifier head and a decoder/reconstruction head
- **Learnable loss weighting** — optionally uses uncertainty-based (Kendall et al. 2018) automatic task balancing
- **Modern training techniques** — AdamW optimizer, CosineAnnealing LR scheduler, AMP mixed precision, gradient clipping, and early stopping
- **Gradient health monitoring** — backward hooks detect vanishing/exploding gradients during training
- **Full observability** — TensorBoard logging, training history plots, confusion matrix, gradient flow visualisation
- **MLflow integration** — experiment tracking and model registry via `mlflow_deploy.py`
- **Clean configuration** — all hyperparameters live in typed Python dataclasses in `config.py`; zero magic numbers scattered across the codebase

---

## Project Structure

```
PyTorch-CNN/
├── config.py           # Centralised config via dataclasses (DataConfig, ModelConfig, TrainingConfig, LoggingConfig)
├── dataset.py          # Custom Dataset + DataLoader with augmentation pipeline
├── losses.py           # MultiTaskLoss (fixed & learnable weights) + GradientAnalyzer with backward hooks
├── trainer.py          # Full training engine — train/val loop, AMP, gradient clipping, early stopping, TensorBoard
├── evaluate.py         # Post-training evaluation — accuracy, confusion matrix, reconstruction comparison
├── inference.py        # InferencePipeline for single-image prediction with confidence scores
├── visualize.py        # TBLogger wrapper + matplotlib plots (history, gradient flow, reconstructions)
├── utils.py            # ModelCheckpoint, EarlyStopping, AverageMeter, seed_everything, count_parameters
├── main.py             # Orchestration entry point — runs all 7 pipeline stages end to end
├── mlflow_deploy.py    # MLflow experiment tracking and model deployment helper
├── models/
│   ├── backbone.py         # CustomCNNBackbone (4-stage conv) and PretrainedBackbone (ResNet-18)
│   ├── multitask_model.py  # MultiTaskModel combining backbone + classifier head + decoder head
│   └── ...
├── mlruns/             # MLflow run artefacts (auto-generated)
├── requirements.txt
└── .gitignore
```

---

## PyTorch Concepts Demonstrated

| # | Concept | Where |
|---|---------|-------|
| 1 | Tensors & Operations | `dataset.py` (custom transforms) |
| 2 | Autograd & Computational Graph | `losses.py` (GradientAnalyzer, backward hooks) |
| 3 | `nn.Module` & Custom Layers | All model files |
| 4 | Custom Dataset & DataLoader | `dataset.py` |
| 5 | CNN Architectures | `models/backbone.py` (CustomCNNBackbone) |
| 6 | Transfer Learning | `models/backbone.py` (PretrainedBackbone / ResNet-18) |
| 7 | Custom Loss Functions | `losses.py` (MultiTaskLoss) |
| 8 | Optimizers (Adam / SGD / AdamW) | `trainer.py` |
| 9 | LR Schedulers | `trainer.py` (Cosine, Plateau) |
| 10 | Regularization | `models/` (Dropout, BatchNorm, WeightDecay) |
| 11 | Model Checkpointing | `utils.py` (`save`/`load` state_dict) |
| 12 | TensorBoard Integration | `visualize.py`, `trainer.py` |
| 13 | Mixed Precision (AMP) | `trainer.py` (GradScaler, autocast) |
| 14 | Gradient Clipping | `trainer.py` (`clip_grad_norm_`) |
| 15 | Early Stopping | `utils.py`, `trainer.py` |
| 16 | Multi-Task Learning | `models/multitask_model.py` |
| 17 | Model Evaluation & Metrics | `evaluate.py` |
| 18 | Inference Pipeline | `inference.py` |

---

## Installation

```bash
git clone https://github.com/Arun612/PyTorch-CNN.git
cd PyTorch-CNN
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- torch >= 2.0.0
- torchvision >= 0.15.0
- tensorboard >= 2.13.0
- mlflow >= 2.10.0
- scikit-learn, matplotlib, tqdm, Pillow, numpy, pandas

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This runs all 7 stages in sequence:

1. **Configuration** — instantiates `ProjectConfig`, auto-detects device (CUDA/CPU), creates directories
2. **Data Loading** — downloads CIFAR-10, applies augmentations, returns train/test DataLoaders
3. **Model Building** — constructs `MultiTaskModel`, verifies a forward pass, prints parameter counts
4. **Training** — runs the `Trainer.fit()` loop for up to 30 epochs with early stopping
5. **Evaluation** — loads best checkpoint, computes accuracy, plots confusion matrix
6. **Inference Demo** — runs `InferencePipeline` on 3 test samples and saves a visualisation
7. **Summary** — prints final accuracy and paths to all generated outputs

### Monitor training

```bash
tensorboard --logdir=./runs
```

### Track experiments with MLflow

```bash
python mlflow_deploy.py
mlflow ui   # opens at http://localhost:5000
```

---

## Configuration

All settings are in `config.py`. Key knobs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone_type` | `"resnet18"` | `"custom_cnn"` or `"resnet18"` |
| `pretrained` | `True` | Use ImageNet pretrained weights |
| `epochs` | `30` | Max training epochs |
| `optimizer` | `"adamw"` | `"adam"`, `"sgd"`, or `"adamw"` |
| `learning_rate` | `1e-3` | Initial LR |
| `scheduler` | `"cosine"` | `"cosine"`, `"plateau"`, or `"none"` |
| `use_amp` | `True` | Mixed precision (disabled on CPU automatically) |
| `use_learnable_loss_weights` | `False` | Uncertainty-based automatic loss balancing |
| `early_stopping_patience` | `10` | Epochs to wait before stopping |

---

## Outputs

After a full run, the `./outputs/` directory contains:

- `training_history.png` — train/val loss and accuracy curves
- `confusion_matrix.png` — per-class classification performance
- `reconstruction_comparison.png` — side-by-side original vs reconstructed images
- `gradient_flow.png` — gradient magnitude per layer
- `inference_demo.png` — single-image prediction visualisation

Checkpoints are saved to `./checkpoints/best_model.pth`.

---

## Dataset

**CIFAR-10** — 60,000 32×32 RGB images across 10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

Downloaded automatically via `torchvision.datasets.CIFAR10` on first run.

---

## License

This project is open source. See the repository for details.
