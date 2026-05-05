# PyTorch-CNN — System Architecture & Workflow

This document describes the end-to-end architecture of the project: how data flows, how the model is structured, and how every module connects to form a complete ML pipeline.

---

## High-Level Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐     ┌───────────────┐
│  config.py  │────▶│  dataset.py  │────▶│  models/          │────▶│  trainer.py   │
│ ProjectConfig│    │ DataLoader   │     │  MultiTaskModel   │     │  Trainer.fit()│
└─────────────┘     └──────────────┘     └───────────────────┘     └───────┬───────┘
                                                                            │
                    ┌───────────────────────────────────────────────────────▼────────┐
                    │                  POST-TRAINING                                  │
                    │  evaluate.py ──▶ confusion matrix, accuracy                    │
                    │  inference.py ──▶ single-image prediction pipeline             │
                    │  visualize.py ──▶ TensorBoard + matplotlib outputs             │
                    │  mlflow_deploy.py ──▶ experiment tracking + model registry     │
                    └────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Configuration (`config.py`)

A single `ProjectConfig` dataclass composes four sub-configs. On instantiation it auto-detects the device and creates all required directories.

```
ProjectConfig
├── DataConfig      — data_dir, batch_size, image_size, class_names, augmentation params,
│                     normalisation mean/std
├── ModelConfig     — backbone_type, pretrained, feature_dim, classifier_hidden_dims,
│                     dropout, decoder_hidden_dim, custom_cnn_channels
├── TrainingConfig  — epochs, optimizer, lr, weight_decay, scheduler, AMP flag,
│                     gradient_clip_value, early_stopping_patience, loss weights
└── LoggingConfig   — checkpoint_dir, tensorboard_dir, output_dir, log_interval
```

---

## Stage 2 — Data Pipeline (`dataset.py`)

```
CIFAR-10 Raw Data (auto-downloaded)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Training Augmentation Transform                       │
│  RandomCrop(32, padding=4) → RandomHorizontalFlip     │
│  → ColorJitter(brightness, contrast)                  │
│  → ToTensor → Normalize(mean, std)                    │
└──────────────────────┬────────────────────────────────┘
                       │                     │ (also returns clean image for reconstruction target)
                       ▼                     ▼
              augmented_image          clean_image
                       │                     │
                       └──────────┬──────────┘
                                  ▼
                    Custom Dataset __getitem__
                    returns: (augmented_img, label, clean_img)
                                  │
                                  ▼
                    DataLoader (batch_size=64, num_workers=2,
                                pin_memory=True, shuffle=True)
```

Each batch therefore contains three tensors: the augmented input, the ground-truth label, and the clean image used as the reconstruction target.

---

## Stage 3 — Model Architecture (`models/`)

### 3a. Backbone (Feature Extractor)

Two backbone options share the same interface:

```
Option A: CustomCNNBackbone
──────────────────────────
Input [B, 3, 32, 32]
  └─▶ ConvBlock-1  (3 → 32,  BN, ReLU, MaxPool) → [B, 32,  16, 16]
  └─▶ ConvBlock-2  (32 → 64, BN, ReLU, MaxPool) → [B, 64,   8,  8]
  └─▶ ConvBlock-3  (64 → 128,BN, ReLU, MaxPool) → [B, 128,  4,  4]
  └─▶ ConvBlock-4  (128→ 256,BN, ReLU, MaxPool) → [B, 256,  2,  2]
  └─▶ AdaptiveAvgPool2d(1,1) + Flatten
Output: feature vector [B, 256]

Option B: PretrainedBackbone (ResNet-18, ImageNet weights)
───────────────────────────────────────────────────────────
Input [B, 3, 32, 32]
  └─▶ ResNet-18 (pretrained, final FC removed)
  └─▶ AdaptiveAvgPool2d
Output: feature vector [B, 512]
```

### 3b. MultiTaskModel (Full Model)

```
Input Image [B, 3, 32, 32]
        │
        ▼
   ┌──────────┐
   │ Backbone │  (CustomCNN or ResNet-18)
   └────┬─────┘
        │ features [B, 512]
        ├────────────────────────────────────────┐
        ▼                                        ▼
 ┌──────────────────┐                  ┌──────────────────┐
 │ Classifier Head  │                  │  Decoder Head    │
 │ Linear(512→256)  │                  │ (Reconstruction) │
 │ ReLU + Dropout   │                  │ Linear(512→256)  │
 │ Linear(256→128)  │                  │ → Reshape        │
 │ ReLU + Dropout   │                  │ → ConvTranspose  │
 │ Linear(128→10)   │                  │ → Sigmoid output │
 └────────┬─────────┘                  └────────┬─────────┘
          ▼                                     ▼
   logits [B, 10]                   reconstructed [B, 3, 32, 32]

Output dict: {
  "classification": [B, 10],
  "reconstruction": [B, 3, 32, 32],
  "features":       [B, 512]
}
```

---

## Stage 4 — Loss Computation (`losses.py`)

```
predictions["classification"] ──▶ CrossEntropyLoss ──▶ cls_loss
predictions["reconstruction"] ──▶ MSELoss           ──▶ recon_loss
targets["labels"]              ──┘
targets["images"]              ──┘

               Fixed weights mode:
               total = α × cls_loss + β × recon_loss
               (default α=1.0, β=0.5)

               Learnable weights mode (Kendall et al.):
               total = (e^{-log_var_cls} × cls_loss + log_var_cls)
                     + (e^{-log_var_recon} × recon_loss + log_var_recon)
               [log_var_cls and log_var_recon are nn.Parameters]

Output: { "total": scalar, "classification": scalar, "reconstruction": scalar }
```

### GradientAnalyzer

Registers `register_full_backward_hook` on every Conv2d and Linear layer. After `loss.backward()`, captures per-layer gradient norm, mean, std, and max — then flags layers as `✅ healthy`, `⚠️ VANISHING`, or `🔥 EXPLODING`.

---

## Stage 5 — Training Loop (`trainer.py`)

```
for epoch in 1..N:
  ┌─────────────────────────────────────────────────────────────────┐
  │  TRAIN PHASE  (model.train())                                    │
  │                                                                   │
  │  for batch in train_loader:                                       │
  │    images, labels, clean → .to(device)                           │
  │                                                                   │
  │    with autocast():          ← AMP mixed precision               │
  │      predictions = model(images)                                  │
  │      loss_dict   = criterion(predictions, targets)               │
  │                                                                   │
  │    optimizer.zero_grad()                                          │
  │    scaler.scale(loss["total"]).backward()                        │
  │    scaler.unscale_(optimizer)                                     │
  │    clip_grad_norm_(model.parameters(), max_norm=1.0)             │
  │    scaler.step(optimizer)                                         │
  │    scaler.update()                                                │
  └─────────────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────────────┐
  │  VALIDATION PHASE  (model.eval(), torch.no_grad())              │
  │    — same forward pass, no backward                              │
  │    — logs reconstruction comparison to TensorBoard              │
  └─────────────────────────────────────────────────────────────────┘
  │
  ├── scheduler.step()          ← CosineAnnealing or ReduceLROnPlateau
  ├── TBLogger.log_scalars()    ← epoch-level metrics
  ├── ModelCheckpoint.save_best() if val_loss improved
  └── EarlyStopping check       ← stop if no improvement for N epochs
```

### Optimizer Options

| Name | Class | Notes |
|------|-------|-------|
| `adam` | `torch.optim.Adam` | Default adaptive |
| `sgd` | `torch.optim.SGD` | With momentum |
| `adamw` | `torch.optim.AdamW` | Decoupled weight decay (default) |

### Scheduler Options

| Name | Behaviour |
|------|-----------|
| `cosine` | `CosineAnnealingLR` — smoothly decays LR to `min_lr` over all epochs |
| `plateau` | `ReduceLROnPlateau` — halves LR when val loss stagnates |
| `none` | Constant LR |

---

## Stage 6 — Evaluation (`evaluate.py`)

```
Load best_model.pth checkpoint
        │
        ▼
model.eval() + torch.no_grad()
        │
        ▼
Iterate test_loader
  └─▶ accumulate predictions
        │
        ├─▶ Overall accuracy
        ├─▶ Per-class accuracy
        ├─▶ Confusion matrix (matplotlib → confusion_matrix.png)
        └─▶ Reconstruction comparison (original vs decoded → reconstruction_comparison.png)
```

---

## Stage 7 — Inference Pipeline (`inference.py`)

```
Single raw image (PIL or tensor)
        │
        ▼
InferencePipeline.predict(image)
  └─▶ Apply inference transform (no augmentation, only normalize)
  └─▶ model.eval() forward pass
  └─▶ softmax on logits
  └─▶ argmax → predicted_class
  └─▶ return { predicted_class, confidence, class_probabilities, reconstruction }

Optional: predict_and_visualize() → saves inference_demo.png
```

---

## Observability & Experiment Tracking

```
Training runtime                     Post-training
──────────────                       ─────────────
TensorBoard (visualize.py)           MLflow (mlflow_deploy.py)
  ├─ Model graph                       ├─ Experiment runs
  ├─ Batch loss (every 50 steps)       ├─ Hyperparameter logging
  ├─ Epoch train/val loss & acc        ├─ Metric history
  ├─ Learning rate                     ├─ Artefact storage
  └─ Reconstruction comparison         └─ Model registry
```

Launch TensorBoard: `tensorboard --logdir=./runs`  
Launch MLflow UI: `mlflow ui` (at http://localhost:5000)

---

## File Dependency Map

```
main.py
 ├── config.py              (no dependencies)
 ├── dataset.py             ← config.py
 ├── models/
 │    ├── backbone.py       ← config.py
 │    └── multitask_model.py ← backbone.py, config.py
 ├── losses.py              (nn.Module only)
 ├── trainer.py             ← config.py, losses.py, utils.py, visualize.py
 ├── evaluate.py            ← config.py
 ├── inference.py           ← config.py
 ├── visualize.py           ← (TensorBoard SummaryWriter)
 └── utils.py               (torch only)
```

---

## Key Design Decisions

**Why multi-task?** Forcing the backbone to reconstruct the input as well as classify it acts as a regulariser — the features must be semantically rich enough to support both tasks.

**Why dataclasses?** Typed, composable, and IDE-friendly. Avoids `argparse` spaghetti for a project of this scale.

**Why learnable loss weights?** With fixed α/β the developer must hand-tune the balance between classification and reconstruction. Uncertainty weighting (Kendall 2018) learns the optimal balance automatically.

**Why GradientAnalyzer with backward hooks?** Exploding/vanishing gradients are silent killers in deep CNNs. Registering hooks at training time gives per-layer visibility without altering the forward graph.
