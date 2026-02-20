# Skin Cancer Detection — Reference Documentation

## Model Configurations

| Model | Config File | TIMM Backbone | Default img_size | Default batch_size |
|-------|-----------|---------------|-----------------|-------------------|
| EfficientNet-B0 | `configs/model/efficientnet_b0.yaml` | `tf_efficientnet_b0_ns` | 224 | 32 |
| EfficientNetV2-L | `configs/model/efficientnetv2_l.yaml` | `tf_efficientnetv2_l.in21k_ft_in1k` | 384 | 16 |
| ConvNeXt-Large | `configs/model/convnext_large.yaml` | `convnext_large.fb_in22k_ft_in1k` | 224 | 16 |
| ConvNeXt-Tiny | `configs/model/convnext_tiny.yaml` | `convnext_tiny` | 224 | 32 |
| EVA02-Large | `configs/model/eva02_large.yaml` | `eva02_large_patch14_448.mim_in22k_ft_in1k` | 448 | 8 |
| Swin-Large | `configs/model/swin_large.yaml` | `swin_large_patch4_window7_224.ms_in22k_ft_in1k` | 224 | 16 |

Each model config specifies a TIMM backbone name, default hyperparameters (lr, weight_decay, dropout), and `num_classes: 1` for binary classification. All backbones are ImageNet-pretrained.

Each model has a corresponding experiment config in `configs/experiment/isic_{model_name}.yaml` which sets the complete training configuration including data, callbacks, logger, and trainer settings.

---

## Class Imbalance Handling (pos_weight)

### The Problem

ISIC 2024 has extreme class imbalance: ~99.5% benign, ~0.5% malignant. With plain `BCEWithLogitsLoss()`, the model learns to predict near-zero probability for everything (since that minimizes loss on 99.5% of samples). This causes:
- Optimal thresholds to be extremely small (~0.00005)
- Poor sensitivity to malignant cases

### The Fix

`pos_weight` in `BCEWithLogitsLoss` multiplies each positive (malignant) sample's loss contribution by the given factor. This is computed dynamically:

```
pos_weight = num_negative_samples / num_positive_samples
```

For ISIC 2024, this is typically ~199. The effect: each malignant sample counts as much as 199 benign ones during training, forcing the model to produce meaningful probability scores for both classes.

### How It Works

1. `ISICDataModule.setup()` computes `pos_weight` from the training split's class distribution
2. `train.py` reads this value and injects it into the model config via `open_dict`
3. `ISICLitModule.__init__` passes it to `BCEWithLogitsLoss(pos_weight=...)`
4. The `pos_weight` value is printed during setup for visibility

The `pos_weight` field in model configs defaults to `1.0` (no weighting) but is always overridden at runtime.

### Expected Effect

| Without pos_weight | With pos_weight |
|---|---|
| Threshold: ~0.00005 | Threshold: ~0.3–0.6 |
| Model predicts near-zero for all | Meaningful probability scores |
| High specificity, low sensitivity | Balanced sensitivity/specificity |

---

## Checkpoint Directory Structure

```
checkpoints/
├── efficientnet_b0/
│   ├── fold_0/
│   │   ├── epoch_015_auroc_0.9234.ckpt    ← best checkpoint (by val/auroc)
│   │   └── last.ckpt                       ← last epoch checkpoint
│   ├── fold_1/
│   │   ├── epoch_012_auroc_0.9189.ckpt
│   │   └── last.ckpt
│   └── ...
├── convnext_large/
│   ├── fold_0/
│   └── ...
└── ...
```

- **Best checkpoint**: Named `epoch_{N}_auroc_{value}.ckpt`, selected by `ModelCheckpoint(monitor="val/auroc", mode="max", save_top_k=1)`
- **Last checkpoint**: Always saved as `last.ckpt` for resume capability
- Checkpoint path is set via Hydra override: `callbacks.model_checkpoint.dirpath=checkpoints/{model_name}/fold_{fold}`

### What's stored in each checkpoint

- Model weights
- Optimizer state
- All hyperparameters (via `save_hyperparameters()`)
- `best_threshold` — the optimal classification threshold found from the ROC curve (registered buffer)
- `best_auroc` — the best validation AUROC achieved (used as weight in ensemble)

---

## Optimal Threshold

### How it's computed

At the end of each validation epoch, the full ROC curve is computed using `torchmetrics.BinaryROC`. The optimal threshold is the one where `(TPR, FPR)` is closest to the ideal point `(1, 0)`:

```
distance = sqrt((1 - TPR)² + FPR²)
optimal_threshold = threshold[argmin(distance)]
```

### Where it's stored

- **In the checkpoint**: Stored as a registered buffer `model.best_threshold` — automatically saved/loaded with the checkpoint
- **In WandB**: Logged as `val/best_threshold` each epoch
- **During inference**: Automatically used by `predict_step()` and `ensemble_predict.py`

### What it means

The default threshold of 0.5 for binary classification is rarely optimal for imbalanced datasets. The optimal threshold maximizes the trade-off between sensitivity (TPR) and specificity (1-FPR), which is critical for medical image classification.

---

## WandB Integration

### Metrics Logged

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per epoch |
| `train/auroc` | Training AUROC per epoch |
| `val/loss` | Validation loss per epoch |
| `val/auroc` | Validation AUROC per epoch |
| `val/auroc_best` | Best validation AUROC so far |
| `val/best_threshold` | Optimal threshold from ROC curve |
| `roc_curve/epoch_N` | Interactive ROC curve for epoch N |

### Viewing AUROC Curves in WandB

1. Go to your WandB project (`isic-2024`)
2. Click on a run (named `{model_name}_fold{N}`)
3. Navigate to **Charts** → Filter for `roc_curve` to see per-epoch ROC curves
4. To compare folds: select multiple runs → **Compare** view
5. The ROC curves show the full TPR vs FPR trade-off with the optimal threshold marked in the title
6. Chart titles include zero-padded epoch numbers (e.g., `ROC Curve | Epoch 015 | AUROC=...`) for easy sorting

---

## Training Speed Optimizations

### check_val_every_n_epoch

All experiment configs set `check_val_every_n_epoch: 2`, meaning validation only runs on even-numbered epochs. This saves ~50% of wall time since validation involves a full forward pass over the validation set + ROC curve computation.

### FP16 Mixed Precision

All experiment configs and notebook platform overrides use `precision: 16-mixed`. The P100's tensor cores process FP16 ~2× faster than FP32.

### prefetch_factor

Both train and val dataloaders use `prefetch_factor=2` so that each worker pre-loads 2 batches into a queue while the GPU is computing. With `num_workers=4`, 8 batches are always ready in RAM.

### accumulate_grad_batches (large models)

EVA02-Large (`accumulate_grad_batches: 4`, batch=8 → effective batch=32) and Swin-Large (`accumulate_grad_batches: 2`, batch=16 → effective batch=32) accumulate gradients across multiple forward/backward passes before updating weights. This provides better training stability without exceeding GPU memory.

### Approximate training times (P100, 40k images, FP16)

| Model | ~Time/epoch |
|-------|-------------|
| EfficientNet-B0 | ~2.5 min |
| ConvNeXt-Tiny | ~5 min |
| ConvNeXt-Large | ~10 min |
| Swin-Large | ~35 min |
| EfficientNetV2-L | ~15 min |
| EVA02-Large | ~30 min |

### Downloading Checkpoints from WandB Artifacts

**Via WandB UI:**
1. Go to project → **Artifacts** tab
2. Find the model artifact (e.g., `model-{run_id}`)
3. Click **Files** → Download the `.ckpt` file

**Programmatically:**
```python
import wandb

api = wandb.Api()
# List all artifacts
artifacts = api.artifacts("your-entity/isic-2024", type_name="model")
for art in artifacts:
    print(art.name, art.state)

# Download a specific artifact
artifact = api.artifact("your-entity/isic-2024/model-{run_id}:v0")
artifact_dir = artifact.download("./downloaded_checkpoints/")
```

### Resume Training from WandB Checkpoint

1. Download the checkpoint (see above)
2. Run training with `ckpt_path` override:
```bash
python src/train.py experiment=isic_efficientnet_b0 \
    ckpt_path=path/to/downloaded/checkpoint.ckpt \
    data.fold=0
```

Or in the notebook, add to hydra overrides:
```python
hydra_overrides.append(f"ckpt_path={path_to_ckpt}")
```

---

## Ensemble Prediction

### CLI Usage

```bash
# Auto img_size, soft voting (default)
python src/ensemble_predict.py \
    --models efficientnet_b0 \
    --checkpoint-dir checkpoints/ \
    --image-paths img1.jpg img2.jpg

# Multi-model ensemble with hard weighted voting
python src/ensemble_predict.py \
    --models efficientnet_b0 convnext_large swin_large \
    --checkpoint-dir checkpoints/ \
    --image-dir test_images/ \
    --strategy hard_weighted \
    --output-csv results.csv

# Override image size (disables auto-detection)
python src/ensemble_predict.py \
    --models eva02_large \
    --checkpoint-dir checkpoints/ \
    --image-paths test.jpg \
    --img-size 448
```

### Ensemble Strategies

| Strategy | Description |
|----------|-------------|
| `soft` (default) | Average probabilities across all folds/models, then apply averaged threshold |
| `hard_weighted` | Apply per-checkpoint threshold to get binary predictions, then weighted average using each fold's AUROC as weight. The resulting fraction is a confidence score |

### How it works

1. For each specified model, auto-discovers available `fold_N/` subdirectories
2. For each fold, finds the best checkpoint (highest AUROC from filename, or `last.ckpt`)
3. Loads the checkpoint (which contains `best_threshold`, `best_auroc`, and backbone name)
4. **Auto-detects image size** from the TIMM backbone config (no manual `--img-size` needed)
5. Each model is preprocessed at its correct resolution (e.g., 448 for EVA02, 224 for EfficientNet-B0)
6. Applies selected ensemble strategy

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--models` | Yes | — | Space-separated model names |
| `--checkpoint-dir` | No | `checkpoints` | Base checkpoint directory |
| `--image-paths` | No* | — | Space-separated image file paths |
| `--image-dir` | No* | — | Directory of images to predict |
| `--img-size` | No | auto-detect | Override image size (disables auto-detection) |
| `--strategy` | No | `soft` | `soft` or `hard_weighted` |
| `--device` | No | auto | `cuda` or `cpu` |
| `--output-csv` | No | — | Save results to CSV |

*At least one of `--image-paths` or `--image-dir` must be specified.

---

## Notebook Usage

### Overriding Models to Train

In Cell 4 of the notebook, modify `MODELS_TO_TRAIN`:

```python
# Train just one model
MODELS_TO_TRAIN = ["efficientnet_b0"]

# Train multiple models
MODELS_TO_TRAIN = ["efficientnet_b0", "convnext_large", "swin_large"]

# Train all available models
MODELS_TO_TRAIN = ["efficientnet_b0", "efficientnetv2_l", "convnext_large", "eva02_large", "swin_large"]
```

### Multi-Fold Training

Also in Cell 4:

```python
N_FOLDS = 5          # Train all 5 folds (for full cross-validation)
N_FOLDS = 1          # Train only fold 0 (for quick testing)
DATA_FRACTION = 0.1  # Use 10% of data (for quick testing)
DATA_FRACTION = 1.0  # Use full dataset
```

### Inference Cell

Cell 6 provides single-image and ensemble inference:

```python
# Single model inference
USE_ENSEMBLE = False
INFERENCE_MODEL = "efficientnet_b0"
INFERENCE_FOLD = 0
IMAGE_PATHS = ["/path/to/image.jpg"]

# Ensemble inference
USE_ENSEMBLE = True
ENSEMBLE_MODELS = ["efficientnet_b0", "convnext_large"]
IMAGE_PATHS = ["/path/to/image.jpg"]
```

---

## Cross-Validation Setup

- **5-fold stratified K-fold** split by patient ID (prevents data leakage between train/val)
- Stratification is based on the target label (malignant/benign) at the patient level
- The fold column is computed on-the-fly with a fixed `random_state=42` for reproducibility
- `data.fold=N` selects fold N as validation, rest as training

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/isic_module.py` | Lightning module with TIMM backbone, pos_weight, ROC threshold, WandB curves |
| `src/data/isic_datamodule.py` | Data loading with HDF5, stratified K-fold, data fraction, pos_weight computation |
| `src/ensemble_predict.py` | Multi-model multi-fold ensemble inference CLI |
| `src/train.py` | Hydra-based training entry point, injects pos_weight from data to model |
| `configs/model/*.yaml` | Model backbone configurations (including pos_weight default) |
| `configs/experiment/isic_*.yaml` | Complete experiment configurations |
| `configs/callbacks/default.yaml` | Checkpoint saving, early stopping configs |
| `configs/logger/wandb.yaml` | WandB logger config (`log_model: False`) |
| `notebooks/skin-cancer-detection.ipynb` | Kaggle/Colab/local training notebook |
