# 📚 Technical Reference — Skin Cancer Detection

> **[← Back to README](../README.md)**

---

## Model Configurations

| Model | Config File | TIMM Backbone | img_size | batch_size |
|-------|-----------|---------------|----------|------------|
| EfficientNet-B0 | `configs/model/efficientnet_b0.yaml` | `tf_efficientnet_b0_ns` | 256 | 32 |
| MobileNetV3 | `configs/model/mobilenet_v3.yaml` | `mobilenetv3_large_100.ra_in1k` | 256 | 32 |

Each model config specifies a TIMM backbone name, default hyperparameters (lr, weight_decay, dropout), `num_classes: 1` for binary classification, and `n_tabular_features: 0` (overridden at runtime). All backbones are ImageNet-pretrained.

Each model has a corresponding experiment config in `configs/experiment/isic_{model_name}.yaml` which sets the complete training configuration including data, callbacks, logger, and trainer settings.

---

## Image + Tabular Fusion Architecture

### Why tabular features matter

ISIC 2024 provides 3D-TBP metadata alongside images: patient age, anatomical site, lesion color/geometry measurements, symmetry scores, and DNN confidence values. Combining image features with this tabular data significantly improves detection — dermatologists also use clinical context (patient age, lesion location) when diagnosing.

### Architecture

```
Image (256×256) → TIMM Backbone (num_classes=0) → image_features (e.g. 1280-dim)
                                                         ↓
Tabular (42-dim) ─────────────────────────────── → [concat] → Fusion MLP → 1 logit
```

**Fusion MLP**: `Linear(img_dim + 42, 128) → BatchNorm → ReLU → Dropout → Linear(128, 1)`

### Tabular features (42 dimensions)

| Category | Features | Count |
|----------|----------|-------|
| Numeric | age, lesion size, LAB/LCH color channels, color deltas, area, perimeter, eccentricity, symmetry, 3D position, DNN confidence | 33 |
| Categorical | sex (1 binary) | 1 |
| Categorical | anatom_site_general (7 one-hot) | 7 |
| **Total** | | **41** |

### How it works

1. `ISICDataModule.setup()` encodes tabular features from the metadata CSV
2. Numeric features are standardized (zero mean, unit variance) using **training set statistics only**
3. `train.py` reads `datamodule.n_tabular_features` and injects it into the model config
4. `ISICLitModule.__init__` creates a TIMM backbone with `num_classes=0` (strips classifier head) and builds a fusion MLP
5. `forward(images, tabular)` extracts image features, concatenates with tabular, and passes through the fusion MLP

### Backward compatibility

When `n_tabular_features=0` (default in model configs), the model falls back to image-only mode with TIMM's built-in classifier head. Old checkpoints load normally.

---

## Advanced 3D-TBP Augmentations

ISIC 2024 images are crops from 3D Total Body Photography. They often contain black borders, hair artifacts, and varying illumination:

| Augmentation | Why |
|---|---|
| `RandomResizedCrop(scale=0.7-1.0)` | Handles black borders by zooming in; forces model to learn from different regions |
| `Transpose + Flip + Rotate90` | Full geometric invariance — lesion orientation doesn't matter |
| `ShiftScaleRotate(rotate=30°, border=black)` | Broader rotation with black border fill matching 3D-TBP edges |
| `HueSaturationValue` | Handles color variation across different 3D-TBP devices |
| `RandomBrightnessContrast` | Handles illumination differences |
| `CoarseDropout(max_holes=8, fill=black)` | Simulates hair and artifact occlusion |
| `GaussNoise / GaussianBlur / MotionBlur` | Robustness to image quality variations |

---

## Higher Resolution Training (256px)

All experiment configs use 256×256 images. Higher resolution is critical for dermatology — small visual details (asymmetry, border irregularity, color variation) that distinguish malignant from benign lesions are better captured at 256px than 224px, while keeping training times manageable compared to 384px or 448px.

---

## Class Imbalance Handling (pos_weight)

### The Problem

ISIC 2024 has extreme class imbalance: **~99.5% benign, ~0.5% malignant**. With plain `BCEWithLogitsLoss()`, the model learns to predict near-zero probability for everything (since that minimizes loss on 99.5% of samples). This causes:
- Optimal thresholds to be extremely small (~0.00005)
- Poor sensitivity to malignant cases

### The Fix

`pos_weight` in `BCEWithLogitsLoss` multiplies each positive (malignant) sample's loss contribution by the given factor:

```
pos_weight = num_negative_samples / num_positive_samples
```

For ISIC 2024, this is typically **~199**. The effect: each malignant sample counts as much as 199 benign ones during training, forcing the model to produce meaningful probability scores for both classes.

### How It Works

1. `ISICDataModule.setup()` computes `pos_weight` from the training split's class distribution
2. `train.py` reads this value and injects it into the model config via `open_dict`
3. `ISICLitModule.__init__` passes it to `BCEWithLogitsLoss(pos_weight=...)`
4. The `pos_weight` value is printed during setup for visibility

| Without pos_weight | With pos_weight |
|---|---|
| Threshold: ~0.00005 | Threshold: ~0.3–0.6 |
| Model predicts near-zero for all | Meaningful probability scores |
| High specificity, low sensitivity | Balanced sensitivity/specificity |

---

## Checkpoint Structure

```
checkpoints/
├── efficientnet_b0/
│   ├── fold_0/
│   │   ├── epoch_015_auroc_0.9234.ckpt    ← best (by val/auroc)
│   │   └── last.ckpt                       ← last epoch
│   ├── fold_1/
│   │   └── ...
│   └── ...
├── mobilenet_v3/
│   └── ...
└── ...
```

- **Best checkpoint**: Named `epoch_{N}_auroc_{value}.ckpt`, selected by `ModelCheckpoint(monitor="val/auroc", mode="max", save_top_k=1)`
- **Last checkpoint**: Always saved as `last.ckpt` for resume capability

### What's stored in each checkpoint

- Model weights
- Optimizer state
- All hyperparameters (via `save_hyperparameters()`)
- `best_threshold` — optimal classification threshold from ROC curve (registered buffer)
- `best_auroc` — best validation AUROC achieved (used as weight in ensemble)

### Downloading Checkpoints

Checkpoints are hosted on [🤗 Hugging Face Hub](https://huggingface.co/RudraShivm/skin-cancer-detection-isic2024):

```bash
# Download all checkpoints
python scripts/download_checkpoints.py

# Download specific model only
python scripts/download_checkpoints.py --model efficientnet_b0
```

---

## Optimal Threshold

### How it's computed

At the end of each validation epoch, the full ROC curve is computed using `torchmetrics.BinaryROC`. The optimal threshold is the one where `(TPR, FPR)` is closest to the ideal point `(1, 0)`:

```
distance = sqrt((1 - TPR)² + FPR²)
optimal_threshold = threshold[argmin(distance)]
```

### Where it's stored

- **In the checkpoint**: Stored as a registered buffer `model.best_threshold`
- **In WandB**: Logged as `val/best_threshold` each epoch
- **During inference**: Automatically used by `predict_step()` and `ensemble_predict.py`

> [!NOTE]
> The default threshold of 0.5 for binary classification is rarely optimal for imbalanced datasets. The optimal threshold maximizes the trade-off between sensitivity (TPR) and specificity (1-FPR).

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

---

## Training Speed Optimizations

| Optimization | Effect |
|---|---|
| `check_val_every_n_epoch: 2` | Validation only on even epochs, ~50% wall time saved |
| `precision: 16-mixed` | FP16 mixed precision, ~2× faster on tensor cores |
| `prefetch_factor: 2` | Workers pre-load 2 batches into queue (8 batches ready with 4 workers) |

### Approximate training times (P100, 40k images, FP16)

| Model | ~Time/epoch |
|-------|-------------|
| EfficientNet-B0 | ~2.5 min |
| MobileNetV3 | ~2 min |

---

## GBDT Stacking

### Overview

GBDT models (LightGBM, XGBoost, CatBoost) are trained as a second stage on top of CNN predictions. The pipeline is:

1. **Extract CNN features** — `extract_cnn_features.py` runs each CNN checkpoint on the training data
2. **Train GBDT models** — `train_gbdt.py` trains gradient-boosted trees on CNN probs + tabular + patient-relative features

### Checkpoint Structure

```
checkpoints/gbdt/
├── lightgbm_fold0_seed42.pkl
├── lightgbm_fold0_seed42_info.json
├── xgboost_fold0_seed42.pkl
├── xgboost_fold0_seed42_info.json
├── catboost_fold0_seed42.pkl
├── catboost_fold0_seed42_info.json
└── ...  (3 types × 5 folds × 3 seeds = 45 models)
```

Each `_info.json` contains: `type`, `fold`, `seed`, `threshold`, `val_auroc`, `n_features`, `feature_names`, `model_name`.

### CLI Usage

```bash
# Step 1: Extract CNN predictions
python src/gbdt/extract_cnn_features.py \
    --checkpoint-dir checkpoints/ \
    --data-dir data/isic-2024-challenge \
    --output-dir outputs/gbdt_features \
    --n-folds 5

# Step 2: Train GBDT models
python src/gbdt/train_gbdt.py \
    --features-dir outputs/gbdt_features \
    --output-dir checkpoints/gbdt \
    --gbdt-types lightgbm xgboost catboost \
    --seeds 42 123 456 \
    --noise-sigma 0.1
```

### Feature Engineering

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Tabular (standardized) | 43 | Same features used by CNN fusion head |
| CNN probabilities | 1 per model | Sigmoid output from each trained CNN |
| Patient-relative | 3 per CNN | Ratio to mean, diff from mean, z-score within patient |

**Noise injection**: During training, Gaussian noise (σ=0.1) is added to CNN probabilities to prevent GBDT from over-relying on CNN predictions.

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
    --models efficientnet_b0 mobilenet_v3 \
    --checkpoint-dir checkpoints/ \
    --image-dir test_images/ \
    --strategy hard_weighted \
    --output-csv results.csv
```

### Ensemble Strategies

| Strategy | Description |
|----------|-------------|
| `soft` (default) | Average probabilities across all folds/models, then apply averaged threshold |
| `hard_weighted` | Apply per-checkpoint threshold, then weighted average using each fold's AUROC as weight |

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--models` | Yes | — | Space-separated model names |
| `--checkpoint-dir` | No | `checkpoints` | Base checkpoint directory |
| `--image-paths` | No* | — | Space-separated image file paths |
| `--image-dir` | No* | — | Directory of images to predict |
| `--img-size` | No | auto-detect | Override image size |
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
MODELS_TO_TRAIN = ["efficientnet_b0", "mobilenet_v3"]
```

### Multi-Fold Training

```python
N_FOLDS = 5          # Train all 5 folds (full cross-validation)
N_FOLDS = 1          # Train only fold 0 (quick testing)
DATA_FRACTION = 0.1  # Use 10% of data (quick testing)
DATA_FRACTION = 1.0  # Use full dataset
```

### Inference Cell

```python
# Single model inference
USE_ENSEMBLE = False
INFERENCE_MODEL = "efficientnet_b0"
IMAGE_PATHS = ["/path/to/image.jpg"]

# Ensemble inference
USE_ENSEMBLE = True
ENSEMBLE_MODELS = ["efficientnet_b0", "mobilenet_v3"]
```

---

## Cross-Validation Setup

- **5-fold stratified K-fold** split by patient ID (prevents data leakage)
- Stratification based on target label (malignant/benign) at the patient level
- Fold column computed on-the-fly with fixed `random_state=42` for reproducibility
- `data.fold=N` selects fold N as validation, rest as training

---

## Key Files

| File | Purpose |
|------|---------|
| `src/models/isic_module.py` | Lightning module: TIMM backbone, fusion MLP, pos_weight, ROC threshold, WandB curves |
| `src/data/isic_datamodule.py` | Data loading: HDF5, tabular features, standardization, stratified K-fold, pos_weight |
| `src/data/components/transforms.py` | 3D-TBP domain-specific augmentations |
| `src/ensemble_predict.py` | Multi-model multi-fold ensemble inference CLI |
| `src/gbdt/extract_cnn_features.py` | Extract CNN predictions for GBDT stacking (Stage 2 input) |
| `src/gbdt/train_gbdt.py` | Train LightGBM/XGBoost/CatBoost GBDT models (Stage 2) |
| `src/gbdt/predict_gbdt.py` | GBDT ensemble loading and prediction |
| `src/gradio_app.py` | Gradio web demo for interactive predictions |
| `src/train.py` | Hydra-based training entry point |
| `scripts/download_checkpoints.py` | Download pretrained models from HF Hub |
| `configs/model/*.yaml` | Model backbone configs |
| `configs/experiment/isic_*.yaml` | Complete experiment configurations |
| `notebooks/skin-cancer-detection.ipynb` | Kaggle/Colab training notebook (CNN + GBDT) |
| `notebooks/submission.ipynb` | Kaggle inference-only submission notebook (with GBDT stacking) |
