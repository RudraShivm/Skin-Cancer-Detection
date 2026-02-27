# 🏗️ Architecture — Skin Cancer Detection

> **[← Back to README](../README.md)**

---

## What This Project Does

This project **detects skin cancer from photos**. Given a photo of a skin lesion (mole/spot) and clinical measurements, it predicts whether the lesion is **malignant** (cancerous) or **benign** (harmless).

It was built for the [ISIC 2024 Kaggle Competition](https://www.kaggle.com/competitions/isic-2024-challenge), which uses images from 3D Total Body Photography (3D-TBP) — a system that photographs a patient's entire body and extracts crops of individual lesions.

---

## System Overview

```mermaid
flowchart TD
    subgraph Data["📂 Data"]
        HDF5["🖼️ train-image.hdf5<br/>(~400K JPEG images)"]
        META["📊 train-metadata.csv<br/>(patient info, color,<br/>geometry, 55 columns)"]
    end

    subgraph Stage1["⚡ Stage 1: CNN Training"]
        DM["ISICDataModule<br/>- Loads images + tabular<br/>- K-fold split<br/>- Augmentations<br/>- Standardization"]
        MODEL["ISICLitModule<br/>- TIMM backbone<br/>- Fusion MLP<br/>- BCEWithLogitsLoss"]
        TRAINER["Lightning Trainer<br/>- Mixed precision FP16<br/>- CosineAnnealing LR<br/>- Early stopping<br/>- WandB logging"]
    end

    subgraph Stage2["🌲 Stage 2: GBDT Stacking"]
        EXTRACT["Extract CNN Predictions<br/>extract_cnn_features.py"]
        GBDT["Train GBDTs<br/>train_gbdt.py<br/>LightGBM / XGBoost / CatBoost"]
    end

    subgraph Output["💾 Output"]
        CKPT["CNN Checkpoints<br/>(.ckpt files)"]
        GBDT_CKPT["GBDT Models<br/>(.pkl files)"]
        WANDB["📈 WandB Dashboard"]
    end

    subgraph Inference["🔮 Submission Pipeline"]
        LOAD["Load CNN checkpoints<br/>+ test data"]
        PREDICT["CNN Forward pass<br/>(image + tabular)"]
        CNN_ENS["CNN Ensemble<br/>(avg probabilities)"]
        GBDT_STACK["🌲 GBDT Stacking<br/>(CNN probs + tabular<br/>→ final prediction)"]
        CSV["submission.csv"]
    end

    HDF5 --> DM
    META --> DM
    DM --> MODEL
    MODEL --> TRAINER
    TRAINER --> CKPT
    TRAINER --> WANDB
    CKPT --> EXTRACT
    EXTRACT --> GBDT
    GBDT --> GBDT_CKPT
    CKPT --> LOAD
    LOAD --> PREDICT
    PREDICT --> CNN_ENS
    CNN_ENS --> GBDT_STACK
    GBDT_CKPT --> GBDT_STACK
    GBDT_STACK --> CSV
```

---

## How the Model Works

### The Problem with Just Looking at Photos

A photo alone often isn't enough to tell if a mole is cancerous. Dermatologists also consider:
- **Patient age** — risk increases with age
- **Location on body** — some sites are higher risk
- **Lesion size, shape, and color** relative to surrounding skin

Our model does the same — it looks at both the **image** and these **clinical measurements** (tabular features).

### Image + Tabular Fusion

```mermaid
flowchart LR
    subgraph Image_Path["🖼️ Image Path"]
        IMG["Skin Lesion Photo<br/>256×256 pixels"]
        AUG["Augmentations<br/>(crop, flip, color jitter,<br/>dropout)"]
        BACKBONE["TIMM Backbone<br/>(e.g. EfficientNet-B0)<br/>ImageNet pretrained"]
        FEAT["Image Features<br/>(1280-dim vector)"]
    end

    subgraph Tabular_Path["📊 Tabular Path"]
        TAB["42 Clinical Features<br/>(age, size, color,<br/>symmetry, ...)"]
        NORM["Standardize<br/>(zero mean,<br/>unit variance)"]
        TABOUT["Tabular Vector<br/>(42-dim)"]
    end

    subgraph Fusion["🔗 Fusion"]
        CONCAT["Concatenate<br/>(1280 + 42 = 1322 dims)"]
        MLP["Fusion MLP<br/>1322 → 128 → 1"]
        LOGIT["1 logit → sigmoid<br/>→ probability"]
        PRED["Prediction:<br/>malignant or benign"]
    end

    IMG --> AUG --> BACKBONE --> FEAT
    TAB --> NORM --> TABOUT
    FEAT --> CONCAT
    TABOUT --> CONCAT
    CONCAT --> MLP --> LOGIT --> PRED
```

### Step-by-Step Explanation

1. **Image Path**: The skin lesion photo (256×256) goes through a pre-trained backbone network (like EfficientNet). This backbone was originally trained on ImageNet (millions of everyday objects) and has learned to extract visual features (edges, textures, colors). We strip off its final classification layer and use the **feature vector** it produces (e.g., 1280 numbers that represent the image).

2. **Tabular Path**: 42 clinical measurements are extracted from the metadata CSV. These are standardized so each feature has mean=0 and std=1 (otherwise features with large values like `tbp_lv_y=1500` would dominate over features like `eccentricity=0.9`).

3. **Fusion**: The image feature vector (1280 dims) and tabular vector (42 dims) are concatenated into one long vector (1322 dims). This goes through a small MLP that learns to combine both signals:
   ```
   Linear(1322 → 128) → BatchNorm → ReLU → Dropout → Linear(128 → 1)
   ```
   The final output is a single number (logit). We apply sigmoid to get a probability between 0 and 1.

---

## Tabular Features

The metadata CSV contains clinical measurements taken by the 3D-TBP system.

### Standardization (Patient-Wise)

Before being fed to the model, tabular features are standardized (zero mean, unit variance).
This project uses **Patient-Wise Standardization** for all models. Instead of using global dataset statistics, it computes the mean and standard deviation for *each patient individually*. This implements the "ugly duckling" concept — highlighting how much a lesion deviates from the patient's own normal skin, rather than the global population.

**Fallback Mechanism**: If a patient only has one lesion (which occurs in single-image inference in the **Gradio app**, or for patients with only one image in the dataset), the standardization automatically falls back to Global Training Statistics. However, the logic is unified and ready to handle multi-image batches in Gradio.
### Patient Demographics

| Feature | What it is |
|---------|-----------|
| `age_approx` | Patient's approximate age (5–85) |
| `sex` | Male/Female → encoded as 1/0 |
| `anatom_site_general` | Where on the body → one-hot encoded (head, arms, legs, torso, etc.) |

### Lesion Size & Shape

| Feature | What it is |
|---------|-----------|
| `clin_size_long_diam_mm` | Longest diameter in mm |
| `tbp_lv_areaMM2` | Lesion area in mm² |
| `tbp_lv_perimeterMM` | Perimeter in mm |
| `tbp_lv_minorAxisMM` | Shortest diameter |
| `tbp_lv_area_perim_ratio` | How compact/round the shape is |
| `tbp_lv_eccentricity` | How elongated (0=circle, 1=line) |
| `tbp_lv_symm_2axis` | How symmetric the lesion is |

### Lesion Color (LAB Color Space)

The 3D-TBP system measures colors in LAB color space (L=lightness, A=red-green, B=yellow-blue):

| Feature | What it is |
|---------|-----------|
| `tbp_lv_L`, `tbp_lv_A`, `tbp_lv_B` | Lesion color |
| `tbp_lv_Lext`, `tbp_lv_Aext`, `tbp_lv_Bext` | **Surrounding** skin color |
| `tbp_lv_deltaL/A/B` | **Difference** (lesion − surround) |
| `tbp_lv_H`, `tbp_lv_C` | Hue and Chroma |
| `tbp_lv_stdL`, `tbp_lv_stdLExt` | Color variation within lesion/surround |
| `tbp_lv_color_std_mean` | Overall color non-uniformity |
| `tbp_lv_radial_color_std_max` | Max color variation from center to edge |

> [!TIP]
> **Why color matters**: Malignant lesions often have irregular color patterns — multiple shades of brown, black, red, or blue within the same lesion. The `delta` features capture how much the lesion differs from surrounding healthy skin.

### Border & Confidence

| Feature | What it is |
|---------|-----------|
| `tbp_lv_norm_border` | Border regularity score (irregular borders = suspicious) |
| `tbp_lv_norm_color` | Color uniformity score |
| `tbp_lv_nevi_confidence` | DNN confidence that it's a normal mole |
| `tbp_lv_dnn_lesion_confidence` | DNN confidence that it's a lesion at all |

### 3D Body Position

| Feature | What it is |
|---------|-----------|
| `tbp_lv_x`, `tbp_lv_y`, `tbp_lv_z` | 3D coordinates on the body surface |

---

## Data Augmentation

ISIC 2024 images have specific characteristics that our augmentations address:

```mermaid
flowchart LR
    subgraph Original["Original 3D-TBP Image"]
        O["Skin lesion crop<br/>with black borders,<br/>possible hair artifacts"]
    end

    subgraph Augmentations["Training Augmentations (random per image)"]
        A1["🔍 RandomResizedCrop<br/>(zoom to 70-100%)<br/>Removes black borders"]
        A2["↔️ Flip + Rotate<br/>(any orientation)<br/>Lesions look same<br/>upside down"]
        A3["🎨 Color Jitter<br/>(brightness, contrast,<br/>hue, saturation)<br/>Handles different<br/>camera settings"]
        A4["⬛ CoarseDropout<br/>(random black patches)<br/>Simulates hair<br/>covering the lesion"]
        A5["🌫️ Blur / Noise<br/>(Gaussian, motion)<br/>Handles image<br/>quality variation"]
    end

    O --> A1 --> A2 --> A3 --> A4 --> A5
```

**Why augment?** We only have ~400K images, but we need the model to be robust to:
- Different camera angles → flips, rotations
- Different lighting/color calibration → color jitter
- Hair or artifacts covering the lesion → CoarseDropout
- Lesion being in different parts of the crop → RandomResizedCrop

---

## Training Pipeline

```mermaid
flowchart TD
    subgraph Config["⚙️ Hydra Configuration"]
        EXP["Experiment YAML<br/>(isic_efficientnet_b0.yaml)"]
        MOD["Model YAML<br/>(efficientnet_b0.yaml)"]
        DAT["Data YAML<br/>(isic2024.yaml)"]
    end

    subgraph Epoch["One Training Epoch"]
        TRAIN["Training Loop<br/>- Forward pass → prediction<br/>- Compute loss (BCE + pos_weight)<br/>- Backward pass → gradients<br/>- Update weights (AdamW)"]
        VAL["Validation (every 2 epochs)<br/>- Forward pass only<br/>- Compute AUROC<br/>- Compute ROC curve<br/>- Find optimal threshold<br/>- Log to WandB"]
    end

    subgraph LR["📉 Learning Rate"]
        COSINE["CosineAnnealingLR<br/><br/>Starts at lr=0.0001<br/>Smoothly decays to<br/>lr=0.000001"]
    end

    EXP --> TRAIN
    MOD --> TRAIN
    DAT --> TRAIN
    TRAIN --> VAL
    COSINE -.-> TRAIN
```

### Key Training Concepts

- **BCEWithLogitsLoss + pos_weight**: Binary Cross-Entropy loss for classification. The `pos_weight` (~199) tells the loss function: "each malignant sample is worth 199 benign samples." Without this, the model would predict "benign" for everything and still get 99.5% accuracy.

- **CosineAnnealing LR**: The learning rate starts at 0.0001 and smoothly decreases to 0.000001 following a cosine curve. Early in training, we want big steps; near the end, tiny steps for fine-tuning.

- **Mixed Precision (FP16)**: Uses 16-bit floating point for most computations. ~2× faster, ~half GPU memory, negligible quality loss.

---

## Cross-Validation & Ensemble

```mermaid
flowchart TD
    subgraph Split["5-Fold Cross-Validation"]
        F0["Fold 0: Train on 1-4, validate on 0"]
        F1["Fold 1: Train on 0,2-4, validate on 1"]
        F2["Fold 2: Train on 0-1,3-4, validate on 2"]
        F3["Fold 3: Train on 0-2,4, validate on 3"]
        F4["Fold 4: Train on 0-3, validate on 4"]
    end

    subgraph Checkpoints["5 Checkpoints"]
        C0["fold_0/best.ckpt<br/>threshold=0.42"]
        C1["fold_1/best.ckpt<br/>threshold=0.38"]
        C2["fold_2/best.ckpt<br/>threshold=0.45"]
        C3["fold_3/best.ckpt<br/>threshold=0.40"]
        C4["fold_4/best.ckpt<br/>threshold=0.41"]
    end

    subgraph Ensemble["Soft Ensemble (at test time)"]
        AVG["Average all 5<br/>probabilities"]
        THR["Apply averaged<br/>threshold"]
        OUT["Malignant or Benign"]
    end

    F0 --> C0
    F1 --> C1
    F2 --> C2
    F3 --> C3
    F4 --> C4
    C0 --> AVG
    C1 --> AVG
    C2 --> AVG
    C3 --> AVG
    C4 --> AVG
    AVG --> THR --> OUT
```

**Why 5 folds?** Each model sees 80% of data during training. By training 5 models on different 80% slices, we:
1. Use ALL data for both training and validation (just not at the same time)
2. Get more robust predictions by averaging 5 opinions
3. Reduce overfitting

**Patient-level splitting**: We split by `patient_id`, not by image. This prevents data leakage — if the same patient has 10 lesion photos, all 10 go into the same fold.

---

## ABCD Rule Connection

Dermatologists use the **ABCD rule** to evaluate moles. Our tabular features capture the same concepts:

```mermaid
flowchart LR
    subgraph ABCD["🩺 Clinical ABCD Rule"]
        A_["A: Asymmetry"]
        B_["B: Border irregularity"]
        C_["C: Color variation"]
        D_["D: Diameter > 6mm"]
    end

    subgraph Features["📊 Our Tabular Features"]
        AF["tbp_lv_symm_2axis<br/>tbp_lv_eccentricity"]
        BF["tbp_lv_norm_border<br/>tbp_lv_area_perim_ratio"]
        CF["tbp_lv_deltaL/A/B<br/>tbp_lv_color_std_mean<br/>tbp_lv_radial_color_std_max"]
        DF["clin_size_long_diam_mm<br/>tbp_lv_areaMM2<br/>tbp_lv_perimeterMM"]
    end

    A_ --> AF
    B_ --> BF
    C_ --> CF
    D_ --> DF
```

---

## GBDT Stacking (Stage 2)

The GBDT stacking layer is inspired by the 1st place ISIC 2024 solution. Instead of relying solely on CNN predictions, we train gradient-boosted trees on top of CNN outputs combined with tabular features.

```mermaid
flowchart LR
    subgraph CNN_Probs["🧠 CNN Probabilities"]
        P1["EfficientNet-B0<br/>p = 0.72"]
        P2["ConvNeXt-Large<br/>p = 0.68"]
    end

    subgraph Tabular["📊 Tabular + Patient-Relative"]
        TAB["43 tabular features"]
        UG["Ugly Duckling Sign<br/>ratio / diff / z-score"]
    end

    subgraph GBDT_Models["🌲 GBDT Ensemble"]
        LGB["LightGBM<br/>3 seeds × 5 folds"]
        XGB["XGBoost<br/>3 seeds × 5 folds"]
        CAT["CatBoost<br/>3 seeds × 5 folds"]
    end

    FINAL["📊 Final Prediction<br/>average of 45 GBDTs"]

    P1 & P2 --> LGB & XGB & CAT
    TAB & UG --> LGB & XGB & CAT
    LGB & XGB & CAT --> FINAL
```

### Key Techniques

- **Noise injection**: Gaussian noise (σ=0.1) added to CNN probabilities during GBDT training to prevent over-reliance on CNN predictions
- **Patient-relative features**: For each CNN probability, ratio/diff/z-score vs. patient mean — implementing the "ugly duckling" sign
- **Multi-seed training**: Each GBDT type trained with 3 seeds (42, 123, 456) for variance reduction
- **Optimal threshold**: Each GBDT finds its best threshold using Youden's J statistic (max TPR − FPR)

---

## Tools & Libraries

| Tool | What it does | Why we use it |
|------|-------------|---------------|
| **PyTorch** | Deep learning framework | Core tensor operations, autograd, GPU support |
| **Lightning** | Training framework | Handles training loop, logging, checkpoints, multi-GPU |
| **TIMM** | Pre-trained model zoo | 700+ ImageNet-pretrained backbones |
| **Hydra** | Configuration management | Mix-and-match experiment configs |
| **Albumentations** | Image augmentation | Fast, GPU-friendly augmentation pipeline |
| **torchmetrics** | Metric computation | Efficient AUROC and ROC curve computation |
| **WandB** | Experiment tracking | Training curves, ROC plots, hyperparameters dashboard |
| **h5py** | HDF5 file I/O | Reads images from competition's HDF5 format |
| **LightGBM** | Gradient boosting | Fast GBDT for Stage 2 stacking |
| **XGBoost** | Gradient boosting | Mature GBDT implementation for stacking |
| **CatBoost** | Gradient boosting | GBDT with built-in class balancing |
| **scikit-learn** | ML utilities | AUROC, ROC curve, metrics for GBDT evaluation |
| **Gradio** | Web UI | Interactive prediction demo with pipeline visualization |

---

## How to Run

### Stage 1: CNN Training (on Kaggle)

1. Open `notebooks/skin-cancer-detection.ipynb` in Kaggle
2. Attach the `isic-2024-challenge` dataset
3. Set `MODELS_TO_TRAIN` and `FOLDS_TO_TRAIN` in Cell 4
4. Run the CNN training cells → checkpoints saved

### Stage 2: GBDT Stacking

After CNN training, run the GBDT cells in the notebook or from CLI:

```bash
# Extract CNN predictions
python src/gbdt/extract_cnn_features.py \
    --checkpoint-dir checkpoints/ \
    --data-dir data/isic-2024-challenge

# Train GBDT models
python src/gbdt/train_gbdt.py \
    --features-dir outputs/gbdt_features \
    --output-dir checkpoints/gbdt \
    --seeds 42 123 456
```

### CNN Training (Local)

```bash
python src/train.py experiment=isic_efficientnet_b0 data.fold=0 logger=wandb
```

### Submission (on Kaggle)

1. Open `notebooks/submission.ipynb`
2. Attach competition data + CNN checkpoints + GBDT checkpoints as datasets
3. Run all cells → `submission.csv` generated (uses GBDT stacking if available)
4. Submit the notebook

> 📖 [Technical reference](reference.md) &nbsp;|&nbsp; [Gradio demo guide](gradio-demo.md)
