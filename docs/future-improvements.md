# 🔮 Future Improvements — Analysis of Winning Solutions

> **[← Back to README](../README.md)**

Analysis of the [1st place](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ilya-novoselskiy-1st-place-solution) and [2nd place](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/yakiniku-2nd-place-solution) ISIC 2024 solutions, with actionable improvements prioritized by expected impact and implementation complexity.

---

## Current Approach vs. Winning Solutions

| Aspect | Our Approach | 1st Place (Ilya) | 2nd Place (Yakiniku) |
|--------|-------------|-------------------|----------------------|
| **Image models** | Single CNN backbone | EVA02-small + EdgeNeXt-base | 9 models (EfficientNet, MaxViT, etc.) |
| **Tabular handling** | Concat fusion MLP | GBDT stacking on CNN predictions | Two-stage: CNN features → GBDT ensemble |
| **Ensemble size** | 5-fold × 1 model | 150 models (5 folds × 10 seeds × 3 GBDT types) | 54 GBDT models |
| **Patient context** | None | Patient-relative features + LOF | "Ugly duckling" patient-wise features |
| **Class imbalance** | pos_weight | 1:1 batch sampling | Triple stratification |
| **CV strategy** | Stratified Group KFold | Same + 10-seed averaging + t-tests | Same + triple stratification |

> [!IMPORTANT]
> **Key insight from both winners**: The biggest gains came NOT from better image models, but from **how they used tabular features and patient context**. Both solutions treat CNN outputs as just one more feature to feed into gradient-boosted trees.

---

## 1st Place Solution — Ilya Novoselskiy

### Core Strategy

A hybrid **image + GBDT stacking** pipeline with 150 models in the final ensemble.

### Key Techniques

#### Patient-Relative Features (🔥 High Impact)

Instead of using raw predictions, compute how a lesion compares to the patient's average:

```python
# For each lesion: ratio of its prediction to patient's mean prediction
patient_avg = df.groupby('patient_id')['pred'].transform('mean')
df['pred_relative'] = df['pred'] / patient_avg
```

**Why it works**: A mole that scores 0.3 on a patient whose average is 0.05 is far more suspicious than 0.3 on a patient averaging 0.25.

#### Local Outlier Factor (LOF) Anomaly Detection

Used LOF scores on top CatBoost features per patient to identify anomalous moles. This implements the dermatological **"ugly duckling" sign** — a lesion that looks different from a patient's other lesions is suspicious.

#### GBDT Stacking with Noise Injection

CNN predictions (standardized) are fed as features into LightGBM/XGBoost/CatBoost. To prevent the GBDTs from over-relying on the image model's predictions:

```python
# Add Gaussian noise during GBDT training (not at test time)
train_df['img_pred'] += np.random.normal(0, 0.1, len(train_df))
```

#### Statistical CV Validation

Ran experiments with **10 different random seeds** and used a **t-test** to determine if improvements were statistically significant before trusting them. This is critical with such small positive class counts.

### What Didn't Work for 1st Place

- ❌ Hard-example mining (selecting negatives by LogLoss)
- ❌ Extensive pretraining on other data sources
- ❌ Test-Time Augmentation (TTA)

---

## 2nd Place Solution — Yakiniku

### Core Strategy

A **two-stage pipeline**: Stage 1 trains image models to extract meta-features, Stage 2 feeds these into a massive GBDT ensemble alongside engineered tabular features.

### Key Techniques

#### Multiple Training Setups for Diversity (🔥 High Impact)

Instead of training all models the same way, they used **5 different training setups**:

| Setup | Description |
|-------|------------|
| 0 | Standard supervised training |
| 1 | Training with **Mixup** augmentation |
| 2 | **Multimodal** — auxiliary loss to predict tabular features from images |
| 3 | **Clustering** — auxiliary loss for diagnosis-type clusters |
| 4 | **TIP** (Tabular-Image Pre-training) — self-supervised pretraining using tabular data |

> [!TIP]
> **TIP Pre-training** is particularly interesting — it uses tabular features to create a pretext task for image model pre-training. The idea is that if a model can predict clinical measurements from an image, it has learned clinically meaningful features.

#### "Ugly Duckling" Patient-Wise Features

```python
# For each lesion, calculate distance from patient's mean feature vector
patient_means = df.groupby('patient_id')[feature_cols].transform('mean')
df['ugly_duckling_score'] = np.linalg.norm(
    df[feature_cols].values - patient_means.values, axis=1
)
```

Lesions that are statistical outliers within a patient's set are flagged as suspicious.

#### Patient-Wise Feature Standardization

Standardize features **within each patient** rather than across the whole dataset. This way, a lesion's features are always relative to that specific patient's "normal":

```python
# Per-patient standardization
patient_std = df.groupby('patient_id')[feature_cols].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-8)
)
```

#### Triple Stratification

Their CV was stratified on three axes:
1. Whether the patient has ANY malignant images
2. The PROPORTION of malignant images per patient
3. The distribution of number of images per patient (binned)

#### Massive GBDT Ensemble

- 18 variants each of LightGBM, XGBoost, and CatBoost = **54 total models**
- Variations created by using different subsets of image meta-features
- 5-fold seed averaging for stability

### What Didn't Work for 2nd Place

- ❌ Past ISIC competition data (domain shift was too strong — could distinguish 2024 from older data with 0.99 AUC)

---

## Prioritized Improvement Roadmap

Ranked by **expected impact** and **implementation effort**:

### 🔴 High Priority — Biggest Gains

| # | Improvement | Estimated Impact | Effort | Source |
|---|------------|-----------------|--------|---------|
| 1 | **Add GBDT stacking** — use CNN probabilities as features in LightGBM/CatBoost | Very High | Medium | Both |
| 2 | **Patient-relative features** — compute lesion prediction relative to patient average | Very High | Low | 1st |
| 3 | **Multi-seed CV validation** — run with 10 seeds, use t-test for significance | High | Low | 1st |
| 4 | **"Ugly duckling" scoring** — LOF or distance-based anomaly detection per patient | High | Medium | Both |

### 🟡 Medium Priority — Meaningful Gains

| # | Improvement | Estimated Impact | Effort | Source |
|---|------------|-----------------|--------|---------|
| 5 | **Patient-wise feature standardization** | Medium | Low | 2nd |
| 6 | **Multi-model diversity** — train with different setups (Mixup, multimodal loss) | Medium | Medium | 2nd |
| 7 | **Triple stratification CV** | Medium | Low | 2nd |
| 8 | **Noise injection in stacking** — add Gaussian noise to image features during GBDT training | Medium | Low | 1st |

### 🟢 Lower Priority — Polish

| # | Improvement | Estimated Impact | Effort | Source |
|---|------------|-----------------|--------|---------|
| 9 | **TIP pre-training** — self-supervised image pre-training using tabular data | Medium | High | 2nd |
| 10 | **External data with domain adaptation** — use prior ISIC competitions with histogram matching | Low-Medium | High | 1st |
| 11 | **More image architectures** — add MaxViT, EdgeNeXt for ensemble diversity | Low-Medium | Medium | Both |
| 12 | **1:1 batch sampling** for image models | Low | Low | 1st |

---

## Quick Wins to Implement First

> [!TIP]
> **Start here** — these changes require minimal code but can significantly improve scores:

1. **Patient-relative predictions (Item #2)**: After getting per-lesion predictions, divide by patient mean. This is a post-processing step that doesn't require retraining.

2. **GBDT stacking (Item #1)**: Train a single CatBoost model on top of your current CNN predictions + raw tabular features. Even a basic version often outperforms the CNN alone.

3. **Multi-seed validation (Item #3)**: Instead of training once, train 3-5 times with different seeds and average predictions. This reduces variance and gives more reliable CV scores.

---

## References

- **1st Place Code**: Solution details in the [Kaggle writeup](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ilya-novoselskiy-1st-place-solution)
- **2nd Place Code**: [GitHub — uchiyama33/isic-2024-2nd-place](https://github.com/uchiyama33/isic-2024-2nd-place)
- **Competition Page**: [ISIC 2024 — Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge)

> 📖 [Architecture docs](architecture.md) &nbsp;|&nbsp; [Technical reference](reference.md)
