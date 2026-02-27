"""Train GBDT models (LightGBM, XGBoost, CatBoost) for stacking.

Stage 2 of the CNN → GBDT pipeline. Takes extracted CNN predictions
and tabular features from extract_cnn_features.py, trains gradient-boosted
trees on top, and saves the trained models.

The GBDT models learn to combine CNN probability outputs with raw tabular
features for improved skin cancer classification. Patient-relative features
are also computed to implement the 'ugly duckling' sign.

Usage:
    python src/gbdt/train_gbdt.py \
        --features-dir outputs/gbdt_features \
        --output-dir checkpoints/gbdt \
        --seeds 42 123 456 \
        --noise-sigma 0.1
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore", category=UserWarning)

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


# =========================================================================
#  FEATURE ENGINEERING
# =========================================================================

def add_patient_relative_features(df: pd.DataFrame, prob_cols: List[str]) -> pd.DataFrame:
    """Add patient-relative features (ugly duckling sign).

    For each CNN probability column, compute:
    - ratio of lesion's prediction to patient's mean prediction
    - difference from patient mean
    - z-score within patient (how many stds from patient mean)

    These features capture whether this lesion is unusual compared to the
    patient's other lesions — the dermatological 'ugly duckling' sign.

    Args:
        df: DataFrame with patient_id and CNN probability columns
        prob_cols: List of column names containing CNN probabilities

    Returns:
        DataFrame with additional patient-relative feature columns
    """
    df = df.copy()

    for col in prob_cols:
        if col not in df.columns:
            continue

        # Patient mean and std for this CNN's predictions
        p_mean = df.groupby("patient_id")[col].transform("mean")
        p_std = df.groupby("patient_id")[col].transform("std").fillna(0)
        p_count = df.groupby("patient_id")[col].transform("count")

        # Ratio: how does this lesion compare to the patient average?
        df[f"{col}_patient_ratio"] = df[col] / (p_mean + 1e-8)

        # Difference from patient mean
        df[f"{col}_patient_diff"] = df[col] - p_mean

        # Z-score within patient (only meaningful for multi-lesion patients)
        safe_std = p_std.clip(lower=1e-8)
        df[f"{col}_patient_zscore"] = np.where(
            p_count > 1,
            (df[col] - p_mean) / safe_std,
            0.0  # Single-lesion patients get 0
        )

    return df


def inject_noise(df: pd.DataFrame, prob_cols: List[str], sigma: float, seed: int) -> pd.DataFrame:
    """Add Gaussian noise to CNN probability features during training.

    This prevents the GBDT from over-relying on CNN predictions and encourages
    it to learn from tabular features too. Noise is only applied to the training
    split; validation predictions remain clean.

    From the 1st place solution methodology.

    Args:
        df: DataFrame with CNN probability columns
        prob_cols: Columns to add noise to
        sigma: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        DataFrame with noised probability columns (train split only)
    """
    df = df.copy()
    rng = np.random.RandomState(seed)

    train_mask = df["split"] == "train"
    for col in prob_cols:
        if col in df.columns:
            noise = rng.normal(0, sigma, size=train_mask.sum())
            df.loc[train_mask, col] = np.clip(df.loc[train_mask, col] + noise, 0, 1)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify all feature columns for GBDT training.

    Returns columns that are:
    - tabular features (tab_*)
    - CNN probability columns (*_prob)
    - Patient-relative features (*_patient_ratio, *_patient_diff, *_patient_zscore)

    Excludes: isic_id, patient_id, target, split, fold
    """
    exclude = {"isic_id", "patient_id", "target", "split", "fold"}
    return [c for c in df.columns if c not in exclude]


# =========================================================================
#  OPTIMAL THRESHOLD
# =========================================================================

def compute_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Find optimal classification threshold using Youden's J statistic.

    The threshold that maximizes TPR - FPR, which is the point on the ROC curve
    closest to perfect classification (top-left corner).

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities

    Returns:
        (optimal_threshold, best_auroc)
    """
    auroc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Youden's J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]

    return float(optimal_threshold), float(auroc)


# =========================================================================
#  GBDT TRAINING
# =========================================================================

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    seed: int,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train a LightGBM model."""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "random_state": seed,
        "verbose": -1,
        "is_unbalance": True,
    }
    params.update(kwargs)

    n_estimators = params.pop("n_estimators")
    random_state = params.pop("random_state")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        **params,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    threshold, auroc = compute_optimal_threshold(y_val, y_pred)

    info = {
        "type": "lightgbm",
        "auroc": auroc,
        "threshold": threshold,
        "seed": seed,
        "best_iteration": model.best_iteration_,
        "feature_names": feature_names,
    }

    return model, info


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    seed: int,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train an XGBoost model."""
    import xgboost as xgb

    # Compute scale_pos_weight for class imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 1000,
        "random_state": seed,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 0,
        "tree_method": "hist",
    }
    params.update(kwargs)

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    threshold, auroc = compute_optimal_threshold(y_val, y_pred)

    info = {
        "type": "xgboost",
        "auroc": auroc,
        "threshold": threshold,
        "seed": seed,
        "best_iteration": model.best_iteration,
        "feature_names": feature_names,
    }

    return model, info


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    seed: int,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Train a CatBoost model."""
    from catboost import CatBoostClassifier

    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "random_seed": seed,
        "verbose": 0,
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "early_stopping_rounds": 50,
    }
    params.update(kwargs)

    model = CatBoostClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=0,
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    threshold, auroc = compute_optimal_threshold(y_val, y_pred)

    info = {
        "type": "catboost",
        "auroc": auroc,
        "threshold": threshold,
        "seed": seed,
        "best_iteration": model.best_iteration_,
        "feature_names": feature_names,
    }

    return model, info


TRAINERS = {
    "lightgbm": train_lightgbm,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
}


def save_gbdt_model(model: Any, info: Dict, output_dir: str, name: str) -> str:
    """Save a trained GBDT model and its metadata.

    Args:
        model: Trained GBDT model
        info: Metadata dict (type, auroc, threshold, etc.)
        output_dir: Output directory
        name: Model filename (without extension)

    Returns:
        Path to saved model file
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{name}.pkl")
    info_path = os.path.join(output_dir, f"{name}_info.json")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return model_path


# =========================================================================
#  MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train GBDT models for stacking on CNN predictions"
    )
    parser.add_argument(
        "--features-dir", type=str, default="outputs/gbdt_features",
        help="Directory with fold_*_features.csv from extract_cnn_features.py"
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/gbdt",
        help="Output directory for trained GBDT models"
    )
    parser.add_argument(
        "--gbdt-types", type=str, nargs="+",
        default=["lightgbm", "xgboost", "catboost"],
        choices=["lightgbm", "xgboost", "catboost"],
        help="Which GBDT types to train"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds for variance reduction"
    )
    parser.add_argument(
        "--noise-sigma", type=float, default=0.1,
        help="Std of Gaussian noise added to CNN probs during training"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of folds"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_models = len(args.gbdt_types) * args.n_folds * len(args.seeds)
    print(f"🌲 GBDT Stacking Training")
    print(f"   Types: {args.gbdt_types}")
    print(f"   Folds: {args.n_folds}")
    print(f"   Seeds: {args.seeds}")
    print(f"   Noise σ: {args.noise_sigma}")
    print(f"   Total models to train: {total_models}")

    all_results: List[Dict] = []

    for fold in range(args.n_folds):
        features_path = os.path.join(args.features_dir, f"fold_{fold}_features.csv")
        if not os.path.exists(features_path):
            print(f"\n⚠️  Features not found for fold {fold}: {features_path}")
            continue

        print(f"\n{'='*60}")
        print(f"📂 FOLD {fold}")
        print(f"{'='*60}")

        df = pd.read_csv(features_path)
        prob_cols = [c for c in df.columns if c.endswith("_prob")]
        print(f"  CNN probability columns: {prob_cols}")

        # Add patient-relative features
        df = add_patient_relative_features(df, prob_cols)

        for seed in args.seeds:
            # Add noise to CNN predictions (train only)
            df_noised = inject_noise(df, prob_cols, args.noise_sigma, seed)

            # Get feature columns
            feature_cols = get_feature_columns(df_noised)
            print(f"\n  Seed={seed} | Features: {len(feature_cols)} columns")

            # Split train / val
            train_mask = df_noised["split"] == "train"
            val_mask = df_noised["split"] == "val"

            X_train = df_noised.loc[train_mask, feature_cols].values.astype(np.float32)
            y_train = df_noised.loc[train_mask, "target"].values.astype(np.float32)
            X_val = df_noised.loc[val_mask, feature_cols].values.astype(np.float32)
            y_val = df_noised.loc[val_mask, "target"].values.astype(np.float32)

            # Handle NaN/Inf
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)

            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Pos rate: {y_train.mean():.4f}")

            for gbdt_type in args.gbdt_types:
                model_name = f"{gbdt_type}_fold{fold}_seed{seed}"
                print(f"\n  🌲 Training {model_name}...")

                try:
                    trainer_fn = TRAINERS[gbdt_type]
                    model, info = trainer_fn(
                        X_train, y_train, X_val, y_val,
                        feature_names=feature_cols,
                        seed=seed,
                    )

                    # Add fold info
                    info["fold"] = fold
                    info["model_name"] = model_name
                    info["n_features"] = len(feature_cols)
                    info["n_cnn_models"] = len(prob_cols)

                    # Save model
                    save_path = save_gbdt_model(model, info, args.output_dir, model_name)
                    print(f"     ✅ AUROC={info['auroc']:.4f} | thr={info['threshold']:.4f} | saved: {save_path}")

                    all_results.append(info)

                except ImportError as e:
                    print(f"     ❌ {gbdt_type} not installed: {e}")
                    print(f"        Install: pip install {gbdt_type}")
                except Exception as e:
                    print(f"     ❌ Error training {model_name}: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"📊 TRAINING SUMMARY")
        print(f"{'='*60}")

        results_df = pd.DataFrame(all_results)
        print(f"\nTotal models trained: {len(results_df)}")
        print(f"\nAUROC by type:")
        for gbdt_type in args.gbdt_types:
            type_results = results_df[results_df["type"] == gbdt_type]
            if len(type_results) > 0:
                print(f"  {gbdt_type}: mean={type_results['auroc'].mean():.4f} "
                      f"±{type_results['auroc'].std():.4f} "
                      f"(min={type_results['auroc'].min():.4f}, max={type_results['auroc'].max():.4f})")

        # Save summary
        summary_path = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Summary saved to {summary_path}")

    print(f"\n✅ GBDT training complete!")


if __name__ == "__main__":
    main()
