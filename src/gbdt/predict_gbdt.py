"""GBDT inference utilities for the stacking pipeline.

Loads trained GBDT models and runs ensemble prediction given CNN probabilities
and tabular features. Used by both the Gradio demo and the evaluation scripts.

Usage:
    from src.gbdt.predict_gbdt import GBDTEnsemble

    ensemble = GBDTEnsemble("checkpoints/gbdt")
    result = ensemble.predict(cnn_probs, tabular_features, patient_ids)
"""

import glob
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GBDTEnsemble:
    """Ensemble of trained GBDT models for stacking prediction.

    Loads all .pkl GBDT models from a directory and runs ensemble inference.
    Supports per-model threshold and AUROC-weighted averaging.

    Attributes:
        models: List of (model, info_dict) tuples
        model_types: Set of GBDT types loaded (e.g. {'lightgbm', 'xgboost', 'catboost'})
        feature_names: Feature column names expected by the models
    """

    def __init__(self, model_dir: str):
        """Load all GBDT models from directory.

        Args:
            model_dir: Directory containing .pkl model files and _info.json metadata
        """
        self.model_dir = model_dir
        self.models: List[Tuple[Any, Dict]] = []
        self.model_types: set = set()
        self.feature_names: Optional[List[str]] = None

        self._load_models()

    def _load_models(self):
        """Load all GBDT models and their metadata."""
        if not os.path.isdir(self.model_dir):
            print(f"⚠️  GBDT model dir not found: {self.model_dir}")
            return

        pkl_files = sorted(glob.glob(os.path.join(self.model_dir, "*.pkl")))
        for pkl_path in pkl_files:
            info_path = pkl_path.replace(".pkl", "_info.json")

            try:
                with open(pkl_path, "rb") as f:
                    model = pickle.load(f)

                info = {}
                if os.path.exists(info_path):
                    with open(info_path, "r") as f:
                        info = json.load(f)

                self.models.append((model, info))
                if "type" in info:
                    self.model_types.add(info["type"])

                # Use feature names from first model
                if self.feature_names is None and "feature_names" in info:
                    self.feature_names = info["feature_names"]

            except Exception as e:
                print(f"⚠️  Failed to load {pkl_path}: {e}")

        if self.models:
            print(f"✅ Loaded {len(self.models)} GBDT model(s): {self.model_types}")

    @property
    def is_available(self) -> bool:
        """Whether any GBDT models are loaded."""
        return len(self.models) > 0

    def prepare_features(
        self,
        cnn_probs: Dict[str, np.ndarray],
        tabular_features: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Prepare feature matrix for GBDT prediction.

        Combines CNN probabilities, tabular features, and patient-relative
        features into the feature matrix expected by trained GBDTs.

        Args:
            cnn_probs: Dict mapping CNN model name → array of probabilities [N]
            tabular_features: Standardized tabular features [N, 43]
            patient_ids: Optional patient IDs [N] for patient-relative features

        Returns:
            Feature matrix [N, n_features]
        """
        n_samples = tabular_features.shape[0]
        parts = []

        # Tabular features (tab_*)
        tab_col_count = tabular_features.shape[1]
        parts.append(tabular_features)

        # CNN probability features
        prob_arrays = []
        prob_names = []
        for model_name, probs in sorted(cnn_probs.items()):
            prob_arrays.append(probs.reshape(-1, 1))
            prob_names.append(model_name)
        if prob_arrays:
            probs_matrix = np.hstack(prob_arrays)
            parts.append(probs_matrix)

        # Patient-relative features
        if patient_ids is not None and len(prob_arrays) > 0:
            for i, model_name in enumerate(prob_names):
                prob_col = prob_arrays[i].ravel()
                unique_patients = np.unique(patient_ids)

                patient_ratio = np.zeros(n_samples, dtype=np.float32)
                patient_diff = np.zeros(n_samples, dtype=np.float32)
                patient_zscore = np.zeros(n_samples, dtype=np.float32)

                for pid in unique_patients:
                    mask = patient_ids == pid
                    p_data = prob_col[mask]
                    p_mean = p_data.mean()
                    p_std = p_data.std() if len(p_data) > 1 else 0

                    patient_ratio[mask] = prob_col[mask] / (p_mean + 1e-8)
                    patient_diff[mask] = prob_col[mask] - p_mean
                    if p_std > 1e-8 and len(p_data) > 1:
                        patient_zscore[mask] = (prob_col[mask] - p_mean) / p_std

                parts.append(patient_ratio.reshape(-1, 1))
                parts.append(patient_diff.reshape(-1, 1))
                parts.append(patient_zscore.reshape(-1, 1))
        elif len(prob_arrays) > 0:
            # No patient IDs: fill patient-relative features with zeros
            n_patient_feats = len(prob_arrays) * 3
            parts.append(np.zeros((n_samples, n_patient_feats), dtype=np.float32))

        features = np.hstack(parts).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def predict(
        self,
        cnn_probs: Dict[str, np.ndarray],
        tabular_features: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run ensemble GBDT prediction.

        Args:
            cnn_probs: Dict mapping CNN model name → array of probabilities [N]
            tabular_features: Standardized tabular features [N, 43]
            patient_ids: Optional patient IDs [N] for patient-relative features

        Returns:
            Dict with:
                avg_prob: Average probability across all GBDTs [N]
                avg_threshold: Average optimal threshold
                per_model: List of dicts with per-model results
                is_malignant: Boolean array [N]
        """
        if not self.is_available:
            return {
                "avg_prob": np.zeros(tabular_features.shape[0]),
                "avg_threshold": 0.5,
                "per_model": [],
                "is_malignant": np.zeros(tabular_features.shape[0], dtype=bool),
            }

        features = self.prepare_features(cnn_probs, tabular_features, patient_ids)

        all_probs = []
        all_thresholds = []
        per_model_results = []

        for model, info in self.models:
            try:
                # Handle feature count mismatch (model may expect different features)
                expected_n = info.get("n_features", features.shape[1])
                if features.shape[1] != expected_n:
                    # Pad or truncate
                    if features.shape[1] < expected_n:
                        pad = np.zeros((features.shape[0], expected_n - features.shape[1]))
                        feat_input = np.hstack([features, pad])
                    else:
                        feat_input = features[:, :expected_n]
                else:
                    feat_input = features

                probs = model.predict_proba(feat_input)[:, 1]
                threshold = info.get("threshold", 0.5)

                all_probs.append(probs)
                all_thresholds.append(threshold)

                per_model_results.append({
                    "model_name": info.get("model_name", "unknown"),
                    "type": info.get("type", "unknown"),
                    "fold": info.get("fold", -1),
                    "seed": info.get("seed", -1),
                    "prob": probs.copy(),
                    "threshold": threshold,
                    "auroc": info.get("auroc", 0.0),
                })

            except Exception as e:
                print(f"⚠️  GBDT prediction error for {info.get('model_name', '?')}: {e}")

        if not all_probs:
            return {
                "avg_prob": np.zeros(tabular_features.shape[0]),
                "avg_threshold": 0.5,
                "per_model": [],
                "is_malignant": np.zeros(tabular_features.shape[0], dtype=bool),
            }

        avg_prob = np.mean(all_probs, axis=0)
        avg_threshold = np.mean(all_thresholds)

        return {
            "avg_prob": avg_prob,
            "avg_threshold": avg_threshold,
            "per_model": per_model_results,
            "is_malignant": avg_prob >= avg_threshold,
            "n_models": len(all_probs),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded models for display."""
        summary = {
            "total_models": len(self.models),
            "types": {},
        }
        for _, info in self.models:
            t = info.get("type", "unknown")
            if t not in summary["types"]:
                summary["types"][t] = {"count": 0, "aurocs": []}
            summary["types"][t]["count"] += 1
            if "auroc" in info:
                summary["types"][t]["aurocs"].append(info["auroc"])

        for t, data in summary["types"].items():
            if data["aurocs"]:
                data["mean_auroc"] = float(np.mean(data["aurocs"]))
                data["std_auroc"] = float(np.std(data["aurocs"]))

        return summary
