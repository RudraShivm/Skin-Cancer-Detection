"""Extract CNN predictions and features for GBDT stacking.

This script runs each trained CNN checkpoint on the full training dataset,
collecting per-image:
  - CNN probability (sigmoid output from the fusion head / classifier)
  - Raw tabular features (43-dim, patient-standardized)
  - Patient ID and target label

The output is a set of CSVs in outputs/gbdt_features/ that serve as input
to train_gbdt.py for the second-stage GBDT stacking.

Usage:
    python src/gbdt/extract_cnn_features.py \
        --checkpoint-dir checkpoints/ \
        --data-dir data/isic-2024-challenge \
        --output-dir outputs/gbdt_features \
        --n-folds 5
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.isic_datamodule import ISICDataModule, TABULAR_NUM_COLS
from src.models.isic_module import ISICLitModule


def get_model_img_size(model: ISICLitModule) -> int:
    """Detect correct image size for a CNN checkpoint."""
    try:
        import timm
        data_config = timm.data.resolve_data_config(model.model.pretrained_cfg)
        return data_config.get("input_size", (3, 224, 224))[-1]
    except Exception:
        return 256


def find_best_checkpoint(model_dir: str, fold: int) -> Optional[str]:
    """Find best checkpoint (highest AUROC) for a model/fold combo."""
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    if not os.path.isdir(fold_dir):
        return None

    auroc_ckpts = sorted(glob.glob(os.path.join(fold_dir, "epoch_*_auroc_*.ckpt")))
    if auroc_ckpts:
        def extract_auroc(path: str) -> float:
            try:
                return float(os.path.basename(path).split("auroc_")[-1].replace(".ckpt", ""))
            except Exception:
                return 0.0
        return max(auroc_ckpts, key=extract_auroc)

    last = os.path.join(fold_dir, "last.ckpt")
    if os.path.exists(last):
        return last

    any_ckpts = glob.glob(os.path.join(fold_dir, "*.ckpt"))
    return any_ckpts[0] if any_ckpts else None


def discover_models(checkpoint_dir: str) -> List[str]:
    """Discover all model architecture directories under checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir):
        return []
    return sorted([
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and not d.startswith(".")
    ])


def extract_predictions(
    model: ISICLitModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], np.ndarray]:
    """Run CNN inference and collect per-image probabilities.

    Args:
        model: Loaded ISICLitModule in eval mode
        dataloader: DataLoader yielding batches with 'image', 'tabular', 'isic_id'
        device: torch device

    Returns:
        (isic_ids, probabilities) where probabilities is [N] array of sigmoid outputs
    """
    model.eval()
    model.to(device)

    all_ids: List[str] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Extracting", leave=False):
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device) if model.fusion_head is not None else None
            isic_ids = batch["isic_id"]

            logits = model(img, tab).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()

            if probs.ndim == 0:
                probs = np.array([probs.item()])

            all_ids.extend(isic_ids)
            all_probs.extend(probs.tolist())

    return all_ids, np.array(all_probs, dtype=np.float32)


def extract_features_for_fold(
    checkpoint_dir: str,
    model_names: List[str],
    fold: int,
    data_dir: str,
    n_folds: int,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Optional[pd.DataFrame]:
    """Extract CNN predictions from all model checkpoints for one fold.

    For each model, loads the fold's best checkpoint and runs inference on
    both the training and validation splits. The CNN probability is stored
    as a column named '{model_name}_prob'.

    Returns:
        DataFrame with columns: isic_id, patient_id, target, split,
        tabular features (43 columns), and one {model}_prob column per CNN.
        Returns None if no checkpoints are found.
    """
    # Set up data module for this fold
    dm = ISICDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        fold=fold,
        n_folds=n_folds,
        img_size=224,  # Will be overridden per model
    )
    dm.setup(stage="fit")

    # Get metadata from data module
    metadata = pd.read_csv(dm.metadata_path, low_memory=False)

    # Get train/val ISIC IDs
    train_ids = set(dm.data_train.isic_ids)
    val_ids = set(dm.data_val.isic_ids)

    # Build base dataframe with tabular features and metadata
    all_ids = dm.data_train.isic_ids + dm.data_val.isic_ids
    all_targets = dm.data_train.targets + dm.data_val.targets
    all_tabular = np.vstack([dm.data_train.tabular_features, dm.data_val.tabular_features])

    # Column names for standardized tabular features
    tab_col_names = (
        [f"tab_{c}" for c in TABULAR_NUM_COLS]
        + ["tab_sex"]
        + [f"tab_site_{i}" for i in range(7)]
    )

    base_df = pd.DataFrame({
        "isic_id": all_ids,
        "target": all_targets,
        "split": ["train" if iid in train_ids else "val" for iid in all_ids],
    })

    # Add patient_id from metadata
    pid_map = metadata.set_index("isic_id")["patient_id"].to_dict()
    base_df["patient_id"] = base_df["isic_id"].map(pid_map)

    # Add tabular features
    tab_df = pd.DataFrame(all_tabular, columns=tab_col_names)
    base_df = pd.concat([base_df, tab_df], axis=1)

    # Extract CNN predictions for each model architecture
    found_any = False
    for model_name in model_names:
        ckpt_path = find_best_checkpoint(
            os.path.join(checkpoint_dir, model_name), fold
        )
        if ckpt_path is None:
            print(f"  ⚠️  No checkpoint for {model_name}/fold_{fold}, skipping")
            continue

        print(f"  📦 Loading {model_name}/fold_{fold}: {os.path.basename(ckpt_path)}")
        try:
            model = ISICLitModule.load_ckpt(ckpt_path, device)
            img_size = get_model_img_size(model)

            # Rebuild the data module with the correct image size for this model
            dm_model = ISICDataModule(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                fold=fold,
                n_folds=n_folds,
                img_size=img_size,
            )
            dm_model.setup(stage="fit")

            # Combine train+val into one dataloader for extraction
            from torch.utils.data import ConcatDataset
            combined_ds = ConcatDataset([dm_model.data_train, dm_model.data_val])
            combined_loader = DataLoader(
                combined_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=True,
            )

            isic_ids, probs = extract_predictions(model, combined_loader, device)

            # Map predictions back by isic_id
            pred_map = dict(zip(isic_ids, probs))
            col_name = f"{model_name}_prob"
            base_df[col_name] = base_df["isic_id"].map(pred_map)
            found_any = True

            # Report AUROC for this model on validation set
            val_mask = base_df["split"] == "val"
            if val_mask.sum() > 0 and col_name in base_df.columns:
                val_preds = base_df.loc[val_mask, col_name].dropna()
                val_targets = base_df.loc[val_mask, "target"]
                if len(val_preds) > 0:
                    from sklearn.metrics import roc_auc_score
                    try:
                        auroc = roc_auc_score(val_targets.loc[val_preds.index], val_preds)
                        print(f"    ✅ {model_name} fold_{fold} val AUROC: {auroc:.4f}")
                    except Exception:
                        pass

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ Error loading {model_name}/fold_{fold}: {e}")
            import traceback
            traceback.print_exc()

    if not found_any:
        return None

    return base_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract CNN predictions for GBDT stacking"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory containing model subdirectories with fold checkpoints"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/isic-2024-challenge",
        help="Path to ISIC data directory (with train-image.hdf5 and metadata)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/gbdt_features",
        help="Output directory for extracted feature CSVs"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of dataloader workers"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Data dir: {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover model architectures
    model_names = discover_models(args.checkpoint_dir)
    if not model_names:
        print(f"❌ No model directories found in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"\n🔍 Found {len(model_names)} model architecture(s): {model_names}")

    # Extract features for each fold
    for fold in range(args.n_folds):
        print(f"\n{'='*60}")
        print(f"📂 FOLD {fold}")
        print(f"{'='*60}")

        df = extract_features_for_fold(
            checkpoint_dir=args.checkpoint_dir,
            model_names=model_names,
            fold=fold,
            data_dir=args.data_dir,
            n_folds=args.n_folds,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if df is not None:
            out_path = os.path.join(args.output_dir, f"fold_{fold}_features.csv")
            df.to_csv(out_path, index=False)
            print(f"  💾 Saved {len(df)} rows to {out_path}")

            # Summary
            prob_cols = [c for c in df.columns if c.endswith("_prob")]
            print(f"  📊 CNN probability columns: {prob_cols}")
        else:
            print(f"  ⚠️ No features extracted for fold {fold}")

    print(f"\n✅ Feature extraction complete. Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
