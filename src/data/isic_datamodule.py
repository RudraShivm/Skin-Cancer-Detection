"""ISIC 2024 DataModule with tabular feature support.

This module loads images from HDF5 and tabular features from the metadata CSV.
Tabular features are standardized (zero mean, unit variance) using statistics
computed from the training set. Categorical features (sex, anatom_site_general)
are label-encoded before being fed to the model.

Reference: Yakiniku 2nd place solution used tabular features (age, site,
color metrics, lesion geometry) alongside image features.
"""

from pathlib import Path
from typing import Optional, List

import pandas as pd
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from src.data.components.transforms import get_train_transforms, get_val_transforms


# Numeric tabular features available in both train and test metadata.
# Selected for clinical relevance to skin cancer detection:
# - Patient demographics: age, lesion size
# - Color features (LAB/LCH): lesion vs surrounding skin color differences
# - Geometry: area, perimeter, eccentricity, symmetry
# - DNN confidence: pre-computed lesion detection confidence
TABULAR_NUM_COLS = [
    'age_approx',
    'clin_size_long_diam_mm',
    'tbp_lv_A', 'tbp_lv_Aext',           # LAB A channel (lesion, surrounding)
    'tbp_lv_B', 'tbp_lv_Bext',           # LAB B channel
    'tbp_lv_C', 'tbp_lv_Cext',           # Chroma
    'tbp_lv_H', 'tbp_lv_Hext',           # Hue
    'tbp_lv_L', 'tbp_lv_Lext',           # Lightness
    'tbp_lv_areaMM2',                     # Lesion area in mm²
    'tbp_lv_area_perim_ratio',            # Compactness
    'tbp_lv_color_std_mean',              # Color variation within lesion
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',  # Color diff (lesion - surround)
    'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',              # Combined color diff
    'tbp_lv_eccentricity',                # Shape elongation
    'tbp_lv_minorAxisMM',                 # Minor axis length
    'tbp_lv_nevi_confidence',             # DNN nevi confidence
    'tbp_lv_norm_border',                 # Border regularity
    'tbp_lv_norm_color',                  # Color uniformity
    'tbp_lv_perimeterMM',                 # Perimeter
    'tbp_lv_radial_color_std_max',        # Max radial color variation
    'tbp_lv_stdL', 'tbp_lv_stdLExt',     # Lightness std (lesion, surrounding)
    'tbp_lv_symm_2axis',                  # 2-axis symmetry score
    'tbp_lv_symm_2axis_angle',            # Symmetry angle
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',  # 3D body position
    'tbp_lv_dnn_lesion_confidence',        # DNN lesion confidence
]

# Categorical columns to encode
TABULAR_CAT_COLS = ['sex', 'anatom_site_general']

# Total feature count = len(TABULAR_NUM_COLS) + encoded categoricals
# sex: 1 col (0/1), anatom_site_general: 7 one-hot cols
# See ISICDataModule.setup() for exact count after encoding


class ISICDataset(Dataset):
    """ISIC dataset that returns images + tabular features from HDF5.

    Each __getitem__ call returns a dict with:
        'image': Tensor [C, H, W]
        'tabular': Tensor [n_features] (standardized numeric + encoded categorical)
        'target': Tensor scalar (0/1)
        'isic_id': str
    """

    def __init__(
        self,
        hdf5_path: str,
        isic_ids: list,
        targets: list,
        tabular_features: np.ndarray,
        transform=None,
    ):
        self.hdf5_path = hdf5_path
        self.isic_ids = isic_ids
        self.targets = targets
        self.tabular_features = tabular_features  # [N, n_features], already standardized
        self.transform = transform
        self.hdf5 = None

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        if self.hdf5 is None:
            self.hdf5 = h5py.File(self.hdf5_path, 'r')

        isic_id = self.isic_ids[idx]
        target = self.targets[idx]

        # Load image from HDF5
        image_bytes = self.hdf5[isic_id][()]
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return {
            'image': image,
            'tabular': torch.tensor(self.tabular_features[idx], dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'isic_id': isic_id,
        }

    def __del__(self):
        if hasattr(self, 'hdf5') and self.hdf5 is not None:
            self.hdf5.close()


class ISICDataModule(L.LightningDataModule):
    """LightningDataModule for ISIC 2024 with tabular features.

    Handles:
    - HDF5 image loading
    - Stratified K-fold splitting by patient_id
    - Tabular feature extraction, encoding, and standardization
    - pos_weight computation for class imbalance
    """

    def __init__(
        self,
        data_dir: str = "data/isic-2024-challenge",
        hdf5_file: str = "train-image.hdf5",
        metadata_file: str = "train-metadata.csv",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        fold: int = 0,
        n_folds: int = 5,
        img_size: int = 224,
        data_fraction: float = 1.0,
    ):
        """Initialize ISIC DataModule.

        Args:
            data_dir: Path to data directory
            hdf5_file: Name of the HDF5 image file
            metadata_file: Name of the metadata CSV file
            batch_size: Batch size per GPU
            num_workers: Number of dataloader workers
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            fold: Which fold to use as validation
            n_folds: Total number of cross-validation folds
            img_size: Image size for transforms
            data_fraction: Fraction of data to use (for debugging)
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.hdf5_path = self.data_dir / hdf5_file
        self.metadata_path = self.data_dir / metadata_file

        self.data_train: Optional[ISICDataset] = None
        self.data_val: Optional[ISICDataset] = None

        # Computed during setup
        self.pos_weight: float = 1.0
        self.n_tabular_features: int = 0

        # Tabular normalization stats (computed from training set)
        self._tab_mean: Optional[np.ndarray] = None
        self._tab_std: Optional[np.ndarray] = None

    def prepare_data(self):
        """Check data files exist."""
        assert self.hdf5_path.exists(), f"HDF5 file not found: {self.hdf5_path}"
        assert self.metadata_path.exists(), f"Metadata not found: {self.metadata_path}"

    def _encode_tabular(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and encode tabular features from metadata DataFrame.

        Numeric features: filled with 0 for NaN, used as-is (standardization later).
        Categorical features: one-hot encoded with fixed categories.

        Args:
            df: DataFrame with metadata columns

        Returns:
            np.ndarray of shape [len(df), n_features]
        """
        parts = []

        # Numeric features
        # Use reindex to handle missing columns (e.g. tbp_lv_dnn_lesion_confidence in test set)
        num_data = df.reindex(columns=TABULAR_NUM_COLS, fill_value=0).fillna(0).values.astype(np.float32)
        parts.append(num_data)

        # Categorical: sex -> binary (male=1, female=0, unknown=0.5)
        sex_map = {'male': 1.0, 'female': 0.0}
        sex_vals = df['sex'].map(sex_map).fillna(0.5).values.astype(np.float32).reshape(-1, 1)
        parts.append(sex_vals)

        # Categorical: anatom_site_general -> one-hot with fixed categories
        site_categories = [
            'head/neck', 'upper extremity', 'lower extremity',
            'anterior torso', 'posterior torso', 'lateral torso', 'palms/soles'
        ]
        site_encoded = np.zeros((len(df), len(site_categories)), dtype=np.float32)
        for i, cat in enumerate(site_categories):
            site_encoded[:, i] = (df['anatom_site_general'] == cat).astype(np.float32)
        parts.append(site_encoded)

        return np.hstack(parts)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data, create folds, encode tabular features, standardize.

        This is called by Lightning with both `stage="fit"` (training)
        and `stage="test"` (testing). We split by fold, compute tabular
        statistics from the training set, and standardize both sets.
        """
        # Load metadata
        df = pd.read_csv(self.metadata_path, low_memory=False)

        # Apply data fraction sampling
        if self.hparams.data_fraction < 1.0:
            if 'target' in df.columns:
                pos = df[df['target'] == 1]
                neg = df[df['target'] == 0]
                pos = pos.sample(frac=self.hparams.data_fraction, random_state=42)
                neg = neg.sample(frac=self.hparams.data_fraction, random_state=42)
                df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
            else:
                df = df.sample(frac=self.hparams.data_fraction, random_state=42).reset_index(drop=True)

        # Create folds if not present
        if 'fold' not in df.columns:
            from sklearn.model_selection import StratifiedKFold

            patient_targets = df.groupby('patient_id')['target'].mean()
            patient_targets = (patient_targets > 0.5).astype(int)

            skf = StratifiedKFold(
                n_splits=self.hparams.n_folds,
                shuffle=True,
                random_state=42
            )

            df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(patient_targets.index, patient_targets)):
                val_patients = patient_targets.index[val_idx]
                df.loc[df['patient_id'].isin(val_patients), 'fold'] = fold

        # Split by fold
        train_df = df[df['fold'] != self.hparams.fold].reset_index(drop=True)
        val_df = df[df['fold'] == self.hparams.fold].reset_index(drop=True)

        # Compute class distribution and pos_weight for BCEWithLogitsLoss
        if 'target' in train_df.columns:
            n_pos = int(train_df['target'].sum())
            n_neg = len(train_df) - n_pos
            self.pos_weight = n_neg / max(n_pos, 1)
            print(f"  Train: {len(train_df)} samples ({n_pos} positive, {n_neg} negative)")
            print(f"  Val:   {len(val_df)} samples")
            print(f"  Class imbalance ratio: 1:{self.pos_weight:.1f} (pos_weight={self.pos_weight:.2f})")
        else:
            self.pos_weight = 1.0

        # Encode tabular features
        train_tabular = self._encode_tabular(train_df)
        val_tabular = self._encode_tabular(val_df)

        # Standardize using training set statistics (zero mean, unit variance)
        self._tab_mean = train_tabular.mean(axis=0)
        self._tab_std = train_tabular.std(axis=0)
        self._tab_std[self._tab_std < 1e-7] = 1.0  # Avoid division by zero

        train_tabular = (train_tabular - self._tab_mean) / self._tab_std
        val_tabular = (val_tabular - self._tab_mean) / self._tab_std

        self.n_tabular_features = train_tabular.shape[1]
        print(f"  Tabular features: {self.n_tabular_features} dims "
              f"({len(TABULAR_NUM_COLS)} numeric + {1 + 7} categorical)")

        # Create datasets
        if stage == "fit" or stage is None:
            self.data_train = ISICDataset(
                hdf5_path=str(self.hdf5_path),
                isic_ids=train_df['isic_id'].tolist(),
                targets=train_df['target'].tolist(),
                tabular_features=train_tabular,
                transform=get_train_transforms(self.hparams.img_size),
            )

            self.data_val = ISICDataset(
                hdf5_path=str(self.hdf5_path),
                isic_ids=val_df['isic_id'].tolist(),
                targets=val_df['target'].tolist(),
                tabular_features=val_tabular,
                transform=get_val_transforms(self.hparams.img_size),
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Create and return train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Create and return validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            shuffle=False,
        )


if __name__ == "__main__":
    """For testing the datamodule."""
    dm = ISICDataModule(data_dir="data/isic-2024-challenge", data_fraction=0.01)
    dm.prepare_data()
    dm.setup()

    print(f"Train dataset size: {len(dm.data_train)}")
    print(f"Val dataset size: {len(dm.data_val)}")
    print(f"Tabular features: {dm.n_tabular_features}")

    # Test one batch
    batch = next(iter(dm.train_dataloader()))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch tabular shape: {batch['tabular'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")