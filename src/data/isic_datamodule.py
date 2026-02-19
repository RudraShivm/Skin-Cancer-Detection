"""ISIC 2024 DataModule."""

from pathlib import Path
from typing import Optional

import pandas as pd
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

# Import transforms (we'll create this next)
from src.data.components.transforms import get_train_transforms, get_val_transforms


class ISICDataset(Dataset):
    """ISIC dataset using HDF5."""
    
    def __init__(
        self,
        hdf5_path: str,
        isic_ids: list,
        targets: list,
        transform=None,
    ):
        self.hdf5_path = hdf5_path
        self.isic_ids = isic_ids
        self.targets = targets
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
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        
        return {
            'image': image,
            'target': torch.tensor(target, dtype=torch.float32),
            'isic_id': isic_id,
        }
    
    def __del__(self):
        if hasattr(self, 'hdf5') and self.hdf5 is not None:
            self.hdf5.close()


class ISICDataModule(L.LightningDataModule):
    """LightningDataModule for ISIC 2024.
    
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # download, split, etc (only on 1 GPU/TPU in distributed)
        def setup(self, stage):
            # load data, set variables, etc (called on every GPU/TPU)
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # clean up after fit or test
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
            hdf5_file: Name of HDF5 file
            metadata_file: Name of metadata CSV
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            persistent_workers: Keep workers alive between epochs
            fold: Which fold to use for validation (0-4)
            n_folds: Total number of folds
            img_size: Image size for resizing
            data_fraction: Fraction of dataset to use (0.0 to 1.0)
        """
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.data_dir = Path(data_dir)
        self.hdf5_path = self.data_dir / hdf5_file
        self.metadata_path = self.data_dir / metadata_file
        
        # Data holders
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def prepare_data(self) -> None:
        """Download data if needed (only called on 1 GPU/TPU in distributed)."""
        # Check if data exists
        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"HDF5 file not found at {self.hdf5_path}. "
                "Please download ISIC 2024 dataset from Kaggle."
            )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: self.data_train, self.data_val, self.data_test.
        
        This method is called by Lightning with both `trainer.fit()` and `trainer.test()`,
        so be careful not to execute things like random split twice!
        """
        # Load metadata
        df = pd.read_csv(self.metadata_path, low_memory=False)
        
        # Apply data fraction sampling
        if self.hparams.data_fraction < 1.0:
            # Sample fraction but keep class balance
            # Use 'target' if present, otherwise random sample
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
            
            # Group by patient for stratification
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
        
        # Create datasets
        if stage == "fit" or stage is None:
            self.data_train = ISICDataset(
                hdf5_path=str(self.hdf5_path),
                isic_ids=train_df['isic_id'].tolist(),
                targets=train_df['target'].tolist(),
                transform=get_train_transforms(self.hparams.img_size),
            )
            
            self.data_val = ISICDataset(
                hdf5_path=str(self.hdf5_path),
                isic_ids=val_df['isic_id'].tolist(),
                targets=val_df['target'].tolist(),
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
    
    # Test one batch
    batch = next(iter(dm.train_dataloader()))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch target shape: {batch['target'].shape}")