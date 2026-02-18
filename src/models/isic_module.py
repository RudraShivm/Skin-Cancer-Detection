"""ISIC 2024 LightningModule."""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import timm
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAUROC


class ISICLitModule(LightningModule):
    """LightningModule for ISIC skin cancer classification.
    
    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation Loop (validation_step)
        - Test Loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    """
    
    def __init__(
        self,
        name: str,
        backbone: str,
        num_classes: int = 1,
        pretrained: bool = True,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 20,
        dropout: float = 0.0,
    ):
        """Initialize ISIC LitModule.
        
        Args:
            backbone: TIMM model name
            num_classes: Number of output classes (1 for binary)
            pretrained: Use ImageNet pretrained weights
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            max_epochs: Total number of training epochs
            dropout: Dropout rate
        """
        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # Create model using TIMM
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        
        # Loss function - BCEWithLogitsLoss combines Sigmoid + BCE
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metric objects for tracking performance
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        
        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # For tracking best validation metric
        self.val_auroc_best = MaxMetric()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)
    
    def on_train_start(self) -> None:
        """Lightning hook called at the beginning of training."""
        # Reset metrics to ensure they start from scratch
        self.val_loss.reset()
        self.val_auroc.reset()
        self.val_auroc_best.reset()
    
    def model_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        
        Args:
            batch: A batch from the dataloader
        
        Returns:
            loss: The loss value
            preds: Model predictions (probabilities)
            targets: Ground truth labels
        """
        images = batch["image"]
        targets = batch["target"]
        
        logits = self.forward(images).squeeze(1)
        loss = self.criterion(logits, targets)
        preds = torch.sigmoid(logits)
        return loss, preds, targets
    
    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data.
        
        Args:
            batch: A batch from the dataloader
            batch_idx: The index of this batch
        
        Returns:
            loss: The loss value for this batch
        """
        loss, preds, targets = self.model_step(batch)
        
        # Update and log metrics
        self.train_loss(loss)
        self.train_auroc(preds, targets.long())
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data.
        
        Args:
            batch: A batch from the dataloader
            batch_idx: The index of this batch
        """
        loss, preds, targets = self.model_step(batch)
        
        # Update and log metrics
        self.val_loss(loss)
        self.val_auroc(preds, targets.long())
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """Lightning hook called at the end of validation epoch."""
        auroc = self.val_auroc.compute()  # get current val auroc
        self.val_auroc_best(auroc)  # update best so far val auroc
        
        # log best so far val auroc as a value through `.compute()` method
        self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Perform a single test step on a batch of data.
        
        Args:
            batch: A batch from the dataloader
            batch_idx: The index of this batch
        """
        loss, preds, targets = self.model_step(batch)
        
        # Update and log metrics
        self.test_loss(loss)
        self.test_auroc(preds, targets.long())
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use.
        
        Returns:
            A dict containing the configured optimizers and LR schedulers
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    """For testing the model."""
    model = ISICLitModule()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be [2, 1]