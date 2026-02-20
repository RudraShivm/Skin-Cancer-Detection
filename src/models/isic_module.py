"""ISIC 2024 LightningModule."""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import timm
import numpy as np
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryROC


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
        pos_weight: float = 1.0,
    ):
        """Initialize ISIC LitModule.
        
        Args:
            name: Model name identifier
            backbone: TIMM model name
            num_classes: Number of output classes (1 for binary)
            pretrained: Use ImageNet pretrained weights
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            max_epochs: Total number of training epochs
            dropout: Dropout rate
            pos_weight: Weight for positive class in BCEWithLogitsLoss.
                        Set to num_negative / num_positive to handle class imbalance.
                        Default 1.0 = no weighting.
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
        # pos_weight handles class imbalance: each positive sample's loss
        # is multiplied by this factor. For ISIC 2024 (~0.5% malignant),
        # pos_weight ≈ 199 makes the model take malignant cases seriously
        # instead of predicting near-zero probability for everything.
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        
        # Metric objects for tracking performance
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        
        # ROC curve metric for threshold calculation
        self.val_roc = BinaryROC()
        
        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        # For tracking best validation metric
        self.val_auroc_best = MaxMetric()
        
        # Best threshold found from ROC curve analysis
        # Stored as a buffer so it's saved in the checkpoint
        self.register_buffer("best_threshold", torch.tensor(0.5))
        
        # Best AUROC achieved (used as weight in ensemble)
        self.register_buffer("best_auroc", torch.tensor(0.0))
        
        # Storage for validation predictions (for ROC curve logging)
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
    
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
    
    def on_validation_epoch_start(self) -> None:
        """Reset val prediction storage at the start of each validation epoch."""
        self._val_preds = []
        self._val_targets = []
    
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
        self.val_roc.update(preds, targets.long())
        
        # Store predictions for ROC curve logging
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu().long())
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
    
    def _compute_optimal_threshold(self, fpr: torch.Tensor, tpr: torch.Tensor, thresholds: torch.Tensor) -> float:
        """Find the optimal threshold where (TPR, FPR) is closest to (1, 0).
        
        This minimizes the geometric distance: sqrt((1-TPR)^2 + FPR^2)
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Corresponding thresholds
            
        Returns:
            Optimal threshold value
        """
        # Distance from the ideal point (0, 1) in ROC space
        distances = torch.sqrt((1 - tpr) ** 2 + fpr ** 2)
        best_idx = torch.argmin(distances)
        return thresholds[best_idx].item()
    
    def on_validation_epoch_end(self) -> None:
        """Lightning hook called at the end of validation epoch."""
        auroc = self.val_auroc.compute()  # get current val auroc
        self.val_auroc_best(auroc)  # update best so far val auroc
        
        # Update best_auroc buffer (saved in checkpoint)
        best_auroc_val = self.val_auroc_best.compute()
        self.best_auroc = torch.tensor(best_auroc_val.item(), device=self.device)
        
        # Compute ROC curve and find optimal threshold
        fpr, tpr, thresholds = self.val_roc.compute()
        optimal_threshold = self._compute_optimal_threshold(fpr, tpr, thresholds)
        self.best_threshold = torch.tensor(optimal_threshold, device=self.device)
        
        # Log metrics
        self.log("val/auroc_best", self.val_auroc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/best_threshold", optimal_threshold, sync_dist=True, prog_bar=True)
        
        # Log ROC curve to WandB if available
        self._log_roc_curve(fpr, tpr, thresholds, optimal_threshold)
        
        # Reset ROC metric for next epoch
        self.val_roc.reset()
    
    def _log_roc_curve(
        self,
        fpr: torch.Tensor,
        tpr: torch.Tensor,
        thresholds: torch.Tensor,
        optimal_threshold: float,
    ) -> None:
        """Log ROC curve to WandB as a custom chart.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            thresholds: Threshold values
            optimal_threshold: The optimal threshold found
        """
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return
        
        # Convert to numpy for wandb
        fpr_np = fpr.cpu().numpy()
        tpr_np = tpr.cpu().numpy()
        thresholds_np = thresholds.cpu().numpy()
        
        # Subsample if too many points (for performance)
        max_points = 500
        if len(fpr_np) > max_points:
            indices = np.linspace(0, len(fpr_np) - 1, max_points, dtype=int)
            fpr_np = fpr_np[indices]
            tpr_np = tpr_np[indices]
            thresholds_np = thresholds_np[indices]
        
        # Create WandB table for ROC curve
        data = []
        for f, t, th in zip(fpr_np, tpr_np, thresholds_np):
            data.append([f, t, th])
        
        table = wandb.Table(
            data=data,
            columns=["fpr", "tpr", "threshold"],
        )
        
        # Log ROC curve as a custom line plot
        auroc_val = self.val_auroc.compute().item()
        epoch = self.current_epoch
        wandb.log({
            f"roc_curve/epoch_{epoch:03d}": wandb.plot.line(
                table,
                "fpr",
                "tpr",
                title=f"ROC Curve | Epoch {epoch:03d} | AUROC={auroc_val:.4f} | threshold={optimal_threshold:.4f}",
            ),
        })
    
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
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single prediction step.
        
        Returns dict with probabilities and binary predictions using the optimal threshold.
        
        Args:
            batch: A batch from the dataloader
            batch_idx: The index of this batch
            
        Returns:
            Dict with 'probabilities', 'predictions', and 'threshold'
        """
        images = batch["image"]
        logits = self.forward(images).squeeze(1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.best_threshold).long()
        
        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "threshold": self.best_threshold,
        }
    
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
    model = ISICLitModule(name="test", backbone="tf_efficientnet_b0_ns")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be [2, 1]
    print(f"Best threshold: {model.best_threshold}")