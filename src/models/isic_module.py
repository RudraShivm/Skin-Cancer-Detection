"""ISIC 2024 LightningModule with image + tabular feature fusion.

Architecture:
    Image backbone (TIMM, e.g. EfficientNet) → feature vector (e.g. 1280-dim)
                                                    ↓
                                            [concat with tabular features]
                                                    ↓
                                            Fusion MLP → 1 logit (binary)

The model extracts image features from a TIMM backbone (without its built-in
classifier head), concatenates them with the standardized tabular features
from the metadata, and passes the combined representation through a small
MLP fusion head for the final prediction.

Reference: Yakiniku 2nd place solution used image features + tabular features
together, feeding them into both neural networks and GBDTs.
"""

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

    Supports two modes:
    - Image-only (n_tabular_features=0): Uses TIMM backbone directly
    - Image + Tabular (n_tabular_features>0): Strips TIMM's head, adds fusion MLP
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
        n_tabular_features: int = 0,
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
            pos_weight: Weight for positive class in BCEWithLogitsLoss
            n_tabular_features: Number of tabular input features (0 = image-only)
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        if n_tabular_features > 0:
            # Image + Tabular mode: create backbone WITHOUT classifier head
            self.model = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,       # Strip the classifier head
                drop_rate=dropout,
            )
            # Detect the actual feature dimension with a dummy forward pass.
            # model.num_features can be unreliable for some architectures (e.g.
            # MobileNetV3 has an extra conv_head layer that adds a second projection),
            # so a dummy pass gives us the true output size.
            # Detect the actual feature dimension with a dummy forward pass.
            with torch.no_grad():
                # Use 336x336 as dummy size to support EVA02 Small and other
                # models that might have strict input size requirements.
                # Most models are flexible or expect 224/pristine 336/448.
                _dummy = torch.zeros(1, 3, 336, 336)
                img_feat_dim = self.model(_dummy).shape[1]

            # Fusion MLP: concatenated [image_features, tabular_features] → 1 logit
            fusion_dim = img_feat_dim + n_tabular_features
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
        else:
            # Image-only mode: use TIMM backbone with built-in classifier
            self.model = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            self.fusion_head = None

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # Metric objects
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.val_roc = BinaryROC()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_auroc_best = MaxMetric()

        # Buffers saved in checkpoint
        self.register_buffer("best_threshold", torch.tensor(0.5))
        self.register_buffer("best_auroc", torch.tensor(0.0))

        # Storage for validation predictions (for ROC curve logging)
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, tabular: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through backbone + optional fusion head.

        Args:
            x: Image tensor [B, C, H, W]
            tabular: Tabular features tensor [B, n_features] or None

        Returns:
            Logits tensor [B, num_classes]
        """
        if self.fusion_head is not None and tabular is not None:
            # Image + Tabular fusion
            img_features = self.model(x)                     # [B, img_feat_dim]
            combined = torch.cat([img_features, tabular], dim=1)  # [B, img_feat_dim + n_tab]
            return self.fusion_head(combined)                 # [B, 1]
        else:
            # Image-only fallback
            return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook called at the beginning of training."""
        self.val_loss.reset()
        self.val_auroc.reset()
        self.val_auroc_best.reset()

    def model_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        images = batch["image"]
        targets = batch["target"]
        tabular = batch.get("tabular", None)

        logits = self.forward(images, tabular).squeeze(1)
        loss = self.criterion(logits, targets)
        preds = torch.sigmoid(logits)
        return loss, preds, targets

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data."""
        loss, preds, targets = self.model_step(batch)

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
        """Perform a single validation step on a batch of data."""
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_auroc(preds, targets.long())
        self.val_roc.update(preds, targets.long())

        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu().long())

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def _compute_optimal_threshold(self, fpr: torch.Tensor, tpr: torch.Tensor, thresholds: torch.Tensor) -> float:
        """Find the optimal threshold where (TPR, FPR) is closest to (1, 0)."""
        distances = torch.sqrt((1 - tpr) ** 2 + fpr ** 2)
        best_idx = torch.argmin(distances)
        return thresholds[best_idx].item()

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called at the end of validation epoch."""
        # Compute ROC curve and find optimal threshold
        fpr, tpr, thresholds = self.val_roc.compute()
        optimal_threshold = self._compute_optimal_threshold(fpr, tpr, thresholds)
        self.best_threshold = torch.tensor(optimal_threshold, device=self.device)

        # Compute AUROC directly from ROC curve data (trapezoidal rule).
        # We don't use self.val_auroc.compute() here because Lightning manages
        # that metric's lifecycle (compute/reset) as part of self.log().
        auroc = torch.trapezoid(tpr, fpr).abs()
        self.val_auroc_best(auroc)

        best_auroc_val = self.val_auroc_best.compute()
        self.best_auroc = torch.tensor(best_auroc_val.item(), device=self.device)

        self.log("val/auroc_best", best_auroc_val, sync_dist=True, prog_bar=True)
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
        """Log ROC curve to WandB as a custom chart."""
        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        fpr_np = fpr.cpu().numpy()
        tpr_np = tpr.cpu().numpy()
        thresholds_np = thresholds.cpu().numpy()

        # Subsample if too many points
        max_points = 500
        if len(fpr_np) > max_points:
            indices = np.linspace(0, len(fpr_np) - 1, max_points, dtype=int)
            fpr_np = fpr_np[indices]
            tpr_np = tpr_np[indices]
            thresholds_np = thresholds_np[indices]

        data = [[f, t, th] for f, t, th in zip(fpr_np, tpr_np, thresholds_np)]

        table = wandb.Table(
            data=data,
            columns=["fpr", "tpr", "threshold"],
        )

        # Compute AUROC from ROC data for accurate title
        auroc_val = torch.trapezoid(tpr, fpr).abs().item()
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
        """Perform a single test step on a batch of data."""
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_auroc(preds, targets.long())

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a single prediction step."""
        images = batch["image"]
        tabular = batch.get("tabular", None)
        logits = self.forward(images, tabular).squeeze(1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= self.best_threshold).long()

        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "threshold": self.best_threshold,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure AdamW optimizer with CosineAnnealing LR scheduler.

        Uses CosineAnnealingLR to smoothly decay the learning rate from lr
        down to eta_min (1e-6) over max_epochs. This provides:
        - Fast initial learning at the original lr
        - Gradual fine-tuning as lr decreases
        - Better final convergence than constant lr
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
    # Test image-only mode
    model = ISICLitModule(name="test", backbone="tf_efficientnet_b0_ns")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Image-only output shape: {out.shape}")

    # Test image + tabular mode
    model_tab = ISICLitModule(
        name="test_tab", backbone="tf_efficientnet_b0_ns", n_tabular_features=42
    )
    tab = torch.randn(2, 42)
    out_tab = model_tab(x, tab)
    print(f"Image+Tabular output shape: {out_tab.shape}")
    print(f"Best threshold: {model_tab.best_threshold}")