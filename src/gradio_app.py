"""Gradio prediction UI for ISIC 2024 Skin Cancer Detection.

A web-based demo that lets you:
- Upload a skin lesion image
- Select which model checkpoint to use
- Optionally provide patient metadata (age, sex, body site)
- Get a malignant/benign prediction with confidence score

Run locally:
    python src/gradio_app.py --checkpoint-dir checkpoints/

Run on Colab:
    See GRADIO_DEMO.md for instructions.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import gradio as gr

# ---------------------------------------------------------------------------
# Try lightning, fall back to pytorch_lightning (for Colab/Kaggle)
# ---------------------------------------------------------------------------
try:
    from lightning import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule

import timm
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryROC
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ===================================================================
#  INLINE DEFINITIONS (self-contained — no repo imports needed)
# ===================================================================

def get_val_transforms(img_size: int = 256):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


TABULAR_NUM_COLS = [
    'age_approx', 'clin_size_long_diam_mm',
    'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext',
    'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
    'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL',
    'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity',
    'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence',
    'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
    'tbp_lv_dnn_lesion_confidence',
]

N_NUM = len(TABULAR_NUM_COLS)  # 35
N_TAB_TOTAL = N_NUM + 1 + 7    # +1 sex binary, +7 site one-hot = 43


def encode_simple_tabular(age: float, sex: str, site: str) -> np.ndarray:
    """Encode the 3 user-provided features into the full 43-dim vector.

    Features the user doesn't provide are set to 0 (the mean after
    standardization), so the model treats them as "average".
    """
    vec = np.zeros(N_TAB_TOTAL, dtype=np.float32)

    # age_approx is the first numeric feature
    vec[0] = age

    # sex → binary at position N_NUM
    sex_val = {'male': 1.0, 'female': 0.0}.get(sex.lower(), 0.5)
    vec[N_NUM] = sex_val

    # anatom_site_general → one-hot at positions N_NUM+1 to N_NUM+7
    site_categories = [
        'head/neck', 'upper extremity', 'lower extremity',
        'anterior torso', 'posterior torso', 'lateral torso', 'palms/soles'
    ]
    for i, cat in enumerate(site_categories):
        if site.lower() == cat.lower():
            vec[N_NUM + 1 + i] = 1.0
            break

    return vec


class ISICLitModule(LightningModule):
    """Matches the training checkpoint structure."""

    def __init__(
        self,
        name: str = "",
        backbone: str = "tf_efficientnet_b0_ns",
        num_classes: int = 1,
        pretrained: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        max_epochs: int = 20,
        dropout: float = 0.0,
        pos_weight: float = 1.0,
        n_tabular_features: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        if n_tabular_features > 0:
            self.model = timm.create_model(
                backbone, pretrained=False, num_classes=0, drop_rate=dropout,
            )
            img_feat_dim = self.model.num_features
            self.fusion_head = nn.Sequential(
                nn.Linear(img_feat_dim + n_tabular_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
        else:
            self.model = timm.create_model(
                backbone, pretrained=False, num_classes=num_classes, drop_rate=dropout,
            )
            self.fusion_head = None

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.val_roc = BinaryROC()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_auroc_best = MaxMetric()
        self.register_buffer("best_threshold", torch.tensor(0.5))
        self.register_buffer("best_auroc", torch.tensor(0.0))
        self._val_preds = []
        self._val_targets = []

    def forward(self, x, tabular=None):
        if self.fusion_head is not None and tabular is not None:
            img_features = self.model(x)
            combined = torch.cat([img_features, tabular], dim=1)
            return self.fusion_head(combined)
        else:
            return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def get_model_img_size(model):
    try:
        data_config = timm.data.resolve_data_config(model.model.pretrained_cfg)
        return data_config.get("input_size", (3, 224, 224))[-1]
    except Exception:
        return 256


# ===================================================================
#  CHECKPOINT DISCOVERY
# ===================================================================

def discover_checkpoints(checkpoint_dir: str) -> Dict[str, List[str]]:
    """Find all available model checkpoints.

    Returns: {display_name: ckpt_path, ...}
    """
    checkpoints = {}
    if not os.path.isdir(checkpoint_dir):
        return checkpoints

    for model_name in sorted(os.listdir(checkpoint_dir)):
        model_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        for fold_dir in sorted(os.listdir(model_dir)):
            fold_path = os.path.join(model_dir, fold_dir)
            if not os.path.isdir(fold_path) or not fold_dir.startswith("fold_"):
                continue

            # Find best checkpoint
            auroc_ckpts = sorted(glob.glob(os.path.join(fold_path, "epoch_*_auroc_*.ckpt")))
            if auroc_ckpts:
                def get_auroc(p):
                    try:
                        return float(os.path.basename(p).split("auroc_")[-1].replace(".ckpt", ""))
                    except:
                        return 0.0
                ckpt = max(auroc_ckpts, key=get_auroc)
            elif os.path.exists(os.path.join(fold_path, "last.ckpt")):
                ckpt = os.path.join(fold_path, "last.ckpt")
            else:
                any_ckpts = glob.glob(os.path.join(fold_path, "*.ckpt"))
                ckpt = any_ckpts[0] if any_ckpts else None

            if ckpt:
                display = f"{model_name} / {fold_dir} / {os.path.basename(ckpt)}"
                checkpoints[display] = ckpt

    return checkpoints


# ===================================================================
#  PREDICTION FUNCTION
# ===================================================================

def predict(
    image: np.ndarray,
    selected_checkpoints: List[str],
    age: float,
    sex: str,
    body_site: str,
    checkpoint_map: Dict[str, str],
    device: torch.device,
) -> Tuple[str, str, str]:
    """Run prediction on a single image.

    Returns: (result_label, confidence_text, details_text)
    """
    if image is None:
        return "⚠️ No image", "", "Please upload an image"

    if not selected_checkpoints:
        return "⚠️ No model", "", "Please select at least one checkpoint"

    # RGB image from Gradio
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    all_probs = []
    all_thresholds = []
    details = []

    for ckpt_name in selected_checkpoints:
        ckpt_path = checkpoint_map.get(ckpt_name)
        if not ckpt_path or not os.path.exists(ckpt_path):
            continue

        # Load model
        model = ISICLitModule.load_from_checkpoint(
            ckpt_path, map_location=device, strict=False
        )
        model.eval()
        model.to(device)

        img_size = get_model_img_size(model)
        uses_tabular = model.fusion_head is not None
        threshold = model.best_threshold.item()

        # Preprocess image
        transform = get_val_transforms(img_size)
        img_tensor = transform(image=image)["image"].unsqueeze(0).to(device)

        # Prepare tabular
        tab_tensor = None
        if uses_tabular:
            tab_vec = encode_simple_tabular(age, sex, body_site)
            tab_tensor = torch.tensor(tab_vec, dtype=torch.float32).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(img_tensor, tab_tensor).squeeze()
            prob = torch.sigmoid(logits).item()

        all_probs.append(prob)
        all_thresholds.append(threshold)
        details.append(f"  {ckpt_name}\n    prob={prob:.4f}, threshold={threshold:.4f}, tabular={uses_tabular}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_probs:
        return "⚠️ Error", "", "No valid checkpoints loaded"

    # Ensemble: average probabilities and thresholds
    avg_prob = np.mean(all_probs)
    avg_threshold = np.mean(all_thresholds)
    is_malignant = avg_prob >= avg_threshold

    # Format output
    if is_malignant:
        label = "🔴 MALIGNANT (Suspicious)"
        color_emoji = "🔴"
    else:
        label = "🟢 BENIGN (Normal)"
        color_emoji = "🟢"

    confidence = f"{color_emoji} Probability: {avg_prob:.4f}  |  Threshold: {avg_threshold:.4f}"

    detail_text = (
        f"Ensemble of {len(all_probs)} checkpoint(s):\n"
        + "\n".join(details)
        + f"\n\nAverage probability: {avg_prob:.4f}"
        + f"\nAverage threshold: {avg_threshold:.4f}"
        + f"\nPatient info: age={age}, sex={sex}, site={body_site}"
    )

    return label, confidence, detail_text


# ===================================================================
#  GRADIO UI
# ===================================================================

def build_app(checkpoint_dir: str, device: torch.device) -> gr.Blocks:
    """Build the Gradio web app."""

    checkpoint_map = discover_checkpoints(checkpoint_dir)
    ckpt_names = list(checkpoint_map.keys())

    if not ckpt_names:
        print(f"⚠️  No checkpoints found in {checkpoint_dir}")
        print("   Expected structure: {checkpoint_dir}/{model_name}/fold_N/*.ckpt")

    def on_predict(image, selected_ckpts, age, sex, site):
        return predict(image, selected_ckpts, age, sex, site, checkpoint_map, device)

    with gr.Blocks(
        title="Skin Cancer Detection",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="orange",
        ),
        css="""
        .result-malignant { color: #dc2626; font-size: 1.5em; font-weight: bold; }
        .result-benign { color: #16a34a; font-size: 1.5em; font-weight: bold; }
        .header { text-align: center; margin-bottom: 1em; }
        """,
    ) as app:
        gr.Markdown(
            """
            # 🔬 Skin Cancer Detection (ISIC 2024)
            Upload a skin lesion image, select model checkpoint(s), and get a prediction.
            Optionally provide patient metadata for improved accuracy (if model supports tabular features).
            """,
        )

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Skin Lesion Image",
                    type="numpy",
                    height=256,
                )

                checkpoint_input = gr.Dropdown(
                    choices=ckpt_names,
                    value=ckpt_names[:1] if ckpt_names else [],
                    multiselect=True,
                    label="Model Checkpoint(s)",
                    info="Select one or more. Multiple = ensemble average.",
                )

                gr.Markdown("### Patient Metadata (optional)")

                with gr.Row():
                    age_input = gr.Slider(
                        minimum=5, maximum=90, value=50, step=5,
                        label="Age (approx)",
                    )
                    sex_input = gr.Radio(
                        choices=["Male", "Female"],
                        value="Male",
                        label="Sex",
                    )

                site_input = gr.Dropdown(
                    choices=[
                        "head/neck", "upper extremity", "lower extremity",
                        "anterior torso", "posterior torso",
                        "lateral torso", "palms/soles",
                    ],
                    value="anterior torso",
                    label="Anatomical Site",
                )

                predict_btn = gr.Button(
                    "🔍 Analyze",
                    variant="primary",
                    size="lg",
                )

            # Right column: results
            with gr.Column(scale=1):
                result_label = gr.Textbox(
                    label="Prediction",
                    interactive=False,
                    lines=1,
                )
                confidence_text = gr.Textbox(
                    label="Confidence",
                    interactive=False,
                    lines=1,
                )
                details_text = gr.Textbox(
                    label="Details",
                    interactive=False,
                    lines=10,
                )

        predict_btn.click(
            fn=on_predict,
            inputs=[image_input, checkpoint_input, age_input, sex_input, site_input],
            outputs=[result_label, confidence_text, details_text],
        )

        gr.Markdown(
            """
            ---
            **Disclaimer**: This is a research tool, not a medical device.
            Always consult a dermatologist for clinical evaluation.

            Built with [Gradio](https://gradio.app) • Model: TIMM + Tabular Fusion • Data: ISIC 2024
            """,
        )

    return app


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Gradio skin cancer detection demo")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio link (for Colab demos)",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run on",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    app = build_app(args.checkpoint_dir, device)
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
