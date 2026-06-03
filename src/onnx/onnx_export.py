import torch
import torch.onnx
import torch.nn as nn
from pathlib import Path
from src.models.isic_module import ISICLitModule
# Load one edge candidate checkpoint.
# Use the demo's standalone ISICLitModule or the training ISICLitModule,
# but keep the same tabular feature layout.
ckpt_path = "checkpoints/mobilenet_v3/fold_2/epoch_009_auroc_0.9796_seed_2105170.ckpt"
model = ISICLitModule.load_from_checkpoint(ckpt_path, map_location="cpu")
model.eval()

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, tabular):
        return self.model(image, tabular)

export_model = ExportWrapper(model).eval()

# This project is multimodal: image + tabular vector.
# Image size may be backbone-dependent. MobileNetV3 experiments use 256.
dummy_image = torch.randn(1, 3, 256, 256)
dummy_tabular = torch.randn(1, 43)

# Export
output_dir = Path("onnx_models")
output_dir.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    export_model,
    (dummy_image, dummy_tabular),
    str(output_dir / "skin_cancer_mobilenetv3_fp32.onnx"),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image", "tabular"],
    output_names=["logit"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "tabular": {0: "batch_size"},
        "logit": {0: "batch_size"},
    }
)
print("Export complete.")
