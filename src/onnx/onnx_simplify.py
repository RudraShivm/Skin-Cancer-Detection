import onnx
from pathlib import Path

output_dir = Path("onnx_models")
model_onnx = onnx.load(output_dir / "skin_cancer_mobilenetv3_fp32.onnx")
onnx.checker.check_model(model_onnx)
print("ONNX model is valid.")

from onnxsim import simplify
model_simplified, check = simplify(model_onnx)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simplified, output_dir / "skin_cancer_mobilenetv3_fp32_simplified.onnx")
