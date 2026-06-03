from pathlib import Path
from typing import Dict, Iterator, List

import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class SkinCancerCalibrationReader:
    """Feed image + tabular calibration batches to ONNX Runtime quantization."""

    def __init__(
        self,
        dataloader,
        image_input_name: str,
        tabular_input_name: str,
        n_samples: int,
    ):
        self.data: List[Dict[str, np.ndarray]] = []
        self.iterator: Iterator[Dict[str, np.ndarray]]

        collected = 0
        for batch in dataloader:
            images = batch["image"].detach().cpu().numpy().astype(np.float32)
            tabular = batch["tabular"].detach().cpu().numpy().astype(np.float32)

            remaining = n_samples - collected
            if remaining <= 0:
                break

            images = images[:remaining]
            tabular = tabular[:remaining]
            if len(images) == 0:
                break

            self.data.append(
                {
                    image_input_name: images,
                    tabular_input_name: tabular,
                }
            )
            collected += len(images)

            if collected >= n_samples:
                break

        if not self.data:
            raise RuntimeError("Calibration dataloader did not produce any samples.")

        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)


def build_val_dataloader(cfg: DictConfig):
    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    return datamodule.val_dataloader()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="onnx_quant.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quant_pre_process,
            quantize_static,
        )
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for INT8 quantization. "
            "Install it with: pip install onnxruntime"
        ) from exc

    input_path = Path(cfg.onnx.input)
    preprocessed_path = Path(cfg.onnx.preprocessed)
    output_path = Path(cfg.onnx.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")

    val_dataloader = build_val_dataloader(cfg)
    cal_reader = SkinCancerCalibrationReader(
        val_dataloader,
        image_input_name=cfg.calibration.image_input_name,
        tabular_input_name=cfg.calibration.tabular_input_name,
        n_samples=cfg.calibration.n_samples,
    )

    print(f"Preprocessing ONNX model: {input_path} -> {preprocessed_path}")
    quant_pre_process(str(input_path), str(preprocessed_path))

    print(f"Quantizing ONNX model: {preprocessed_path} -> {output_path}")
    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=cal_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=cfg.quantization.per_channel,
        weight_type=QuantType[cfg.quantization.weight_type],
        activation_type=QuantType[cfg.quantization.activation_type],
    )
    print(f"INT8 quantization complete: {output_path}")


if __name__ == "__main__":
    main()
