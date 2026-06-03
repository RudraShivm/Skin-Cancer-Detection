from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import psutil
import rootutils
import time
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def partial_auc_above_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_tpr: float = 0.80,
) -> float:
    """Compute scaled pAUC over the high-sensitivity region TPR >= min_tpr."""
    max_fpr = 1.0 - min_tpr
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def build_val_dataloader(cfg: DictConfig):
    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    return datamodule.val_dataloader()


def run_onnx_batch(session, batch: Dict, image_name: str, tabular_name: str) -> np.ndarray:
    images = batch["image"].detach().cpu().numpy().astype(np.float32)
    tabular = batch["tabular"].detach().cpu().numpy().astype(np.float32)
    outputs = session.run(None, {image_name: images, tabular_name: tabular})[0]
    return sigmoid(outputs.reshape(-1))


def benchmark_latency(
    session,
    dataloader,
    image_name: str,
    tabular_name: str,
    warmup_batches: int,
    max_batches: int,
) -> Tuple[np.ndarray, float]:
    process = psutil.Process()
    latencies: List[float] = []

    iterator = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        run_onnx_batch(session, batch, image_name, tabular_name)

    mem_before = process.memory_info().rss / 1024 / 1024

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        batch_size = int(batch["image"].shape[0])
        start = time.perf_counter()
        run_onnx_batch(session, batch, image_name, tabular_name)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.extend([elapsed_ms / batch_size] * batch_size)

    mem_after = process.memory_info().rss / 1024 / 1024
    return np.array(latencies, dtype=np.float32), mem_after - mem_before


def evaluate_model(session, dataloader, image_name: str, tabular_name: str) -> Tuple[np.ndarray, np.ndarray]:
    probs: List[float] = []
    targets: List[float] = []

    for batch in dataloader:
        probs.extend(run_onnx_batch(session, batch, image_name, tabular_name).tolist())
        targets.extend(batch["target"].detach().cpu().numpy().reshape(-1).tolist())

    return np.array(probs, dtype=np.float32), np.array(targets, dtype=np.int32)


def summarize_model(
    model_name: str,
    model_path: Path,
    dataloader,
    cfg: DictConfig,
) -> Dict[str, float]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=list(cfg.runtime.providers))
    image_name = cfg.inputs.image
    tabular_name = cfg.inputs.tabular

    latencies, mem_delta = benchmark_latency(
        session,
        dataloader,
        image_name=image_name,
        tabular_name=tabular_name,
        warmup_batches=cfg.benchmark.warmup_batches,
        max_batches=cfg.benchmark.max_latency_batches,
    )
    probs, targets = evaluate_model(session, dataloader, image_name, tabular_name)

    size_mb = model_path.stat().st_size / 1024 / 1024
    auroc = float(roc_auc_score(targets, probs))
    pauc = partial_auc_above_tpr(targets, probs, cfg.metrics.min_tpr)

    return {
        "name": model_name,
        "size_mb": size_mb,
        "mean_ms": float(latencies.mean()),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "memory_delta_mb": float(mem_delta),
        "auroc": auroc,
        "pauc": pauc,
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="onnx_benchmark.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    fp32_path = Path(cfg.onnx.fp32)
    int8_path = Path(cfg.onnx.int8)

    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_path}")
    if not int8_path.exists():
        raise FileNotFoundError(f"INT8 ONNX model not found: {int8_path}")

    val_dataloader = build_val_dataloader(cfg)

    print("\nBenchmarking FP32 model on held-out validation fold...")
    fp32 = summarize_model("FP32", fp32_path, val_dataloader, cfg)

    val_dataloader = build_val_dataloader(cfg)
    print("\nBenchmarking INT8 model on held-out validation fold...")
    int8 = summarize_model("INT8", int8_path, val_dataloader, cfg)

    print("\nONNX Benchmark Summary")
    print("======================")
    print(f"Validation fold: {cfg.data.fold}")
    print(f"Image size:      {cfg.data.img_size}")
    print(f"Batch size:      {cfg.data.batch_size}")
    print()
    print("| Model | Size MB | Mean ms/img | P95 ms/img | P99 ms/img | AUROC | pAUC |")
    print("|-------|---------|-------------|------------|------------|-------|------|")
    for result in [fp32, int8]:
        print(
            f"| {result['name']} "
            f"| {result['size_mb']:.2f} "
            f"| {result['mean_ms']:.2f} "
            f"| {result['p95_ms']:.2f} "
            f"| {result['p99_ms']:.2f} "
            f"| {result['auroc']:.5f} "
            f"| {result['pauc']:.5f} |"
        )

    print()
    print(f"Speedup:        {fp32['mean_ms'] / int8['mean_ms']:.2f}x")
    print(f"Size reduction: {fp32['size_mb'] / int8['size_mb']:.2f}x")
    print(f"AUROC delta:    {fp32['auroc'] - int8['auroc']:.5f}")
    print(f"pAUC delta:     {fp32['pauc'] - int8['pauc']:.5f}")


if __name__ == "__main__":
    main()
