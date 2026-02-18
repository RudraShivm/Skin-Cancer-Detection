"""Ensemble prediction script for multi-model, multi-fold inference.

Supports two ensemble strategies:
  - soft (default): Average probabilities, then apply averaged threshold
  - hard_weighted: Apply per-ckpt threshold, then weighted vote by AUROC

Image size is auto-detected from each model's TIMM backbone config.

Usage:
    # Auto img_size, soft voting (default)
    python src/ensemble_predict.py \
        --models efficientnet_b0 convnext_large \
        --checkpoint-dir checkpoints/ \
        --image-paths img1.jpg img2.jpg

    # Hard weighted voting
    python src/ensemble_predict.py \
        --models efficientnet_b0 convnext_large \
        --checkpoint-dir checkpoints/ \
        --image-dir test_images/ \
        --strategy hard_weighted
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.isic_module import ISICLitModule
from src.data.components.transforms import get_val_transforms


def get_model_img_size(model: ISICLitModule) -> int:
    """Auto-detect the optimal input image size from the TIMM backbone.

    Uses TIMM's data config to resolve the correct input size for the backbone.
    Falls back to 224 if detection fails.

    Args:
        model: Loaded ISICLitModule

    Returns:
        Image size (single int, assumes square)
    """
    try:
        import timm
        data_config = timm.data.resolve_data_config(model.model.pretrained_cfg)
        input_size = data_config.get("input_size", (3, 224, 224))
        # input_size is typically (C, H, W)
        return input_size[-1]  # Use the last dim (width = height for square)
    except Exception:
        return 224


def find_best_checkpoint(model_dir: str, fold: int) -> Optional[str]:
    """Find the best checkpoint for a given model and fold.

    Searches for checkpoints in {model_dir}/fold_{fold}/ directory.
    Prefers checkpoint with highest AUROC in filename, falls back to last.ckpt.

    Args:
        model_dir: Path to model checkpoint directory
        fold: Fold number

    Returns:
        Path to best checkpoint, or None if not found
    """
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    if not os.path.isdir(fold_dir):
        return None

    # Look for AUROC-named checkpoints (e.g., epoch_015_auroc_0.9234.ckpt)
    ckpt_files = glob.glob(os.path.join(fold_dir, "epoch_*_auroc_*.ckpt"))

    if ckpt_files:
        # Sort by AUROC value extracted from filename (highest first)
        def extract_auroc(path: str) -> float:
            try:
                name = os.path.basename(path)
                auroc_str = name.split("auroc_")[-1].replace(".ckpt", "")
                return float(auroc_str)
            except (ValueError, IndexError):
                return 0.0

        ckpt_files.sort(key=extract_auroc, reverse=True)
        return ckpt_files[0]

    # Fallback: look for last.ckpt
    last_ckpt = os.path.join(fold_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    # Fallback: any .ckpt file
    any_ckpts = glob.glob(os.path.join(fold_dir, "*.ckpt"))
    if any_ckpts:
        return any_ckpts[0]

    return None


def discover_folds(model_dir: str) -> List[int]:
    """Discover available fold directories for a model.

    Args:
        model_dir: Path to model checkpoint directory

    Returns:
        List of available fold numbers
    """
    folds = []
    if not os.path.isdir(model_dir):
        return folds

    for entry in os.listdir(model_dir):
        if entry.startswith("fold_") and os.path.isdir(os.path.join(model_dir, entry)):
            try:
                fold_num = int(entry.split("_")[1])
                folds.append(fold_num)
            except (ValueError, IndexError):
                continue

    return sorted(folds)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ISICLitModule:
    """Load a trained model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded ISICLitModule
    """
    model = ISICLitModule.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def load_and_preprocess_image(
    image_path: str,
    img_size: int = 224,
) -> torch.Tensor:
    """Load and preprocess a single image.

    Args:
        image_path: Path to image file
        img_size: Image size for resizing

    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_val_transforms(img_size)
    image = transform(image=image)["image"]

    return image.unsqueeze(0)  # Add batch dimension


def predict_single_model(
    model: ISICLitModule,
    image_paths: List[str],
    img_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, float, float]:
    """Run prediction with a single model on all images.

    Handles preprocessing with the model's correct img_size.

    Args:
        model: Trained ISICLitModule
        image_paths: List of image paths
        img_size: Image size for this model
        device: Device for inference

    Returns:
        Tuple of (probabilities array, optimal threshold, best auroc)
    """
    # Preprocess images with model-specific img_size
    image_tensors = []
    for path in image_paths:
        tensor = load_and_preprocess_image(path, img_size)
        image_tensors.append(tensor)

    images = torch.cat(image_tensors, dim=0).to(device)

    with torch.no_grad():
        logits = model(images).squeeze(1)
        probabilities = torch.sigmoid(logits)

    threshold = model.best_threshold.item()
    best_auroc = model.best_auroc.item() if hasattr(model, "best_auroc") else 0.5

    return probabilities.cpu().numpy(), threshold, best_auroc


def ensemble_predict(
    models: List[str],
    checkpoint_dir: str,
    image_paths: List[str],
    img_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    strategy: str = "soft",
) -> Dict[str, np.ndarray]:
    """Run ensemble prediction across multiple models and folds.

    Args:
        models: List of model names
        checkpoint_dir: Base directory containing model checkpoint subdirs
        image_paths: List of image file paths
        img_size: Image size override (None = auto-detect per model)
        device: Device for inference
        strategy: Ensemble strategy - "soft" or "hard_weighted"
            - "soft": Average probabilities, apply averaged threshold
            - "hard_weighted": Apply per-ckpt threshold, weighted vote by AUROC

    Returns:
        Dict with ensemble results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate image paths
    valid_paths = []
    for path in image_paths:
        if os.path.exists(path):
            valid_paths.append(path)
        else:
            print(f"  Warning: Image not found: {path}")

    if not valid_paths:
        raise ValueError("No valid images found!")

    print(f"Processing {len(valid_paths)} images...")
    print(f"Ensemble strategy: {strategy}")

    # Collect predictions from all models and folds
    all_probabilities = []     # List of prob arrays, one per fold
    all_thresholds = []        # Threshold per fold
    all_aurocs = []            # AUROC per fold (for weighting)
    per_model_results = {}

    for model_name in models:
        model_dir = os.path.join(checkpoint_dir, model_name)
        folds = discover_folds(model_dir)

        if not folds:
            print(f"  Warning: No folds found for model '{model_name}' in {model_dir}")
            continue

        print(f"\nModel: {model_name} ({len(folds)} folds: {folds})")
        model_probs = []
        model_thresholds = []
        model_aurocs = []

        for fold in folds:
            ckpt_path = find_best_checkpoint(model_dir, fold)
            if ckpt_path is None:
                print(f"  Fold {fold}: No checkpoint found, skipping")
                continue

            print(f"  Fold {fold}: Loading {os.path.basename(ckpt_path)}")
            model = load_model_from_checkpoint(ckpt_path, device)

            # Auto-detect image size from the model if not overridden
            fold_img_size = img_size if img_size is not None else get_model_img_size(model)
            if fold == folds[0]:
                print(f"    Image size: {fold_img_size} {'(auto-detected)' if img_size is None else '(override)'}")

            probs, threshold, auroc = predict_single_model(
                model, valid_paths, fold_img_size, device
            )
            model_probs.append(probs)
            model_thresholds.append(threshold)
            model_aurocs.append(auroc)

            print(f"    Threshold: {threshold:.4f}, AUROC: {auroc:.4f}")

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if model_probs:
            avg_model_probs = np.mean(model_probs, axis=0)
            avg_model_threshold = np.mean(model_thresholds)

            per_model_results[model_name] = {
                "probabilities": avg_model_probs,
                "predictions": (avg_model_probs >= avg_model_threshold).astype(int),
                "threshold": avg_model_threshold,
                "n_folds": len(model_probs),
                "avg_auroc": np.mean(model_aurocs),
            }

            all_probabilities.extend(model_probs)
            all_thresholds.extend(model_thresholds)
            all_aurocs.extend(model_aurocs)

    if not all_probabilities:
        raise ValueError("No valid model checkpoints found!")

    # Apply ensemble strategy
    if strategy == "hard_weighted":
        # Weighted hard voting: apply each threshold individually,
        # then weighted average of binary predictions by AUROC
        weights = np.array(all_aurocs)
        if weights.sum() == 0:
            weights = np.ones_like(weights)  # fallback to equal weights
        weights = weights / weights.sum()  # normalize

        hard_preds = []
        for probs, threshold in zip(all_probabilities, all_thresholds):
            hard_preds.append((probs >= threshold).astype(float))

        # Weighted average of binary predictions = confidence score
        hard_preds = np.array(hard_preds)  # shape: [n_models_folds, n_images]
        ensemble_confidence = np.average(hard_preds, axis=0, weights=weights)

        # Final prediction: majority vote at 0.5
        ensemble_preds = (ensemble_confidence >= 0.5).astype(int)
        ensemble_threshold = 0.5  # threshold on the confidence score

        print(f"\nWeights (by AUROC): {dict(zip(range(len(weights)), [f'{w:.4f}' for w in weights]))}")

        return {
            "probabilities": ensemble_confidence,  # This is the confidence score
            "predictions": ensemble_preds,
            "threshold": ensemble_threshold,
            "per_model": per_model_results,
            "image_paths": valid_paths,
            "strategy": "hard_weighted",
            "weights": weights.tolist(),
        }
    else:
        # Soft voting: average probabilities, apply averaged threshold
        ensemble_probs = np.mean(all_probabilities, axis=0)
        ensemble_threshold = np.mean(all_thresholds)
        ensemble_preds = (ensemble_probs >= ensemble_threshold).astype(int)

        return {
            "probabilities": ensemble_probs,
            "predictions": ensemble_preds,
            "threshold": ensemble_threshold,
            "per_model": per_model_results,
            "image_paths": valid_paths,
            "strategy": "soft",
        }


def print_results(results: Dict[str, np.ndarray]) -> None:
    """Pretty-print ensemble prediction results.

    Args:
        results: Dict from ensemble_predict()
    """
    print("\n" + "=" * 70)
    print("ENSEMBLE PREDICTION RESULTS")
    print(f"Strategy: {results.get('strategy', 'soft')}")
    print("=" * 70)

    # Per-model summary
    if results["per_model"]:
        print("\nPer-Model Results:")
        print("-" * 50)
        for model_name, model_result in results["per_model"].items():
            print(
                f"  {model_name} ({model_result['n_folds']} folds, "
                f"threshold={model_result['threshold']:.4f}, "
                f"auroc={model_result.get('avg_auroc', 'N/A'):.4f})"
            )

    # Ensemble results
    print(f"\nEnsemble Threshold: {results['threshold']:.4f}")
    total_folds = sum(r["n_folds"] for r in results["per_model"].values())
    print(f"Total models×folds used: {total_folds}")

    if "weights" in results:
        print(f"Weights: {[f'{w:.4f}' for w in results['weights']]}")

    prob_label = "Confidence" if results.get("strategy") == "hard_weighted" else "Prob"

    print(f"\nPer-Image Predictions:")
    print("-" * 70)
    print(f"{'Image':<40} {prob_label:>8} {'Pred':>6} {'Label':>10}")
    print("-" * 70)

    for i, path in enumerate(results["image_paths"]):
        prob = results["probabilities"][i]
        pred = results["predictions"][i]
        label = "MALIGNANT" if pred == 1 else "BENIGN"
        name = os.path.basename(path)
        if len(name) > 38:
            name = "..." + name[-35:]
        print(f"  {name:<38} {prob:>8.4f} {pred:>6} {label:>10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble prediction for skin cancer detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto img_size, soft voting (default)
  python src/ensemble_predict.py \\
      --models efficientnet_b0 \\
      --checkpoint-dir checkpoints/ \\
      --image-paths test1.jpg test2.jpg

  # Multi-model ensemble with hard weighted voting
  python src/ensemble_predict.py \\
      --models efficientnet_b0 convnext_large swin_large \\
      --checkpoint-dir checkpoints/ \\
      --image-dir test_images/ \\
      --strategy hard_weighted

  # Override image size (disables auto-detection)
  python src/ensemble_predict.py \\
      --models eva02_large \\
      --checkpoint-dir checkpoints/ \\
      --image-paths test.jpg \\
      --img-size 448
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to use (e.g., efficientnet_b0 convnext_large)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Base directory containing model checkpoint subdirectories (default: checkpoints)",
    )
    parser.add_argument(
        "--image-paths",
        nargs="+",
        default=None,
        help="Paths to individual image files",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images to predict on",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Image size override (default: auto-detect from model backbone)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["soft", "hard_weighted"],
        default="soft",
        help="Ensemble strategy: 'soft' (avg probs) or 'hard_weighted' (threshold then weighted vote)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional: save results to CSV file",
    )

    args = parser.parse_args()

    # Collect image paths
    image_paths = []

    if args.image_paths:
        image_paths.extend(args.image_paths)

    if args.image_dir:
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        for f in sorted(os.listdir(args.image_dir)):
            if Path(f).suffix.lower() in img_extensions:
                image_paths.append(os.path.join(args.image_dir, f))

    if not image_paths:
        parser.error("No images specified. Use --image-paths or --image-dir")

    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)

    # Run ensemble prediction
    print(f"Models: {args.models}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    img_size_str = str(args.img_size) if args.img_size else "auto-detect"
    print(f"Image size: {img_size_str}")
    print(f"Strategy: {args.strategy}")
    print(f"Number of images: {len(image_paths)}")

    results = ensemble_predict(
        models=args.models,
        checkpoint_dir=args.checkpoint_dir,
        image_paths=image_paths,
        img_size=args.img_size,
        device=device,
        strategy=args.strategy,
    )

    # Print results
    print_results(results)

    # Optionally save to CSV
    if args.output_csv:
        import pandas as pd

        df = pd.DataFrame(
            {
                "image_path": results["image_paths"],
                "probability": results["probabilities"],
                "prediction": results["predictions"],
                "label": [
                    "MALIGNANT" if p == 1 else "BENIGN"
                    for p in results["predictions"]
                ],
            }
        )
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
