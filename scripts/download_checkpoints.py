#!/usr/bin/env python3
"""Download model checkpoints from Hugging Face Hub.

Downloads pretrained checkpoints to the local `checkpoints/` directory,
preserving the expected folder structure:
    checkpoints/{model_name}/fold_{N}/{checkpoint_file}.ckpt

Usage:
    # Download all available checkpoints
    python scripts/download_checkpoints.py

    # Download only a specific model
    python scripts/download_checkpoints.py --model efficientnet_b0

    # Download to a custom directory
    python scripts/download_checkpoints.py --output-dir /path/to/checkpoints

    # List available models without downloading
    python scripts/download_checkpoints.py --list
"""

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def _load_hf_repo_id() -> str:
    """Load HF repo ID from Hydra config (configs/paths/default.yaml)."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "paths", "default.yaml")
    if OmegaConf is not None and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        repo_id = cfg.get("hf_repo_id")
        if repo_id and isinstance(repo_id, str):
            return repo_id
    # Fallback if config not found or omegaconf not installed
    return "RudraShivm/skin-cancer-detection-isic2024"


HF_REPO_ID = _load_hf_repo_id()


def list_available_models(repo_id: str) -> dict[str, list[str]]:
    """List available models and their checkpoints on Hugging Face Hub.

    Returns:
        Dict mapping model names to lists of checkpoint file paths.
    """
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print(f"Error accessing repository '{repo_id}': {e}")
        print("\nMake sure the repository exists and is public.")
        print(f"Expected URL: https://huggingface.co/{repo_id}")
        sys.exit(1)

    # Filter for .ckpt files and organize by model name
    models: dict[str, list[str]] = {}
    for f in files:
        if f.endswith(".ckpt"):
            parts = f.split("/")
            if len(parts) >= 2:
                model_name = parts[0]
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(f)

    return models


def download_checkpoints(
    repo_id: str,
    output_dir: str,
    model_filter: str | None = None,
) -> None:
    """Download checkpoint files from Hugging Face Hub.

    Args:
        repo_id: HF repo ID (e.g. 'RudraShivm/skin-cancer-detection-isic2024')
        output_dir: Local directory to save checkpoints
        model_filter: If specified, only download checkpoints for this model
    """
    models = list_available_models(repo_id)

    if not models:
        print(f"No checkpoint files found in repository '{repo_id}'.")
        print("Make sure you have uploaded .ckpt files to the repository.")
        return

    # Filter to specific model if requested
    if model_filter:
        if model_filter not in models:
            print(f"Model '{model_filter}' not found in repository.")
            print(f"Available models: {', '.join(sorted(models.keys()))}")
            return
        models = {model_filter: models[model_filter]}

    # Count total files
    total_files = sum(len(files) for files in models.values())
    print(f"Found {total_files} checkpoint file(s) across {len(models)} model(s)")
    print(f"Downloading to: {output_dir}\n")

    downloaded = 0
    skipped = 0

    for model_name, files in sorted(models.items()):
        print(f"📦 {model_name} ({len(files)} checkpoint(s))")
        for filepath in sorted(files):
            local_path = os.path.join(output_dir, filepath)

            # Skip if already exists
            if os.path.exists(local_path):
                print(f"   ⏭️  {filepath} (already exists)")
                skipped += 1
                continue

            # Download
            print(f"   ⬇️  {filepath} ...", end=" ", flush=True)
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filepath,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                )
                print("✅")
                downloaded += 1
            except Exception as e:
                print(f"❌ Error: {e}")

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Total: {total_files}")


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Download all checkpoints
  %(prog)s --model efficientnet_b0      # Download specific model
  %(prog)s --list                       # List available models
  %(prog)s --output-dir ./my_ckpts      # Custom output directory
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Download only checkpoints for this model (e.g. 'efficientnet_b0')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for checkpoints (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help=f"Hugging Face repository ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit (no download)",
    )
    args = parser.parse_args()

    if args.list:
        models = list_available_models(args.repo_id)
        if not models:
            print("No models found.")
            return
        print(f"Available models in {args.repo_id}:\n")
        for name, files in sorted(models.items()):
            print(f"  📦 {name}")
            for f in sorted(files):
                print(f"     └── {f}")
            print()
        return

    download_checkpoints(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()
