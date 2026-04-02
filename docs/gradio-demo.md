# 🎨 Gradio Demo — Skin Cancer Detection UI

> **[← Back to README](../README.md)**

A web-based prediction interface for the ISIC 2024 skin cancer detection model. Upload a skin lesion image, select model checkpoint(s), and get a malignant/benign prediction with confidence score.

The demo is maintained as a Git submodule that natively deploys to [Hugging Face Spaces](https://huggingface.co/spaces/RudraShivm/skin-cancer-detection-demo).

## Table of Contents

- [Running Locally](#running-locally)
- [Code Structure (in `demo/`)](#code-structure-in-demo)

---

## Running Locally

If you have PyTorch installed locally:

```bash
# 1. Pull the demo submodule (if you haven't already)
git submodule update --init --recursive

# 2. Authenticate with Hugging Face if needed
hf auth login
# or: huggingface-cli login

# 3. Download pretrained checkpoints
python scripts/download_checkpoints.py

# 4. Launch the app
python demo/app.py
```

Then open `http://localhost:7860` in your browser. The app automatically looks for checkpoints in the parent repo's `checkpoints/` directory.

---

## Code Structure (in `demo/`)

| File | Purpose |
|------|---------|
| `app.py` | Self-contained Gradio app (auto-downloads checkpoints, runs inference) |
| `requirements.txt` | Minimal dependencies for the Space |
| `README.md` | HF Spaces YAML configuration |
| `samples/` | Sample images and `samples.json` metadata |

> 📖 [Technical reference](reference.md) &nbsp;|&nbsp; [Architecture docs](architecture.md)
