"""Export tabular feature normalization statistics for inference.

During training, the ISICDataModule standardizes tabular features
(zero mean, unit variance). This script computes those stats from
the full training CSV and saves them as JSON so that the Gradio app
can apply the same normalization at inference time.

Usage:
    python scripts/export_tab_stats.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.isic_datamodule import TABULAR_NUM_COLS

# ─── Config ───
DATA_DIR = "data/isic-2024-challenge"
METADATA_FILE = "train-metadata.csv"
OUTPUT_PATH = "demo/tab_stats.json"

print(f"Loading metadata from {DATA_DIR}/{METADATA_FILE}...")
df = pd.read_csv(os.path.join(DATA_DIR, METADATA_FILE), low_memory=False)
print(f"  Loaded {len(df)} rows")

# ─── Replicate _encode_tabular logic exactly ───
parts = []

# Numeric features
num_data = df.reindex(columns=TABULAR_NUM_COLS, fill_value=0).fillna(0).values.astype(np.float32)
parts.append(num_data)

# Sex: male=1, female=0, unknown=0.5
sex_map = {'male': 1.0, 'female': 0.0}
sex_vals = df['sex'].map(sex_map).fillna(0.5).values.astype(np.float32).reshape(-1, 1)
parts.append(sex_vals)

# Anatomical site: one-hot
site_categories = [
    'head/neck', 'upper extremity', 'lower extremity',
    'anterior torso', 'posterior torso', 'lateral torso', 'palms/soles'
]
site_encoded = np.zeros((len(df), len(site_categories)), dtype=np.float32)
for i, cat in enumerate(site_categories):
    site_encoded[:, i] = (df['anatom_site_general'] == cat).astype(np.float32)
parts.append(site_encoded)

full_tabular = np.hstack(parts)
print(f"  Tabular shape: {full_tabular.shape}")

# ─── Compute stats ───
t_mean = full_tabular.mean(axis=0)
t_std = full_tabular.std(axis=0)
t_std[t_std < 1e-7] = 1.0  # avoid div-by-zero

stats = {
    "mean": t_mean.tolist(),
    "std": t_std.tolist(),
    "n_features": int(full_tabular.shape[1]),
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"✅ Saved {OUTPUT_PATH} ({stats['n_features']} features)")
print(f"   Mean range: [{t_mean.min():.4f}, {t_mean.max():.4f}]")
print(f"   Std range:  [{t_std.min():.4f}, {t_std.max():.4f}]")
