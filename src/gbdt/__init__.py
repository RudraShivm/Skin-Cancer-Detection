"""GBDT stacking module for ISIC skin cancer detection.

Implements Stage 2 of the CNN → GBDT stacking pipeline:
1. Extract CNN predictions + tabular features (extract_cnn_features.py)
2. Train LightGBM, XGBoost, CatBoost on top (train_gbdt.py)
3. Run GBDT ensemble inference (predict_gbdt.py)
"""
