"""
Generate full-dataset predictions for all ML models.
=====================================================
- Linear / Quadratic: loads saved .joblib if present, otherwise trains and saves.
- ANN (PyTorch):      loads outputs/best_model.pt — never retrains.
- IRI:                reads outputs/FINAL_MASTER_IRI.csv directly (already full dataset).

All prediction CSVs are written with all 37035 rows so compare_models.py can
evaluate every model on the complete dataset.

Usage:
    python generate_predictions.py
    python compare_models.py        # run after this script
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FEATURES     = ["DayOfYear", "Hour", "Longitude", "Latitude", "F10.7"]
TARGET       = "foF2"

project_root   = Path(__file__).parent
data_path      = project_root / "FINAL_MASTER.csv"
output_dir     = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

LIN_MODEL_PATH  = output_dir / "linear_model.joblib"
QUAD_MODEL_PATH = output_dir / "quadratic_model.joblib"
ANN_MODEL_PATH  = output_dir / "best_model.pt"

LIN_PRED_PATH   = output_dir / "predictions_linear.csv"
QUAD_PRED_PATH  = output_dir / "predictions_quadratic.csv"
ANN_PRED_PATH   = output_dir / "predictions_ann.csv"
IRI_PRED_PATH   = output_dir / "predictions_iri.csv"

print("=" * 65)
print("GENERATING FULL-DATASET PREDICTIONS")
print("=" * 65)
print()

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data: {data_path}")
df       = pd.read_csv(data_path)
X        = df[FEATURES].values
y        = df[TARGET].values
stations = df["Station"].values
print(f"  {len(df):,} rows  |  features: {FEATURES}")
print()

# 70% train split — matches the seed used in all model scripts
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE)

# ── 1. Linear Regression ──────────────────────────────────────────────────────
print("--- Linear Regression ---")
if LIN_MODEL_PATH.exists():
    print(f"  Loading saved model from {LIN_MODEL_PATH.name}")
    lin_model = joblib.load(LIN_MODEL_PATH)
else:
    print("  No saved model found — training on 70% split...")
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    joblib.dump(lin_model, LIN_MODEL_PATH)
    print(f"  Saved to {LIN_MODEL_PATH.name}")

lin_pred = lin_model.predict(X)
lin_df_out = pd.DataFrame({
    "Station":   stations,
    "DayOfYear": df["DayOfYear"].values,
    "Hour":      df["Hour"].values,
    "foF2_obs":  y,
    "foF2_pred": lin_pred,
})
lin_df_out.to_csv(LIN_PRED_PATH, index=False)
print(f"  Saved {LIN_PRED_PATH.name}  ({len(lin_df_out):,} rows)")
print()

# ── 2. Quadratic Regression ───────────────────────────────────────────────────
print("--- Quadratic Regression ---")
if QUAD_MODEL_PATH.exists():
    print(f"  Loading saved model from {QUAD_MODEL_PATH.name}")
    quad_model = joblib.load(QUAD_MODEL_PATH)
else:
    print("  No saved model found — training on 70% split...")
    quad_model = Pipeline([
        ("poly",      PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler",    StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    quad_model.fit(X_train, y_train)
    joblib.dump(quad_model, QUAD_MODEL_PATH)
    print(f"  Saved to {QUAD_MODEL_PATH.name}")

quad_pred = quad_model.predict(X)
quad_df_out = pd.DataFrame({
    "Station":   stations,
    "DayOfYear": df["DayOfYear"].values,
    "Hour":      df["Hour"].values,
    "foF2_obs":  y,
    "foF2_pred": quad_pred,
})
quad_df_out.to_csv(QUAD_PRED_PATH, index=False)
print(f"  Saved {QUAD_PRED_PATH.name}  ({len(quad_df_out):,} rows)")
print()

# ── 3. ANN — load from checkpoint, never retrain ─────────────────────────────
print("--- ANN (PyTorch) ---")
if not ANN_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ANN checkpoint not found: {ANN_MODEL_PATH}\n"
        "Run ann_pytorch.py first to train and save the model."
    )

print(f"  Loading checkpoint: {ANN_MODEL_PATH.name}")
ck = torch.load(ANN_MODEL_PATH, map_location="cpu", weights_only=False)

class ANNModel(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return self.fc8(x)

ann_model = ANNModel(input_size=len(FEATURES))
ann_model.load_state_dict(ck["model_state_dict"])
ann_model.eval()

train_mean = ck["train_mean"].astype(np.float32)
train_std  = ck["train_std"].astype(np.float32)
print(f"  Checkpoint epoch: {ck['epoch'] + 1}")

X_norm   = ((X.astype(np.float32) - train_mean) / train_std)
X_tensor = torch.from_numpy(X_norm)

with torch.no_grad():
    ann_pred = ann_model(X_tensor).numpy().flatten()

ann_df_out = pd.DataFrame({
    "Station":   stations,
    "DayOfYear": df["DayOfYear"].values,
    "Hour":      df["Hour"].values,
    "foF2_obs":  y,
    "foF2_pred": ann_pred,
})
ann_df_out.to_csv(ANN_PRED_PATH, index=False)
print(f"  Saved {ANN_PRED_PATH.name}  ({len(ann_df_out):,} rows)")
print()

# ── 4. IRI — already full dataset ────────────────────────────────────────────
if not IRI_PRED_PATH.exists():
    print(f"WARNING: {IRI_PRED_PATH.name} not found in outputs/. Run run_iri.py first.")
else:
    iri_check = pd.read_csv(IRI_PRED_PATH)
    print(f"--- IRI-2020 ---")
    print(f"  {IRI_PRED_PATH.name} already covers {len(iri_check):,} rows  (CCIR + URSI)")
    print()

print("=" * 65)
print("ALL PREDICTION CSVs READY — run compare_models.py next")
print("=" * 65)
