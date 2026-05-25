"""
ANN Predictions for Equinox and Solstice Days
==============================================
Loads the ANN trained on the full FINAL_MASTER.csv dataset
(ann_evals/outputs/ann_full_model.pt) and runs inference on
the held-out equinox/solstice observations in ann_evals/data/equinox_solstice.csv.

These dates were excluded from FINAL_MASTER.csv during preprocessing,
so this is a true out-of-distribution evaluation.

Input  : ann_evals/data/equinox_solstice.csv
Model  : ann_evals/outputs/ann_full_model.pt
Outputs:
  ann_evals/outputs/predictions_ann_equinox_solstice.csv
  ann_evals/outputs/ann_equinox_solstice_scatter.png

Usage:
    python predict_equinox_solstice.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent          # ann_evals/python_scripts/
evals_dir    = script_dir.parent                        # ann_evals/
data_dir     = evals_dir / "data"
output_dir   = evals_dir / "outputs" / "scenario5_equinox_solstice"
output_dir.mkdir(parents=True, exist_ok=True)

DATA_CSV     = data_dir  / "equinox_solstice.csv"
CHECKPOINT   = evals_dir / "outputs" / "scenario1_full" / "ann_full_model.pt"

FEATURES  = ["DayOfYear", "Hour", "Longitude", "Latitude", "F10.7"]
TARGET    = "foF2"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATIONS       = ["CP", "ElginAB", "Jicamarca", "MilstonHill", "Ramey"]
STATION_COLOR  = dict(zip(STATIONS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
STATION_MARKER = dict(zip(STATIONS, ["o", "s", "^", "D", "v"]))

DOY_LABEL = {79: "Mar Equinox (79)", 172: "Jun Solstice (172)",
             266: "Sep Equinox (266)", 356: "Dec Solstice (356)"}

print("=" * 65)
print("ANN PREDICTIONS — EQUINOX & SOLSTICE DAYS")
print("=" * 65)
print(f"Device: {DEVICE}")
print()

# ── Load checkpoint ───────────────────────────────────────────────────────────
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}\n"
        "Run ann_evals/python_scripts/train_full.py first."
    )

ck = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
train_mean = ck["train_mean"]
train_std  = ck["train_std"]
print(f"Loaded checkpoint : {CHECKPOINT.name}  (best epoch {ck['epoch']+1})")
print(f"Features          : {ck['features']}")
print()

# ── Rebuild model ─────────────────────────────────────────────────────────────
class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,   16), nn.ReLU(),
            nn.Linear(16,  32), nn.ReLU(),
            nn.Linear(32,  64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64,  32), nn.ReLU(),
            nn.Linear(32,  16), nn.ReLU(),
            nn.Linear(16,   1),
        )
    def forward(self, x):
        return self.net(x)

model = ANNModel().to(DEVICE)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print(f"Model parameters  : {sum(p.numel() for p in model.parameters()):,}")
print()

# ── Load data ─────────────────────────────────────────────────────────────────
if not DATA_CSV.exists():
    raise FileNotFoundError(f"Input data not found: {DATA_CSV}")

df       = pd.read_csv(DATA_CSV)
X        = df[FEATURES].values.astype(np.float32)
y        = df[TARGET].values.astype(np.float32)
stations = df["Station"].values

print(f"Loaded: {DATA_CSV.name}")
print(f"  {len(df):,} rows  |  stations: {sorted(df['Station'].unique())}")
print(f"  DOYs: {sorted(df['DayOfYear'].unique())}")
print()

# ── Normalise using training statistics ───────────────────────────────────────
X_norm = ((X - train_mean) / train_std).astype(np.float32)

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    y_pred = model(torch.from_numpy(X_norm).to(DEVICE)).cpu().numpy().flatten()

# ── Overall metrics ───────────────────────────────────────────────────────────
mse  = float(mean_squared_error(y, y_pred))
rmse = float(np.sqrt(mse))
mae  = float(mean_absolute_error(y, y_pred))
r2   = float(r2_score(y, y_pred))
bias = float(np.mean(y_pred - y))

print("=" * 65)
print("OVERALL METRICS (equinox + solstice, all stations)")
print("=" * 65)
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f} MHz")
print(f"  MAE  = {mae:.4f} MHz")
print(f"  R²   = {r2:.4f}")
print(f"  Bias = {bias:+.4f} MHz")
print(f"  N    = {len(y):,}")
print()

# ── Per-DOY metrics ───────────────────────────────────────────────────────────
print("=" * 65)
print("PER-DAY METRICS")
print("=" * 65)
per_doy_rows = []
for doy in sorted(df["DayOfYear"].unique()):
    mask = df["DayOfYear"].values == doy
    yt, yp = y[mask], y_pred[mask]
    rmse_d = float(np.sqrt(mean_squared_error(yt, yp)))
    r2_d   = float(r2_score(yt, yp))
    bias_d = float(np.mean(yp - yt))
    mae_d  = float(mean_absolute_error(yt, yp))
    label  = DOY_LABEL.get(doy, f"DOY {doy}")
    print(f"  {label:<25}  RMSE={rmse_d:.4f}  R²={r2_d:.4f}  "
          f"MAE={mae_d:.4f}  Bias={bias_d:+.4f}  N={mask.sum()}")
    per_doy_rows.append({"DOY": doy, "Event": label, "RMSE": rmse_d,
                         "MAE": mae_d, "R2": r2_d, "Bias": bias_d, "N": int(mask.sum())})
print()

# ── Per-station metrics ───────────────────────────────────────────────────────
print("=" * 65)
print("PER-STATION METRICS")
print("=" * 65)
per_stn_rows = []
for stn in STATIONS:
    mask = stations == stn
    if not mask.any():
        continue
    yt, yp = y[mask], y_pred[mask]
    rmse_s = float(np.sqrt(mean_squared_error(yt, yp)))
    r2_s   = float(r2_score(yt, yp))
    bias_s = float(np.mean(yp - yt))
    mae_s  = float(mean_absolute_error(yt, yp))
    print(f"  {stn:<14}  RMSE={rmse_s:.4f}  R²={r2_s:.4f}  "
          f"MAE={mae_s:.4f}  Bias={bias_s:+.4f}  N={mask.sum()}")
    per_stn_rows.append({"Station": stn, "RMSE": rmse_s, "MAE": mae_s,
                         "R2": r2_s, "Bias": bias_s, "N": int(mask.sum())})
print()

# ── Save predictions CSV ──────────────────────────────────────────────────────
out_df = df[["Station", "DayOfYear", "Hour", "Longitude", "Latitude", "F10.7", "foF2"]].copy()
out_df.rename(columns={"foF2": "foF2_obs"}, inplace=True)
out_df["foF2_pred"] = y_pred.round(4)
out_df["residual"]  = (out_df["foF2_pred"] - out_df["foF2_obs"]).round(4)

pred_path = output_dir / "predictions_ann_equinox_solstice.csv"
out_df.to_csv(pred_path, index=False)
print(f"Predictions saved : {pred_path.name}  ({len(out_df):,} rows)")

# ── Scatter plot (coloured by station) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

for stn in STATIONS:
    mask = stations == stn
    if not mask.any():
        continue
    ax.scatter(y[mask], y_pred[mask], s=20, alpha=0.6,
               color=STATION_COLOR[stn], marker=STATION_MARKER[stn], label=stn)

lims = [min(y.min(), y_pred.min()) - 0.3, max(y.max(), y_pred.max()) + 0.3]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed foF2 (MHz)")
ax.set_ylabel("Predicted foF2 (MHz)")
ax.set_title("Scenario 5 — Equinox & Solstice Prediction", fontsize=10)
ax.legend(fontsize=8, markerscale=1.4)

txt = (f"RMSE={rmse:.3f}  MAE={mae:.3f}\n"
       f"R²={r2:.4f}   Bias={bias:+.3f}\nN={len(y):,}")
ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.tight_layout()
scatter_path = output_dir / "ann_equinox_solstice_scatter.png"
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"Scatter plot saved: {scatter_path.name}")

print()
print("=" * 65)
print("DONE")
print("=" * 65)
