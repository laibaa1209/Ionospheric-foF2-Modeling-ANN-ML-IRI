"""
Scenario 2 — Train on non-quiet days, predict quiet days.
==========================================================
Training data : master_no_quiet.csv
Evaluation    : quiet_days_only.csv

Saved to ann_evals/outputs/:
  ann_no_quiet_model.pt
  ann_no_quiet_loss_curve.png
  ann_no_quiet_scatter.png
  predictions_ann_no_quiet.csv

Usage:
    python train_no_quiet.py
    (run ann_evals/data/prepare_datasets.py first)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent
ann_evals    = script_dir.parent
data_dir     = ann_evals / "data"
output_dir   = ann_evals / "outputs" / "scenario2_no_quiet"
output_dir.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = data_dir / "master_no_quiet.csv"
EVAL_CSV  = data_dir / "quiet_days_only.csv"

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FEATURES     = ["DayOfYear", "Hour", "Longitude", "Latitude", "F10.7"]
TARGET       = "foF2"
BATCH_SIZE   = 128
LR           = 1e-3
MAX_EPOCHS   = 200
PATIENCE     = 12
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATIONS       = ["CP", "ElginAB", "Jicamarca", "MilstonHill", "Ramey"]
STATION_COLOR  = dict(zip(STATIONS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
STATION_MARKER = dict(zip(STATIONS, ["o", "s", "^", "D", "v"]))

print("=" * 65)
print("SCENARIO 2 — Train on Non-Quiet Days, Predict Quiet Days")
print("=" * 65)
print(f"Device : {DEVICE}")
print()

# ── Load ──────────────────────────────────────────────────────────────────────
df_train = pd.read_csv(TRAIN_CSV)
df_eval  = pd.read_csv(EVAL_CSV)
print(f"Training data : {len(df_train):,} rows  ({TRAIN_CSV.name})")
print(f"Evaluation    : {len(df_eval):,} rows  ({EVAL_CSV.name})")
print()

X_all  = df_train[FEATURES].values.astype(np.float32)
y_all  = df_train[TARGET].values.astype(np.float32).reshape(-1, 1)

X_eval = df_eval[FEATURES].values.astype(np.float32)
y_eval = df_eval[TARGET].values.astype(np.float32)
s_eval = df_eval["Station"].values

# ── Train / Val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.15 / 0.85, random_state=RANDOM_STATE)
print(f"Train  : {len(X_train):,}  Val: {len(X_val):,}")
print()

# ── Normalise ─────────────────────────────────────────────────────────────────
train_mean = X_train.mean(axis=0)
train_std  = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))

X_train_n = (X_train - train_mean) / train_std
X_val_n   = (X_val   - train_mean) / train_std
X_eval_n  = (X_eval  - train_mean) / train_std

# ── Model ─────────────────────────────────────────────────────────────────────
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

model     = ANNModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

def to_t(a):
    return torch.from_numpy(a).to(DEVICE)

loader = DataLoader(
    TensorDataset(to_t(X_train_n), to_t(y_train)),
    batch_size=BATCH_SIZE, shuffle=True)
X_va_t = to_t(X_val_n);  y_va_t = to_t(y_val)

# ── Training ──────────────────────────────────────────────────────────────────
best_val_loss, best_epoch, patience_count, best_state = float("inf"), 0, 0, None
train_losses, val_losses = [], []

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for Xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * Xb.size(0)
    epoch_loss /= len(X_train)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_va_t), y_va_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS}  train={epoch_loss:.5f}  val={val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss, best_epoch, patience_count = val_loss, epoch, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stop @ epoch {epoch+1}  (best val={best_val_loss:.5f} @ {best_epoch+1})")
            break

model.load_state_dict(best_state)
print(f"\nBest epoch: {best_epoch+1}  |  val MSE={best_val_loss:.5f}")

# ── Save checkpoint ───────────────────────────────────────────────────────────
ck_path = output_dir / "ann_no_quiet_model.pt"
torch.save({
    "model_state_dict": model.state_dict(),
    "train_mean":       train_mean,
    "train_std":        train_std,
    "features":         FEATURES,
    "epoch":            best_epoch,
}, ck_path)
print(f"Checkpoint: {ck_path}")

# ── Loss curve ────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.axvline(best_epoch, color="r", linestyle="--", label=f"Best epoch ({best_epoch+1})")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Scenario 2 — No Quiet Days: Training & Validation Loss")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
lc_path = output_dir / "ann_no_quiet_loss_curve.png"
plt.savefig(lc_path, dpi=150); plt.close()
print(f"Loss curve: {lc_path}")

# ── Predict quiet days ────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred = model(to_t(X_eval_n)).cpu().numpy().flatten()

print(f"\nQUIET DAYS  MSE={mean_squared_error(y_eval, y_pred):.4f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_eval, y_pred)):.4f}  "
      f"MAE={mean_absolute_error(y_eval, y_pred):.4f}  "
      f"R2={r2_score(y_eval, y_pred):.4f}  N={len(y_eval):,}")
print("\nPER STATION (quiet days)")
for s in STATIONS:
    mask = s_eval == s
    if not mask.any():
        continue
    rmse_s = float(np.sqrt(mean_squared_error(y_eval[mask], y_pred[mask])))
    r2_s   = float(r2_score(y_eval[mask], y_pred[mask]))
    bias_s = float(np.mean(y_pred[mask] - y_eval[mask]))
    print(f"  {s:<14} RMSE={rmse_s:.4f}  R2={r2_s:.4f}  Bias={bias_s:+.4f}  N={mask.sum()}")

# ── Scatter plot ──────────────────────────────────────────────────────────────
def scatter_plot(y_obs, y_pred, stations_arr, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for s in STATIONS:
        mask = stations_arr == s
        if not mask.any():
            continue
        ax.scatter(y_obs[mask], y_pred[mask], s=8, alpha=0.4,
                   color=STATION_COLOR[s], marker=STATION_MARKER[s], label=s)
    lims = [min(y_obs.min(), y_pred.min()) - 0.3, max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed foF2 (MHz)")
    ax.set_ylabel("Predicted foF2 (MHz)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, markerscale=1.5)
    rmse = float(np.sqrt(mean_squared_error(y_obs, y_pred)))
    mae  = float(mean_absolute_error(y_obs, y_pred))
    r2   = float(r2_score(y_obs, y_pred))
    bias = float(np.mean(y_pred - y_obs))
    txt = (f"RMSE={rmse:.3f}  MAE={mae:.3f}\n"
           f"R2={r2:.4f}   Bias={bias:+.3f}\nN={len(y_obs):,}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter   : {save_path}")

scatter_plot(y_eval, y_pred, s_eval,
             "Scenario 2 — Quiet Day Predictions",
             output_dir / "ann_no_quiet_scatter.png")

# ── Save predictions ──────────────────────────────────────────────────────────
pred_path = output_dir / "predictions_ann_no_quiet.csv"
pd.DataFrame({
    "Station":   s_eval,
    "DayOfYear": df_eval["DayOfYear"].values,
    "Hour":      df_eval["Hour"].values,
    "foF2_obs":  y_eval,
    "foF2_pred": y_pred,
}).to_csv(pred_path, index=False)
print(f"Predictions: {pred_path}  ({len(df_eval):,} rows)")
print("\nDone.")
