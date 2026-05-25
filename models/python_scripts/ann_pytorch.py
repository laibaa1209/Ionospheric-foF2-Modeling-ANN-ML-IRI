"""
PyTorch ANN for foF2 Prediction
================================
Architecture : Input(5) → 16 → 32 → 64 → 128 → 64 → 32 → 16 → 1  (ReLU, linear out)
Normalisation: mean/std computed on training data, applied at inference
Optimizer    : Adam (lr=1e-3)   Loss: MSE   Batch: 128
Early stopping: patience=12 on val loss, best weights restored
Max epochs   : 200
Split        : 70 % train / 15 % val / 15 % test  (random_state=42)
Outputs      : outputs/best_model.pt         (checkpoint)
               outputs/ann_loss_curve.png    (training history)
               outputs/scatter_ann.png       (all-data scatter)
               outputs/predictions_ann.csv   (all 37 035 rows)
"""

import numpy as np
import pandas as pd
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
project_root = Path(__file__).resolve().parent.parent.parent
data_path    = project_root / "FINAL_MASTER.csv"
output_dir   = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FEATURES     = ["DayOfYear", "Hour", "Longitude", "Latitude", "F10.7"]
TARGET       = "foF2"
BATCH_SIZE   = 128
LR           = 1e-3
MAX_EPOCHS   = 200
PATIENCE     = 12
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}\n")

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data: {data_path}")
df       = pd.read_csv(data_path)
X        = df[FEATURES].values.astype(np.float32)
y        = df[TARGET].values.astype(np.float32).reshape(-1, 1)
stations = df["Station"].values
print(f"  {len(df):,} rows  |  {df['Station'].nunique()} stations\n")

# ── 70 / 15 / 15 split ───────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
    X, y, stations, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
    X_temp, y_temp, s_temp, test_size=0.50, random_state=RANDOM_STATE)

print(f"Train : {len(X_train):,}  ({len(X_train)/len(X)*100:.1f} %)")
print(f"Val   : {len(X_val):,}  ({len(X_val)/len(X)*100:.1f} %)")
print(f"Test  : {len(X_test):,}  ({len(X_test)/len(X)*100:.1f} %)\n")

# ── Normalise on training data ────────────────────────────────────────────────
train_mean = X_train.mean(axis=0)
train_std  = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))

X_train_n = (X_train - train_mean) / train_std
X_val_n   = (X_val   - train_mean) / train_std
X_test_n  = (X_test  - train_mean) / train_std

print("Normalisation stats (train):")
for f, m, s in zip(FEATURES, train_mean, train_std):
    print(f"  {f:<12}  mean={m:.4f}  std={s:.4f}")
print()

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

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}\n")

# ── Tensors & DataLoader ──────────────────────────────────────────────────────
def to_tensor(arr):
    return torch.from_numpy(arr).to(DEVICE)

X_tr_t = to_tensor(X_train_n); y_tr_t = to_tensor(y_train)
X_va_t = to_tensor(X_val_n);   y_va_t = to_tensor(y_val)
X_te_t = to_tensor(X_test_n)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

# ── Training loop ─────────────────────────────────────────────────────────────
print("=" * 55)
print("TRAINING")
print("=" * 55)

best_val_loss   = float("inf")
best_epoch      = 0
patience_count  = 0
best_state      = None
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
    epoch_loss /= len(X_tr_t)
    train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_va_t), y_va_t).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{MAX_EPOCHS}  train={epoch_loss:.5f}  val={val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        patience_count = 0
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch+1}  (best val={best_val_loss:.5f} @ epoch {best_epoch+1})")
            break

model.load_state_dict(best_state)
print(f"\nBest epoch: {best_epoch+1}  |  val MSE={best_val_loss:.5f}\n")

# ── Save checkpoint ───────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "train_mean":       train_mean,
    "train_std":        train_std,
    "features":         FEATURES,
    "epoch":            best_epoch,
}, output_dir / "best_model.pt")
print("Checkpoint: outputs/best_model.pt")

# ── Test-set diagnostics ──────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_test_pred = model(X_te_t).cpu().numpy().flatten()

y_test_flat = y_test.flatten()
mse_t  = mean_squared_error(y_test_flat, y_test_pred)
rmse_t = np.sqrt(mse_t)
r2_t   = r2_score(y_test_flat, y_test_pred)

print("\n" + "=" * 55)
print("TEST-SET METRICS (diagnostic)")
print("=" * 55)
print(f"  MSE={mse_t:.4f}  RMSE={rmse_t:.4f}  R2={r2_t:.4f}\n")

# ── All-data prediction ───────────────────────────────────────────────────────
X_all_n = ((X - train_mean) / train_std).astype(np.float32)
with torch.no_grad():
    y_pred_all = model(to_tensor(X_all_n)).cpu().numpy().flatten()

y_flat   = y.flatten()
mse_all  = mean_squared_error(y_flat, y_pred_all)
rmse_all = np.sqrt(mse_all)
r2_all   = r2_score(y_flat, y_pred_all)

print("=" * 55)
print("ALL-DATA METRICS")
print("=" * 55)
print(f"  MSE={mse_all:.4f}  RMSE={rmse_all:.4f}  MAE={mean_absolute_error(y_flat, y_pred_all):.4f}  R2={r2_all:.4f}  N={len(y_flat):,}")
print()

# ── Per-station ───────────────────────────────────────────────────────────────
print("=" * 55)
print("PER-STATION METRICS (all data)")
print("=" * 55)
for s in sorted(np.unique(stations)):
    mask = stations == s
    yt, yp = y_flat[mask], y_pred_all[mask]
    rmse_s = float(np.sqrt(mean_squared_error(yt, yp)))
    r2_s   = float(r2_score(yt, yp))
    bias_s = float(np.mean(yp - yt))
    print(f"  {s:<14} RMSE={rmse_s:.4f}  R2={r2_s:.4f}  Bias={bias_s:+.4f}  N={mask.sum()}")
print()

# ── Loss curve ────────────────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.axvline(best_epoch, color="r", linestyle="--", label=f"Best epoch ({best_epoch+1})")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("ANN Training & Validation Loss")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(output_dir / "ann_loss_curve.png", dpi=150)
plt.close()
print("Loss curve: outputs/ann_loss_curve.png")

# ── Scatter plot (all data) ───────────────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y_flat, y_pred_all, alpha=0.3, s=8)
lims = [min(y_flat.min(), y_pred_all.min()) - 0.3, max(y_flat.max(), y_pred_all.max()) + 0.3]
plt.plot(lims, lims, "r--", lw=1.5, label="1:1")
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Observed foF2 (MHz)"); plt.ylabel("Predicted foF2 (MHz)")
plt.title(f"ANN (PyTorch) — All Data\nR²={r2_all:.4f}  RMSE={rmse_all:.4f}")
plt.legend(); plt.tight_layout()
plt.savefig(output_dir / "scatter_ann.png", dpi=150)
plt.close()
print("Scatter   : outputs/scatter_ann.png")

# ── Save predictions CSV ──────────────────────────────────────────────────────
pd.DataFrame({
    "Station":   stations,
    "DayOfYear": df["DayOfYear"].values,
    "Hour":      df["Hour"].values,
    "foF2_obs":  y_flat,
    "foF2_pred": y_pred_all,
}).to_csv(output_dir / "predictions_ann.csv", index=False)
print(f"Predictions: outputs/predictions_ann.csv  ({len(df):,} rows)")
