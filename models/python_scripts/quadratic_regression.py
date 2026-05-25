"""
Quadratic Regression Model for foF2 Prediction
===============================================
Features  : DayOfYear, Hour, Longitude, Latitude, F10.7
Target    : foF2 (MHz)
Pipeline  : PolynomialFeatures(degree=2) → StandardScaler → LinearRegression
Split     : 70 % train / 15 % val / 15 % test  (random_state=42)
Evaluation: test-set diagnostics + full-dataset metrics + per-station breakdown
Outputs   : outputs/quadratic_model.joblib  (saved pipeline)
            outputs/scatter_quadratic.png   (all-data scatter)
            outputs/predictions_quadratic.csv (all 37 035 rows)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent.parent
data_path    = project_root / "FINAL_MASTER.csv"
output_dir   = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FEATURES     = ["DayOfYear", "Hour", "Longitude", "Latitude", "F10.7"]
TARGET       = "foF2"
POLY_DEGREE  = 2

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data: {data_path}")
df       = pd.read_csv(data_path)
X        = df[FEATURES].values
y        = df[TARGET].values
stations = df["Station"].values
print(f"  {len(df):,} rows  |  {df['Station'].nunique()} stations")
print()

# ── 70 / 15 / 15 split ───────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
    X, y, stations, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
    X_temp, y_temp, s_temp, test_size=0.50, random_state=RANDOM_STATE)

print(f"Train : {len(X_train):,}  ({len(X_train)/len(X)*100:.1f} %)")
print(f"Val   : {len(X_val):,}  ({len(X_val)/len(X)*100:.1f} %)")
print(f"Test  : {len(X_test):,}  ({len(X_test)/len(X)*100:.1f} %)")
print()

# ── Metrics helper ────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return None
    return dict(
        MSE  = float(mean_squared_error(yt, yp)),
        RMSE = float(np.sqrt(mean_squared_error(yt, yp))),
        MAE  = float(mean_absolute_error(yt, yp)),
        R2   = float(r2_score(yt, yp)),
        Bias = float(np.mean(yp - yt)),
        N    = int(mask.sum()),
    )

# ── Build pipeline & train ────────────────────────────────────────────────────
print(f"Building Quadratic Regression pipeline (degree={POLY_DEGREE})...")
model = Pipeline([
    ("poly",      PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
    ("scaler",    StandardScaler()),
    ("regressor", LinearRegression()),
])
model.fit(X_train, y_train)
print("Done.\n")

# ── Test-set diagnostics ──────────────────────────────────────────────────────
m = metrics(y_test, model.predict(X_test))
print("=" * 55)
print("TEST-SET METRICS (diagnostic)")
print("=" * 55)
print(f"  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}  Bias={m['Bias']:+.4f}")
print()

# ── All-data evaluation ───────────────────────────────────────────────────────
y_pred_all = model.predict(X)
m_all = metrics(y, y_pred_all)

print("=" * 55)
print("ALL-DATA METRICS")
print("=" * 55)
print(f"  MSE={m_all['MSE']:.4f}  RMSE={m_all['RMSE']:.4f}  MAE={m_all['MAE']:.4f}  R2={m_all['R2']:.4f}  N={m_all['N']:,}")
print()

# ── Per-station ───────────────────────────────────────────────────────────────
print("=" * 55)
print("PER-STATION METRICS (all data)")
print("=" * 55)
for s in sorted(np.unique(stations)):
    mask = stations == s
    ms = metrics(y[mask], y_pred_all[mask])
    print(f"  {s:<14} RMSE={ms['RMSE']:.4f}  R2={ms['R2']:.4f}  Bias={ms['Bias']:+.4f}  N={ms['N']}")
print()

# ── Scatter plot (all data) ───────────────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred_all, alpha=0.3, s=8)
lims = [min(y.min(), y_pred_all.min()) - 0.3, max(y.max(), y_pred_all.max()) + 0.3]
plt.plot(lims, lims, "r--", lw=1.5, label="1:1")
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Observed foF2 (MHz)"); plt.ylabel("Predicted foF2 (MHz)")
plt.title(f"Quadratic Regression (deg={POLY_DEGREE}) — All Data\nR²={m_all['R2']:.4f}  RMSE={m_all['RMSE']:.4f}")
plt.legend(); plt.tight_layout()
plt.savefig(output_dir / "scatter_quadratic.png", dpi=150)
plt.close()
print(f"Scatter saved: outputs/scatter_quadratic.png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, output_dir / "quadratic_model.joblib")
print(f"Model saved : outputs/quadratic_model.joblib")

# ── Save predictions CSV ──────────────────────────────────────────────────────
pd.DataFrame({
    "Station":   stations,
    "DayOfYear": df["DayOfYear"].values,
    "Hour":      df["Hour"].values,
    "foF2_obs":  y,
    "foF2_pred": y_pred_all,
}).to_csv(output_dir / "predictions_quadratic.csv", index=False)
print(f"Predictions : outputs/predictions_quadratic.csv  ({len(df):,} rows)")
