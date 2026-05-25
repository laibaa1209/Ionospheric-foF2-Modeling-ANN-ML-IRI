"""
Model Comparison: Linear vs Quadratic vs ANN vs IRI-2020 (CCIR & URSI)
=======================================================================
All model prediction CSVs cover the entire dataset (37035 rows).
IRI source: outputs/FINAL_MASTER_IRI.csv  (foF2_C=CCIR, foF2_U=URSI)

Run order:
    1. python linear_regression.py
    2. python quadratic_regression.py
    3. python ann_pytorch.py
    4. python run_iri.py
    5. python compare_models.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent
output_dir   = project_root / "outputs"
output_dir.mkdir(exist_ok=True)

LINEAR_PATH    = output_dir / "predictions_linear.csv"
QUADRATIC_PATH = output_dir / "predictions_quadratic.csv"
ANN_PATH       = output_dir / "predictions_ann.csv"
IRI_PATH       = output_dir / "predictions_iri.csv"

print("=" * 70)
print("MODEL COMPARISON: LINEAR / QUADRATIC / ANN / IRI-CCIR / IRI-URSI")
print("=" * 70)

# ── Check files ───────────────────────────────────────────────────────────────
required = [
    ("Linear Regression",          LINEAR_PATH),
    ("Quadratic Regression",       QUADRATIC_PATH),
    ("ANN (PyTorch)",              ANN_PATH),
    ("IRI-2020 (FINAL_MASTER_IRI)", IRI_PATH),
]
missing = [n for n, p in required if not p.exists()]
for name, path in required:
    tag = "OK" if path.exists() else "XX"
    print(f"  [{tag}] {name}: {path.name}")
if missing:
    print("\nERROR: missing files:", missing)
    raise SystemExit(1)
print()

# ── Load ──────────────────────────────────────────────────────────────────────
lin_df  = pd.read_csv(LINEAR_PATH)
quad_df = pd.read_csv(QUADRATIC_PATH)
ann_df  = pd.read_csv(ANN_PATH)
iri_df  = pd.read_csv(IRI_PATH)

# Standardise observed column name in ML frames
for df in (lin_df, quad_df, ann_df):
    for alias in ("foF2_actual", "foF2"):
        if alias in df.columns and "foF2_obs" not in df.columns:
            df.rename(columns={alias: "foF2_obs"}, inplace=True)

# Standardise observed column in IRI frame
if "foF2" in iri_df.columns and "foF2_obs" not in iri_df.columns:
    iri_df.rename(columns={"foF2": "foF2_obs"}, inplace=True)

print(f"  Linear   : {len(lin_df):,} rows")
print(f"  Quadratic: {len(quad_df):,} rows")
print(f"  ANN      : {len(ann_df):,} rows")
print(f"  IRI      : {len(iri_df):,} rows  (CCIR + URSI)")
print()

# ── Merge all on (DayOfYear, Hour, Station) ───────────────────────────────────
KEYS = ["DayOfYear", "Hour", "Station"]

merged = (
    lin_df[KEYS + ["foF2_obs", "foF2_pred"]]
    .rename(columns={"foF2_pred": "pred_Linear"})
    .merge(
        quad_df[KEYS + ["foF2_pred"]].rename(columns={"foF2_pred": "pred_Quadratic"}),
        on=KEYS, how="inner"
    )
    .merge(
        ann_df[KEYS + ["foF2_pred"]].rename(columns={"foF2_pred": "pred_ANN"}),
        on=KEYS, how="inner"
    )
    .merge(
        iri_df[KEYS + ["foF2_C", "foF2_U"]].rename(
            columns={"foF2_C": "pred_IRI_CCIR", "foF2_U": "pred_IRI_URSI"}),
        on=KEYS, how="inner"
    )
)

print(f"  Merged dataset: {len(merged):,} rows (all models aligned on same observations)")
print()

# ── Metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(y_obs, y_pred):
    y_obs  = np.asarray(y_obs,  float)
    y_pred = np.asarray(y_pred, float)
    mask   = ~(np.isnan(y_obs) | np.isnan(y_pred))
    y_obs, y_pred = y_obs[mask], y_pred[mask]
    if len(y_obs) == 0:
        return None
    return {
        "MSE":  float(mean_squared_error(y_obs, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_obs, y_pred))),
        "MAE":  float(mean_absolute_error(y_obs, y_pred)),
        "R2":   float(r2_score(y_obs, y_pred)),
        "Bias": float(np.mean(y_pred - y_obs)),
        "N":    int(mask.sum()),
    }

# ── Model definitions ─────────────────────────────────────────────────────────
MODELS = [
    ("Linear",    "pred_Linear"),
    ("Quadratic", "pred_Quadratic"),
    ("ANN",       "pred_ANN"),
    ("IRI-CCIR",  "pred_IRI_CCIR"),
    ("IRI-URSI",  "pred_IRI_URSI"),
]

STATIONS = sorted(merged["Station"].unique())
COLORS   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS  = ["o", "s", "^", "D", "v"]
STATION_COLOR = dict(zip(STATIONS, COLORS))
STATION_MARKER = dict(zip(STATIONS, MARKERS))

# ── Scatter-plot helper ───────────────────────────────────────────────────────
def scatter_plot(df, obs_col, pred_col, title, save_path):
    """Single-panel scatter coloured by station with metrics annotation."""
    fig, ax = plt.subplots(figsize=(6, 5))

    all_obs, all_pred = [], []
    for s in STATIONS:
        sub = df[df["Station"] == s]
        if sub.empty:
            continue
        o = sub[obs_col].values
        p = sub[pred_col].values
        mask = ~(np.isnan(o) | np.isnan(p))
        o, p = o[mask], p[mask]
        ax.scatter(o, p, s=8, alpha=0.4,
                   color=STATION_COLOR[s], marker=STATION_MARKER[s], label=s)
        all_obs.append(o)
        all_pred.append(p)

    y_obs  = np.concatenate(all_obs)
    y_pred = np.concatenate(all_pred)
    m = compute_metrics(y_obs, y_pred)

    lims = [min(y_obs.min(), y_pred.min()) - 0.3,
            max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed foF2 (MHz)")
    ax.set_ylabel("Predicted foF2 (MHz)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, markerscale=1.5)

    txt = (f"RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}\n"
           f"R2={m['R2']:.4f}   Bias={m['Bias']:+.3f}\nN={m['N']:,}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")
    return m

# ── Individual scatter plots ──────────────────────────────────────────────────
print("=" * 70)
print("GENERATING SCATTER PLOTS")
print("=" * 70)

individual_metrics = {}
plot_specs = [
    ("Linear",    "pred_Linear",    "Linear Regression – All Data",    "scatter_linear.png"),
    ("Quadratic", "pred_Quadratic", "Quadratic Regression – All Data", "scatter_quadratic.png"),
    ("ANN",       "pred_ANN",       "ANN (PyTorch) – All Data",        "scatter_ann.png"),
    ("IRI-CCIR",  "pred_IRI_CCIR",  "IRI-2020 CCIR – All Data",        "scatter_iri_ccir.png"),
    ("IRI-URSI",  "pred_IRI_URSI",  "IRI-2020 URSI – All Data",        "scatter_iri_ursi.png"),
]

for name, pred_col, title, fname in plot_specs:
    m = scatter_plot(merged, "foF2_obs", pred_col, title, output_dir / fname)
    individual_metrics[name] = m

# ── Combined 5-panel figure ───────────────────────────────────────────────────
def _panel(ax, df, obs_col, pred_col, title):
    all_obs, all_pred = [], []
    for s in STATIONS:
        sub = df[df["Station"] == s]
        if sub.empty:
            continue
        o = sub[obs_col].values
        p = sub[pred_col].values
        mask = ~(np.isnan(o) | np.isnan(p))
        o, p = o[mask], p[mask]
        ax.scatter(o, p, s=5, alpha=0.3,
                   color=STATION_COLOR[s], marker=STATION_MARKER[s], label=s)
        all_obs.append(o); all_pred.append(p)

    y_obs  = np.concatenate(all_obs)
    y_pred = np.concatenate(all_pred)
    m = compute_metrics(y_obs, y_pred)

    lims = [min(y_obs.min(), y_pred.min()) - 0.3,
            max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=0.8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed (MHz)", fontsize=7)
    ax.set_ylabel("Predicted (MHz)", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)

    txt = f"R2={m['R2']:.3f}\nRMSE={m['RMSE']:.3f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

panel_specs = [
    ("pred_Linear",    "Linear Regression"),
    ("pred_Quadratic", "Quadratic Regression"),
    ("pred_ANN",       "ANN (PyTorch)"),
    ("pred_IRI_CCIR",  "IRI-2020 CCIR"),
    ("pred_IRI_URSI",  "IRI-2020 URSI"),
]
for ax, (pred_col, title) in zip(axes, panel_specs):
    _panel(ax, merged, "foF2_obs", pred_col, title)

# Station legend in the 6th panel
axes[5].axis("off")
handles = [plt.Line2D([0], [0], marker=STATION_MARKER[s], color="w",
                       markerfacecolor=STATION_COLOR[s], markersize=9, label=s)
           for s in STATIONS]
axes[5].legend(handles=handles, title="Station", loc="center",
               fontsize=10, title_fontsize=11)

fig.suptitle("foF2 Model Comparison – All Data (2019)", fontsize=13, y=1.01)
plt.tight_layout()
combined_path = output_dir / "scatter_all_models.png"
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {combined_path.name}")
print()

# ── Overall metrics table ─────────────────────────────────────────────────────
print("=" * 70)
print("OVERALL METRICS (ALL DATA)")
print("=" * 70)
print()

overall_rows = []
for name, pred_col in MODELS:
    m = compute_metrics(merged["foF2_obs"], merged[pred_col])
    if m:
        overall_rows.append({"Model": name, **m})
        print(f"{name:<14}  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"MAE={m['MAE']:.4f}  R2={m['R2']:.4f}  Bias={m['Bias']:+.4f}  N={m['N']}")

overall_df = pd.DataFrame(overall_rows)
print()
print(overall_df[["Model", "MSE", "RMSE", "MAE", "R2", "Bias"]].to_string(index=False))
print()

# ── Per-station metrics ───────────────────────────────────────────────────────
print("=" * 70)
print("PER-STATION METRICS (ALL DATA)")
print("=" * 70)

per_station_rows = []

for station in STATIONS:
    sub = merged[merged["Station"] == station]
    print(f"\n--- {station} ---")
    for name, pred_col in MODELS:
        m = compute_metrics(sub["foF2_obs"], sub[pred_col])
        if m:
            print(f"  {name:<14} RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}  "
                  f"MAE={m['MAE']:.4f}  Bias={m['Bias']:+.4f}  N={m['N']}")
            per_station_rows.append({"Station": station, "Model": name, **m})

per_station_df = pd.DataFrame(per_station_rows)
print()

# ── Save CSVs ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("SAVING TABLES")
print("=" * 70)

overall_path = output_dir / "model_comparison.csv"
overall_df[["Model", "MSE", "RMSE", "MAE", "R2", "Bias", "N"]].round(6).to_csv(
    overall_path, index=False)
print(f"  Saved: {overall_path.name}")

per_station_path = output_dir / "comparison_per_station.csv"
per_station_df[["Station", "Model", "MSE", "RMSE", "MAE", "R2", "Bias", "N"]].round(6).to_csv(
    per_station_path, index=False)
print(f"  Saved: {per_station_path.name}")

print()
print("=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)
