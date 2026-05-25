"""
Pseudo-color foF2 comparison plots — all scenarios, all stations.
==================================================================
For each scenario and each station produces a 7-panel figure:

  Row 1 (values):   [Observed] [ANN] [IRI-CCIR] [IRI-URSI]
  Row 2 (residuals):  [empty]  [Δ ANN] [Δ IRI-CCIR] [Δ IRI-URSI]

X-axis : local time hour (0–23)
Y-axis : month (Jan–Dec)
Colour : median foF2 per (month, local hour) cell, smoothed with a
         Gaussian kernel (sigma=0.8).

Local-time conversion (fixed standard-time UTC offsets, no DST):
  CP          UTC −3    ElginAB     UTC −6
  Jicamarca   UTC −5    MilstonHill UTC −5    Ramey UTC −4

Scenarios
  1 — Full dataset       data/scenario1_full/
  2 — Quiet days         data/scenario2_no_quiet/
  3 — Disturbed days     data/scenario3_no_disturbed/
  4 — Midday/Midnight    data/scenario4_no_midday_midnight/
  5 — Equinox/Solstice   data/scenario5_equinox_solstice/

ANN CSV columns   : Station, DayOfYear, Hour, foF2_obs, foF2_pred
IRI CSV columns   : Station, DayOfYear, Hour, foF2, foF2_C, foF2_U

Hour column is UTC for Scenarios 1, 2, 3, 5.
Hour column is standard local time for Scenario 4.

Outputs land in  plots/outputs/scenarioN_*/  one PNG per station.

Usage:
    python make_pseudocolor_plots.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent          # plots/python_scripts/
plots_root = script_dir.parent                        # plots/
data_root  = plots_root / "data"
out_root   = plots_root / "outputs"

# ── Station config ────────────────────────────────────────────────────────────
STATIONS = ["CP", "ElginAB", "Jicamarca", "MilstonHill", "Ramey"]

STATION_UTC_OFFSET = {
    "CP":          -3,
    "ElginAB":     -6,
    "Jicamarca":   -5,
    "MilstonHill": -5,
    "Ramey":       -4,
}

STATION_LABEL = {
    "CP":          "Cachoeira Paulista",
    "ElginAB":     "Eglin AFB",
    "Jicamarca":   "Jicamarca",
    "MilstonHill": "Millstone Hill",
    "Ramey":       "Ramey",
}

# ── Scenario config ───────────────────────────────────────────────────────────
# ann_file / iri_file: filenames inside data/scenarioN/ folder
# hour_utc: True → Hour column is UTC, convert to LT; False → already LT
SCENARIOS = {
    "scenario1_full": {
        "title":    "Scenario 1 — Full Dataset",
        "ann_file": "predictions_ann_full.csv",
        "iri_file": "full_dataset_IRI.csv",
        "hour_utc": True,
        "ann_obs_col": "foF2_obs",
        "ann_pred_col": "foF2_pred",
    },
    "scenario2_no_quiet": {
        "title":    "Scenario 2 — Quiet Days",
        "ann_file": "predictions_ann_no_quiet.csv",
        "iri_file": "quiet_days_only_IRI.csv",
        "hour_utc": True,
        "ann_obs_col": "foF2_obs",
        "ann_pred_col": "foF2_pred",
    },
    "scenario3_no_disturbed": {
        "title":    "Scenario 3 — Disturbed Days",
        "ann_file": "predictions_ann_no_disturbed.csv",
        "iri_file": "disturbed_days_only_IRI.csv",
        "hour_utc": True,
        "ann_obs_col": "foF2_obs",
        "ann_pred_col": "foF2_pred",
    },
    "scenario4_no_midday_midnight": {
        "title":    "Scenario 4 — Midday & Midnight",
        "ann_file": "predictions_ann_no_midday_midnight.csv",
        "iri_file": "midday_midnight_only_IRI.csv",
        "hour_utc": False,   # Hour is already standard local time in both CSVs
        "ann_obs_col": "foF2_obs",
        "ann_pred_col": "foF2_pred",
    },
    "scenario5_equinox_solstice": {
        "title":    "Scenario 5 — Equinox & Solstice",
        "ann_file": "predictions_ann_equinox_solstice.csv",
        "iri_file": "equinox_solstice_IRI.csv",
        "hour_utc": True,
        "ann_obs_col": "foF2_obs",
        "ann_pred_col": "foF2_pred",
    },
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
SIGMA = 0.8   # Gaussian smoothing sigma (grid cells)


# ── Helpers ───────────────────────────────────────────────────────────────────
def doy_to_month(doy_series: pd.Series) -> pd.Series:
    """Convert day-of-year (2019) to month number 1–12."""
    base = date(2019, 1, 1).toordinal()
    return doy_series.apply(lambda d: date.fromordinal(base + int(d) - 1).month)


def utc_to_lt(hour: int, station: str) -> int:
    return int((hour + STATION_UTC_OFFSET.get(station, 0)) % 24)


def make_grid(df_stn: pd.DataFrame, col: str) -> np.ndarray:
    """Build a (12 × 24) grid of median values (Month × LocalHour)."""
    grid = np.full((12, 24), np.nan)
    grp  = df_stn.groupby(["Month", "LT"])[col].median()
    for (m, h), val in grp.items():
        if 1 <= m <= 12 and 0 <= h <= 23:
            grid[m - 1, h] = val
    return grid


def smooth(grid: np.ndarray) -> np.ndarray:
    """Gaussian smoothing that respects NaN boundaries."""
    mask   = np.isnan(grid)
    filled = np.where(mask, 0.0, grid)
    sm     = gaussian_filter(filled, sigma=SIGMA)
    wt     = gaussian_filter((~mask).astype(float), sigma=SIGMA)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(wt > 0.01, sm / wt, np.nan)
    result[mask & (wt < 0.01)] = np.nan
    return result


def add_panel(ax, grid, vmin, vmax, cmap, title, cbar_label):
    """Draw one pseudo-color panel with pcolormesh."""
    sm  = smooth(grid)
    x   = np.arange(-0.5, 24)          # 25 edges for 24 hour bins
    y   = np.arange(0.5, 13)           # 13 edges for 12 month bins
    pcm = ax.pcolormesh(x, y, sm, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0.5, 12.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels([0, 6, 12, 18, 23], fontsize=7)
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(MONTH_LABELS, fontsize=7)
    ax.set_title(title, fontsize=8, pad=3)
    cb = plt.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return pcm


# ── Main loop ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("PSEUDO-COLOR foF2 COMPARISON PLOTS")
print("=" * 65)

for scen_key, scen_cfg in SCENARIOS.items():
    data_dir = data_root / scen_key
    out_dir  = out_root  / scen_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{scen_cfg['title']}  (hour_utc={scen_cfg['hour_utc']})")

    # ── Load and align ────────────────────────────────────────────────────────
    ann_raw = pd.read_csv(data_dir / scen_cfg["ann_file"])
    iri_raw = pd.read_csv(data_dir / scen_cfg["iri_file"])

    ann_df = ann_raw.rename(columns={
        scen_cfg["ann_obs_col"]:  "foF2_obs",
        scen_cfg["ann_pred_col"]: "foF2_ann",
    })[["Station", "DayOfYear", "Hour", "foF2_obs", "foF2_ann"]]

    iri_df = iri_raw.rename(columns={
        "foF2":   "foF2_obs_iri",
        "foF2_C": "foF2_ccir",
        "foF2_U": "foF2_ursi",
    })[["Station", "DayOfYear", "Hour", "foF2_obs_iri", "foF2_ccir", "foF2_ursi"]]

    df = pd.merge(ann_df, iri_df, on=["Station", "DayOfYear", "Hour"], how="inner")
    print(f"  Merged rows: {len(df):,}")

    # ── Local time ────────────────────────────────────────────────────────────
    if scen_cfg["hour_utc"]:
        df["LT"] = df.apply(
            lambda r: utc_to_lt(int(r["Hour"]), r["Station"]), axis=1
        )
        print(f"  UTC->LT conversion applied")
    else:
        df["LT"] = df["Hour"].astype(int)
        print(f"  Hour already in local time — no conversion")

    # ── Month ─────────────────────────────────────────────────────────────────
    df["Month"] = doy_to_month(df["DayOfYear"])

    # ── Per-station plots ─────────────────────────────────────────────────────
    for stn in STATIONS:
        df_s = df[df["Station"] == stn].copy()
        if df_s.empty:
            print(f"  {stn}: no data — skipping")
            continue

        g_obs   = make_grid(df_s, "foF2_obs")
        g_ann   = make_grid(df_s, "foF2_ann")
        g_ccir  = make_grid(df_s, "foF2_ccir")
        g_ursi  = make_grid(df_s, "foF2_ursi")
        g_dann  = g_ann  - g_obs
        g_dccir = g_ccir - g_obs
        g_dursi = g_ursi - g_obs

        # Shared colour limits across all four value panels
        all_vals = np.concatenate([
            g[~np.isnan(g)] for g in [g_obs, g_ann, g_ccir, g_ursi]
        ])
        vmin_val = float(np.nanpercentile(all_vals, 2))
        vmax_val = float(np.nanpercentile(all_vals, 98))

        all_deltas = np.concatenate([
            g[~np.isnan(g)] for g in [g_dann, g_dccir, g_dursi]
        ])
        delta_abs = max(float(np.nanpercentile(np.abs(all_deltas), 95)), 0.1)

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(24, 11))
        fig.suptitle(
            f"{scen_cfg['title']} — {STATION_LABEL[stn]} ({stn})   "
            f"[X: Local Time (h)  |  Y: Month]",
            fontsize=12, y=0.98,
        )

        gs = gridspec.GridSpec(2, 4, figure=fig,
                               left=0.05, right=0.97,
                               top=0.91, bottom=0.07,
                               hspace=0.32, wspace=0.30)
        ax_obs   = fig.add_subplot(gs[0, 0])
        ax_ann   = fig.add_subplot(gs[0, 1])
        ax_ccir  = fig.add_subplot(gs[0, 2])
        ax_ursi  = fig.add_subplot(gs[0, 3])
        # gs[1, 0] is intentionally left blank (no delta for observed)
        ax_dann  = fig.add_subplot(gs[1, 1])
        ax_dccir = fig.add_subplot(gs[1, 2])
        ax_dursi = fig.add_subplot(gs[1, 3])

        add_panel(ax_obs,   g_obs,   vmin_val,   vmax_val,   "jet",    "Observed foF2",          "MHz")
        add_panel(ax_ann,   g_ann,   vmin_val,   vmax_val,   "jet",    "ANN Predicted",           "MHz")
        add_panel(ax_ccir,  g_ccir,  vmin_val,   vmax_val,   "jet",    "IRI-CCIR",                "MHz")
        add_panel(ax_ursi,  g_ursi,  vmin_val,   vmax_val,   "jet",    "IRI-URSI",                "MHz")
        add_panel(ax_dann,  g_dann,  -delta_abs, delta_abs,  "RdBu_r", "D ANN (pred-obs)",        "dMHz")
        add_panel(ax_dccir, g_dccir, -delta_abs, delta_abs,  "RdBu_r", "D IRI-CCIR (pred-obs)",   "dMHz")
        add_panel(ax_dursi, g_dursi, -delta_abs, delta_abs,  "RdBu_r", "D IRI-URSI (pred-obs)",   "dMHz")

        for ax in [ax_obs, ax_ann, ax_ccir, ax_ursi, ax_dann, ax_dccir, ax_dursi]:
            ax.set_xlabel("Local Time (h)", fontsize=8)
        for ax in [ax_obs, ax_dann]:
            ax.set_ylabel("Month", fontsize=8)

        out_path = out_dir / f"{stn}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {stn}: saved {out_path.name}")

print("\nDone.")
