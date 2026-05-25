"""
IRI-2020 Scatter Plots — all evaluation scenarios.
===================================================
Produces scatter plots (observed vs IRI-CCIR and observed vs IRI-URSI) for:
  Scenario 2 — Quiet days          (quiet_days_only_IRI.csv)
  Scenario 3 — Disturbed days      (disturbed_days_only_IRI.csv)
  Scenario 4 — Midday/Midnight     (midday_midnight_only_IRI.csv)
  Scenario 5 — Equinox/Solstice    (equinox_solstice_IRI.csv)

Column key in every IRI CSV:
  foF2    — observed foF2 (MHz)
  foF2_C  — IRI-CCIR prediction
  foF2_U  — IRI-URSI prediction
  Hour    — UTC for Scenarios 2,3,5 | standard local time for Scenario 4

Saved to iri_evals/outputs/scenarioN_*/

Usage:
    python make_iri_scatter.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent          # iri_evals/python_scripts/
iri_evals  = script_dir.parent                        # iri_evals/

DATA = {
    "scenario2_no_quiet": {
        "csv":   iri_evals / "quiet_days_only_IRI.csv",
        "title": "Scenario 2 — Quiet Days",
        "tag":   "no_quiet",
    },
    "scenario3_no_disturbed": {
        "csv":   iri_evals / "disturbed_days_only_IRI.csv",
        "title": "Scenario 3 — Disturbed Days",
        "tag":   "no_disturbed",
    },
    "scenario4_no_midday_midnight": {
        "csv":   iri_evals / "midday_midnight_only_IRI.csv",
        "title": "Scenario 4 — Midday & Midnight",
        "tag":   "no_midday_midnight",
        "midday_midnight": True,
    },
    "scenario5_equinox_solstice": {
        "csv":   iri_evals / "equinox_solstice_IRI.csv",
        "title": "Scenario 5 — Equinox & Solstice",
        "tag":   "equinox_solstice",
    },
}

STATIONS       = ["CP", "ElginAB", "Jicamarca", "MilstonHill", "Ramey"]
STATION_COLOR  = dict(zip(STATIONS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
STATION_MARKER = dict(zip(STATIONS, ["o", "s", "^", "D", "v"]))

MIDDAY_HOURS   = {11, 12, 13}
MIDNIGHT_HOURS = {23, 0, 1}

print("=" * 65)
print("IRI-2020 SCATTER PLOTS")
print("=" * 65)

# ── Scatter helper ────────────────────────────────────────────────────────────
def scatter_plot(y_obs, y_pred, stations_arr, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for s in STATIONS:
        mask = stations_arr == s
        if not mask.any():
            continue
        ax.scatter(y_obs[mask], y_pred[mask], s=8, alpha=0.4,
                   color=STATION_COLOR[s], marker=STATION_MARKER[s], label=s)
    lims = [min(y_obs.min(), y_pred.min()) - 0.3,
            max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed foF2 (MHz)")
    ax.set_ylabel("Predicted foF2 (MHz)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, markerscale=1.5)
    rmse = float(np.sqrt(mean_squared_error(y_obs, y_pred)))
    mae  = float(mean_absolute_error(y_obs, y_pred))
    r2   = float(r2_score(y_obs, y_pred))
    bias = float(np.mean(y_pred - y_obs))
    txt  = (f"RMSE={rmse:.3f}  MAE={mae:.3f}\n"
            f"R²={r2:.4f}   Bias={bias:+.3f}\nN={len(y_obs):,}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


def combined_scatter_s4(df, model_col, model_label, title, save_path):
    """Combined midday+midnight scatter with filled=midday, hollow=midnight."""
    y_obs  = df["foF2"].values
    y_pred = df[model_col].values
    st     = df["Station"].values
    mm     = df["Hour"].isin(MIDDAY_HOURS).values
    mn     = df["Hour"].isin(MIDNIGHT_HOURS).values

    fig, ax = plt.subplots(figsize=(6, 5))
    for s in STATIONS:
        mask = st == s
        if not mask.any():
            continue
        if (mask & mm).any():
            ax.scatter(y_obs[mask & mm], y_pred[mask & mm], s=8, alpha=0.5,
                       color=STATION_COLOR[s], marker=STATION_MARKER[s],
                       label=f"{s} midday")
        if (mask & mn).any():
            ax.scatter(y_obs[mask & mn], y_pred[mask & mn], s=8, alpha=0.5,
                       color=STATION_COLOR[s], marker=STATION_MARKER[s],
                       facecolors="none", edgecolors=STATION_COLOR[s],
                       linewidths=0.8, label=f"{s} midnight")

    lims = [min(y_obs.min(), y_pred.min()) - 0.3,
            max(y_obs.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed foF2 (MHz)")
    ax.set_ylabel(f"{model_label} foF2 (MHz)")
    ax.set_title(title + "\n(filled = midday, hollow = midnight)", fontsize=10)
    ax.legend(fontsize=6, markerscale=1.8, ncol=2)

    rmse = float(np.sqrt(mean_squared_error(y_obs, y_pred)))
    mae  = float(mean_absolute_error(y_obs, y_pred))
    r2   = float(r2_score(y_obs, y_pred))
    bias = float(np.mean(y_pred - y_obs))
    txt  = (f"RMSE={rmse:.3f}  MAE={mae:.3f}\n"
            f"R²={r2:.4f}   Bias={bias:+.3f}\nN={len(y_obs):,}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")


# ── Process each scenario ─────────────────────────────────────────────────────
for folder, cfg in DATA.items():
    print(f"\n{cfg['title']}")
    out_dir = iri_evals / "outputs" / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    df  = pd.read_csv(cfg["csv"])
    obs = df["foF2"].values
    st  = df["Station"].values

    is_s4 = cfg.get("midday_midnight", False)

    for model_col, model_label, short in [
        ("foF2_C", "IRI-CCIR", "ccir"),
        ("foF2_U", "IRI-URSI", "ursi"),
    ]:
        pred = df[model_col].values
        tag  = cfg["tag"]

        if is_s4:
            # Combined scatter
            combined_scatter_s4(
                df, model_col, model_label,
                f"{cfg['title']} — {model_label}",
                out_dir / f"iri_{short}_{tag}_scatter.png",
            )
            # Midday only
            mask_mid = df["Hour"].isin(MIDDAY_HOURS).values
            scatter_plot(
                obs[mask_mid], pred[mask_mid], st[mask_mid],
                f"{cfg['title']} — {model_label} (Midday, std LT 11–13)",
                out_dir / f"iri_{short}_midday_scatter.png",
            )
            # Midnight only
            mask_mn = df["Hour"].isin(MIDNIGHT_HOURS).values
            scatter_plot(
                obs[mask_mn], pred[mask_mn], st[mask_mn],
                f"{cfg['title']} — {model_label} (Midnight, std LT 23, 0, 1)",
                out_dir / f"iri_{short}_midnight_scatter.png",
            )
        else:
            scatter_plot(
                obs, pred, st,
                f"{cfg['title']} — {model_label}",
                out_dir / f"iri_{short}_{tag}_scatter.png",
            )

        # Print per-station metrics
        mse = mean_squared_error(obs, pred)
        print(f"  {model_label}  RMSE={np.sqrt(mse):.4f}  "
              f"MAE={mean_absolute_error(obs, pred):.4f}  "
              f"R²={r2_score(obs, pred):.4f}  "
              f"Bias={np.mean(pred-obs):+.4f}  N={len(obs):,}")
        for s in STATIONS:
            m = st == s
            if not m.any():
                continue
            print(f"    {s:<14} RMSE={np.sqrt(mean_squared_error(obs[m],pred[m])):.4f}  "
                  f"R²={r2_score(obs[m],pred[m]):.4f}  "
                  f"Bias={np.mean(pred[m]-obs[m]):+.4f}  N={m.sum()}")

print("\nDone.")
