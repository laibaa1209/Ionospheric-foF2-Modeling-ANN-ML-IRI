"""
Station map — global overview + Americas zoom-in.
==================================================
Produces station_map.png in the project root.

Layout
------
Left  : World map (PlateCarree) with all 5 ionosonde stations and a
        dashed rectangle indicating the Americas zoom region.
Right : Zoomed Americas view with station details (name, lat, lon,
        station code) and lat/lon gridlines.

Usage:
    python station_map.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path

# ── Station data ──────────────────────────────────────────────────────────────
STATIONS = [
    {
        "code":  "CP",
        "name":  "Cachoeira Paulista",
        "lat":   -22.70,
        "lon":   -45.00,   # 315.00° E → -45° (west)
        "color": "#1f77b4",
        "marker":"o",
        "label_offset": (4, -6),    # (dx, dy) in points for zoomed label
    },
    {
        "code":  "EA",
        "name":  "Eglin AFB",
        "lat":    30.50,
        "lon":   -86.50,   # 273.50° E
        "color": "#ff7f0e",
        "marker":"s",
        "label_offset": (-6, 5),
    },
    {
        "code":  "JI",
        "name":  "Jicamarca",
        "lat":   -12.00,
        "lon":   -76.80,   # 283.20° E
        "color": "#2ca02c",
        "marker":"^",
        "label_offset": (-8, -8),
    },
    {
        "code":  "MH",
        "name":  "Millstone Hill",
        "lat":    42.60,
        "lon":   -71.50,   # 288.50° E
        "color": "#d62728",
        "marker":"D",
        "label_offset": (4, 4),
    },
    {
        "code":  "RA",
        "name":  "Ramey",
        "lat":    18.50,
        "lon":   -67.10,   # 292.90° E
        "color": "#9467bd",
        "marker":"v",
        "label_offset": (4, -7),
    },
]

# Americas bounding box for zoom panel
ZOOM_EXTENT = [-115, -40, -45, 55]   # [lon_min, lon_max, lat_min, lat_max]

PROJ   = ccrs.PlateCarree()
OUTPUT = Path(__file__).resolve().parent / "station_map.png"

# ── Helper: draw stations ──────────────────────────────────────────────────────
def plot_stations(ax, s, zorder=6, ms=70, annotate=False, fontsize=8):
    ax.scatter(
        s["lon"], s["lat"],
        s=ms, color=s["color"], marker=s["marker"],
        edgecolors="k", linewidths=0.6,
        transform=PROJ, zorder=zorder,
    )
    if annotate:
        dx, dy = s["label_offset"]
        label  = (f"{s['code']} — {s['name']}\n"
                  f"Lat {s['lat']:+.2f}°   Lon {s['lon']:+.2f}°")
        ax.annotate(
            label,
            xy=(s["lon"], s["lat"]),
            xycoords=ax.transData,
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=fontsize,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec=s["color"], lw=0.8),
            arrowprops=dict(arrowstyle="-", color=s["color"], lw=0.7),
            transform=PROJ,
            zorder=7,
        )


# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor("#f7f9fc")

ax_world = fig.add_axes([0.02, 0.05, 0.54, 0.90], projection=PROJ)
ax_zoom  = fig.add_axes([0.57, 0.05, 0.41, 0.90], projection=PROJ)

# ── World map ─────────────────────────────────────────────────────────────────
ax_world.set_global()
ax_world.set_facecolor("#d6eaf8")   # ocean

ax_world.add_feature(cfeature.NaturalEarthFeature(
    "physical", "land", "110m",
    facecolor="#eaecee", edgecolor="none"))
ax_world.add_feature(cfeature.NaturalEarthFeature(
    "cultural", "admin_0_countries", "110m",
    facecolor="none", edgecolor="#aab0b8", linewidth=0.4))
ax_world.add_feature(cfeature.NaturalEarthFeature(
    "physical", "coastline", "110m",
    facecolor="none", edgecolor="#5d6d7e", linewidth=0.5))
for stn in STATIONS:
    plot_stations(ax_world, stn, ms=55, annotate=False)

# Dashed rectangle marking zoom region
lon0, lon1, lat0, lat1 = ZOOM_EXTENT
rect_lons = [lon0, lon1, lon1, lon0, lon0]
rect_lats = [lat0, lat0, lat1, lat1, lat0]
ax_world.plot(rect_lons, rect_lats, "r--", lw=1.5, transform=PROJ, zorder=5)
ax_world.text(lon0 + 1, lat1 + 1, "Zoomed\nregion →",
              color="red", fontsize=7, transform=PROJ, zorder=6)

# Legend
legend_handles = [
    mpatches.Patch(
        facecolor=s["color"], edgecolor="k", linewidth=0.5,
        label=f"{s['code']} — {s['name']}",
    )
    for s in STATIONS
]
ax_world.legend(handles=legend_handles, loc="lower left",
                fontsize=7.5, framealpha=0.9, edgecolor="#aab0b8")

ax_world.set_title("Global Overview — Ionosonde Stations", fontsize=11, pad=6)
ax_world.gridlines(draw_labels=False, linewidth=0.3, linestyle="--", alpha=0.5)

# ── Americas zoom panel ───────────────────────────────────────────────────────
ax_zoom.set_extent(ZOOM_EXTENT, crs=PROJ)
ax_zoom.set_facecolor("#d6eaf8")

ax_zoom.add_feature(cfeature.NaturalEarthFeature(
    "physical", "land", "50m",
    facecolor="#eaecee", edgecolor="none"))
ax_zoom.add_feature(cfeature.NaturalEarthFeature(
    "cultural", "admin_0_countries", "50m",
    facecolor="none", edgecolor="#7f8c8d", linewidth=0.5))
ax_zoom.add_feature(cfeature.NaturalEarthFeature(
    "cultural", "admin_1_states_provinces_lines", "50m",
    facecolor="none", edgecolor="#bdc3c7", linewidth=0.3))
ax_zoom.add_feature(cfeature.NaturalEarthFeature(
    "physical", "coastline", "50m",
    facecolor="none", edgecolor="#2e4057", linewidth=0.7))
ax_zoom.add_feature(cfeature.NaturalEarthFeature(
    "physical", "rivers_lake_centerlines", "50m",
    facecolor="none", edgecolor="#aed6f1", linewidth=0.4))
ax_zoom.add_feature(cfeature.NaturalEarthFeature(
    "physical", "lakes", "50m",
    facecolor="#d6eaf8", edgecolor="#aed6f1", linewidth=0.4))

for stn in STATIONS:
    plot_stations(ax_zoom, stn, ms=90, annotate=True, fontsize=7.5)

# Gridlines with labels
gl = ax_zoom.gridlines(
    crs=PROJ, draw_labels=True,
    linewidth=0.5, linestyle="--", color="#7f8c8d", alpha=0.6,
)
gl.top_labels    = False
gl.right_labels  = False
gl.xlocator  = mticker.FixedLocator(range(-110, -30, 10))
gl.ylocator  = mticker.FixedLocator(range(-40, 60, 10))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {"size": 7, "color": "#2c3e50"}
gl.ylabel_style = {"size": 7, "color": "#2c3e50"}

ax_zoom.set_title("American Longitudinal Sector — Station Details", fontsize=11, pad=6)

# ── Shared title ──────────────────────────────────────────────────────────────
fig.text(
    0.50, 0.985,
    "Ionosonde Stations Used in foF2 Modeling Study (2019 Dataset)",
    ha="center", va="top", fontsize=13, fontweight="bold", color="#1a252f",
)

plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUTPUT}")
