"""
Station map v2 — world map with Americas inset overlaid on eastern hemisphere.
==============================================================================
Produces station_map_v2.png in the project root.

Layout
------
One full-figure world map (Robinson projection, dark theme).
The Americas zoom inset is overlaid on the right (eastern hemisphere) side
of the globe. Two connector lines link the zoom rectangle to the inset
corners for visual clarity.

Usage:
    python station_map_v2.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path

# ── Colour palette (dark theme) ───────────────────────────────────────────────
BG_COLOR     = "#0d1117"
OCEAN_COLOR  = "#0e2238"
LAND_COLOR   = "#1c2e22"
BORDER_COLOR = "#2e4a3a"
COAST_COLOR  = "#4a7a5a"
GRID_COLOR   = "#1e3050"
ACCENT       = "#e8c84a"      # gold accent for titles / borders

# ── Station data ──────────────────────────────────────────────────────────────
STATIONS = [
    {"code": "CP",  "name": "Cachoeira Paulista", "lat": -22.70, "lon": -45.00,
     "color": "#4fc3f7", "marker": "o",
     "label_offset": (10, -12)},
    {"code": "EA",  "name": "Eglin AFB",           "lat":  30.50, "lon": -86.50,
     "color": "#ffb74d", "marker": "s",
     "label_offset": (-110, 8)},
    {"code": "JI",  "name": "Jicamarca",           "lat": -12.00, "lon": -76.80,
     "color": "#a5d6a7", "marker": "^",
     "label_offset": (-115, -14)},
    {"code": "MH",  "name": "Millstone Hill",      "lat":  42.60, "lon": -71.50,
     "color": "#ef9a9a", "marker": "D",
     "label_offset": (10, 6)},
    {"code": "RA",  "name": "Ramey",               "lat":  18.50, "lon": -67.10,
     "color": "#ce93d8", "marker": "v",
     "label_offset": (10, -14)},
]

ZOOM_EXTENT = [-115, -40, -48, 58]   # [lon_min, lon_max, lat_min, lat_max]
PROJ        = ccrs.PlateCarree()
ROB         = ccrs.Robinson()
OUTPUT      = Path(__file__).resolve().parent / "station_map_v2.png"


# ── Helpers ───────────────────────────────────────────────────────────────────
def glow_scatter(ax, lon, lat, color, marker, ms_outer=260, ms_inner=90,
                 alpha_outer=0.25, zorder=8):
    """Draw a glowing station marker: large translucent halo + sharp dot."""
    kw = dict(transform=PROJ, zorder=zorder, linewidths=0)
    ax.scatter(lon, lat, s=ms_outer * 2, color=color, marker="o",
               alpha=alpha_outer, **kw)
    ax.scatter(lon, lat, s=ms_outer,     color=color, marker="o",
               alpha=alpha_outer * 0.5, **kw)
    ax.scatter(lon, lat, s=ms_inner, color=color, marker=marker,
               edgecolors="white", linewidths=0.7, alpha=0.95,
               zorder=zorder + 1, transform=PROJ)


def annotate_station_inset(ax, stn):
    dx, dy = stn["label_offset"]
    label  = (f"  {stn['code']} — {stn['name']}\n"
              f"  {stn['lat']:+.2f}°   {stn['lon']:+.2f}°")
    ax.annotate(
        label,
        xy=(stn["lon"], stn["lat"]),
        xycoords=ax.transData,
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=7.5,
        color="white",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117",
                  alpha=0.85, ec=stn["color"], lw=1.0),
        arrowprops=dict(arrowstyle="-", color=stn["color"],
                        lw=0.8, alpha=0.9),
        transform=PROJ,
        zorder=12,
    )


def add_map_features(ax, res="110m", borders=True, states=False,
                     rivers=False, lakes=False):
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", res,
        facecolor=LAND_COLOR, edgecolor="none"))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", res,
        facecolor="none", edgecolor=COAST_COLOR, linewidth=0.6))
    if borders:
        ax.add_feature(cfeature.NaturalEarthFeature(
            "cultural", "admin_0_countries", res,
            facecolor="none", edgecolor=BORDER_COLOR, linewidth=0.35))
    if states:
        ax.add_feature(cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "50m",
            facecolor="none", edgecolor="#243d30", linewidth=0.25))
    if rivers:
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "rivers_lake_centerlines", "50m",
            facecolor="none", edgecolor="#163550", linewidth=0.4))
    if lakes:
        ax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "lakes", "50m",
            facecolor=OCEAN_COLOR, edgecolor="#163550", linewidth=0.3))


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor(BG_COLOR)

# ── World map (full figure, Robinson projection) ───────────────────────────────
ax_world = fig.add_axes([0.0, 0.02, 1.0, 0.93], projection=ROB)
ax_world.set_global()
ax_world.set_facecolor(OCEAN_COLOR)
ax_world.patch.set_facecolor(OCEAN_COLOR)

add_map_features(ax_world, res="110m", borders=True)

# Subtle graticule
gl_w = ax_world.gridlines(crs=PROJ, linewidth=0.25, linestyle="--",
                           color=GRID_COLOR, alpha=0.7, draw_labels=False)

# Station glows on world map
for stn in STATIONS:
    glow_scatter(ax_world, stn["lon"], stn["lat"],
                 stn["color"], stn["marker"], ms_outer=200, ms_inner=70)

# ── Zoom rectangle on world map ───────────────────────────────────────────────
# Draw the box in Robinson-projected coordinates
lon0, lon1, lat0, lat1 = ZOOM_EXTENT
box_lons = [lon0, lon1, lon1, lon0, lon0]
box_lats = [lat0, lat0, lat1, lat1, lat0]
ax_world.plot(box_lons, box_lats,
              transform=PROJ, color=ACCENT, lw=1.8,
              linestyle="-", zorder=6,
              path_effects=[pe.Stroke(linewidth=3.5, foreground=BG_COLOR, alpha=0.6),
                            pe.Normal()])

# ── Inset axes — Americas zoom, overlaid on eastern hemisphere ────────────────
# Placed in figure coords over the right (eastern-hemisphere) side
ax_inset = fig.add_axes([0.55, 0.06, 0.40, 0.84], projection=PROJ)
ax_inset.set_extent(ZOOM_EXTENT, crs=PROJ)

add_map_features(ax_inset, res="50m", borders=True, states=True,
                 rivers=True, lakes=True)

# Inset station glows + annotations
for stn in STATIONS:
    glow_scatter(ax_inset, stn["lon"], stn["lat"],
                 stn["color"], stn["marker"], ms_outer=260, ms_inner=90)
    annotate_station_inset(ax_inset, stn)

# Gridlines with labels
gl_i = ax_inset.gridlines(
    crs=PROJ, draw_labels=True,
    linewidth=0.4, linestyle=":", color="#2a5a3a", alpha=0.8,
)
gl_i.top_labels   = False
gl_i.right_labels = False
gl_i.xlocator  = mticker.FixedLocator(range(-110, -35, 15))
gl_i.ylocator  = mticker.FixedLocator(range(-40, 65, 15))
gl_i.xformatter = LONGITUDE_FORMATTER
gl_i.yformatter = LATITUDE_FORMATTER
gl_i.xlabel_style = {"size": 7, "color": "#aabbaa"}
gl_i.ylabel_style = {"size": 7, "color": "#aabbaa"}

# Fancy inset border
for spine in ax_inset.spines.values():
    spine.set_edgecolor(ACCENT)
    spine.set_linewidth(2.0)

# ── Connector lines: zoom box corners → inset corners ────────────────────────
# We connect the top-right corner of the zoom box (world map)
# to the top-left corner of the inset, and similarly bottom-right→bottom-left.
# Use ConnectionPatch with data coords on world map and axes coords on inset.
for (lat_c, inset_y) in [(lat1, 1.0), (lat0, 0.0)]:
    con = ConnectionPatch(
        xyA=(lon1, lat_c), coordsA=ax_world.transData,
        xyB=(0.0, inset_y), coordsB=ax_inset.transAxes,
        color=ACCENT, lw=1.0, linestyle="--", alpha=0.65,
        zorder=5,
    )
    fig.add_artist(con)

# ── Inset title label ─────────────────────────────────────────────────────────
ax_inset.set_title("American Longitudinal Sector",
                   fontsize=10, color=ACCENT, pad=5,
                   fontweight="bold",
                   path_effects=[pe.Stroke(linewidth=2, foreground=BG_COLOR),
                                 pe.Normal()])

# ── Legend (on world map) ─────────────────────────────────────────────────────
legend_handles = []
for stn in STATIONS:
    h = plt.Line2D(
        [0], [0],
        marker=stn["marker"], color="none",
        markerfacecolor=stn["color"], markeredgecolor="white",
        markeredgewidth=0.5, markersize=8,
        label=f"{stn['code']}  {stn['name']}  ({stn['lat']:+.1f}°, {stn['lon']:+.1f}°)",
    )
    legend_handles.append(h)

leg = ax_world.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=8,
    framealpha=0.85,
    facecolor="#0d1a10",
    edgecolor=ACCENT,
    labelcolor="white",
    title="Ionosonde Stations",
    title_fontsize=8.5,
)
leg.get_title().set_color(ACCENT)

# ── Figure title ──────────────────────────────────────────────────────────────
fig.text(
    0.28, 0.975,
    "Ionosonde Stations — foF2 Modeling Study  (2019, American Sector)",
    ha="center", va="top",
    fontsize=13, fontweight="bold",
    color=ACCENT,
    path_effects=[pe.Stroke(linewidth=2, foreground=BG_COLOR, alpha=0.8),
                  pe.Normal()],
)

plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"Saved: {OUTPUT}")
