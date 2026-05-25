"""
Prepare filtered datasets for ANN evaluation scenarios.
=======================================================
Reads quiet_and_disturbed.txt (NOAA format) and FINAL_MASTER.csv to produce:

  master_no_quiet.csv              — all rows EXCEPT quiet days          (Hour = UTC)
  master_no_disturbed.csv          — all rows EXCEPT disturbed days      (Hour = UTC)
  master_no_midday_midnight.csv    — all rows EXCEPT midday/midnight     (Hour = standard LT)
  quiet_days_only.csv              — rows that fall on quiet days        (Hour = UTC)
  disturbed_days_only.csv          — rows that fall on disturbed days    (Hour = UTC)
  midday_midnight_only.csv         — rows where LT is midday or midnight (Hour = standard LT)

Quiet/disturbed day parsing:
  Line format: YYYY MM q1q2...q5 q6q7...q0 d1d2...d5
  Fixed 2-char fields:
    positions  8-17  -> 5 quiet days  (q1-q5)
    positions 19-28  -> 5 quiet days  (q6-q10)
    positions 30-39  -> 5 disturbed days (d1-d5)

Local time (midday/midnight only):
  Fixed standard-time UTC offsets — DST is NOT applied.
  Station offsets:
    CP          -> UTC-3  (BRT, Brazil Standard Time)
    ElginAB     -> UTC-6  (CST, Central Standard Time)
    Jicamarca   -> UTC-5  (PET, Peru Time — no DST)
    MilstonHill -> UTC-5  (EST, Eastern Standard Time)
    Ramey       -> UTC-4  (AST, Atlantic Standard Time — PR no DST)

  local_hour = (utc_hour + offset) % 24

Midday window  : standard LT in {11, 12, 13}
Midnight window: standard LT in {23,  0,  1}

The Hour column in master_no_midday_midnight.csv and midday_midnight_only.csv
contains standard local time, not UTC, so the Scenario 4 ANN trains and
predicts on local-hour values directly.

Usage:
    python prepare_datasets.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent          # ann_evals/python_scripts/
project_root = script_dir.parent.parent                 # project root
data_dir     = script_dir.parent / "data"               # ann_evals/data/

MASTER_CSV   = project_root / "FINAL_MASTER.csv"
QD_TXT       = project_root / "quiet_and_disturbed.txt"

# ── Standard-time UTC offsets (no DST) ────────────────────────────────────────
STATION_UTC_OFFSET = {
    'CP':          -3,   # BRT  — Brazil Standard Time
    'ElginAB':     -6,   # CST  — Central Standard Time
    'Jicamarca':   -5,   # PET  — Peru Time (no DST)
    'MilstonHill': -5,   # EST  — Eastern Standard Time
    'Ramey':       -4,   # AST  — Atlantic Standard Time (Puerto Rico, no DST)
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _parse_field(line: str, start: int):
    """Return int from a 2-char field at `start`, or None if blank."""
    chunk = line[start:start + 2] if len(line) >= start + 2 else "  "
    return int(chunk.strip()) if chunk.strip() else None


def parse_quiet_disturbed(path: Path):
    """
    Returns (quiet_doys, disturbed_doys) as sets of integer day-of-year values
    derived from the NOAA monthly quiet/disturbed day file.
    """
    quiet_doys: set[int] = set()
    disturbed_doys: set[int] = set()

    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if len(line) < 7 or not line[:4].strip().isdigit():
                continue          # header / blank

            year  = int(line[0:4])
            month = int(line[5:7])

            quiet_days     = []
            disturbed_days = []

            for i in range(5):                     # q1-q5  -> positions 8-17
                d = _parse_field(line, 8 + i * 2)
                if d:
                    quiet_days.append(d)
            for i in range(5):                     # q6-q10 -> positions 19-28
                d = _parse_field(line, 19 + i * 2)
                if d:
                    quiet_days.append(d)
            for i in range(5):                     # d1-d5  -> positions 30-39
                d = _parse_field(line, 30 + i * 2)
                if d:
                    disturbed_days.append(d)

            for day in quiet_days:
                try:
                    quiet_doys.add(date(year, month, day).timetuple().tm_yday)
                except ValueError:
                    pass

            for day in disturbed_days:
                try:
                    disturbed_doys.add(date(year, month, day).timetuple().tm_yday)
                except ValueError:
                    pass

    return quiet_doys, disturbed_doys


def compute_standard_lt(df: pd.DataFrame) -> pd.Series:
    """
    Convert UTC Hour to standard local time using fixed UTC offsets (no DST).
    local_hour = (utc_hour + offset) % 24
    """
    lt = pd.Series(-1, index=df.index, dtype=int)
    for station, offset in STATION_UTC_OFFSET.items():
        mask = df["Station"] == station
        if mask.any():
            lt[mask] = (df.loc[mask, "Hour"].astype(int) + offset) % 24
    if (lt == -1).any():
        unknown = df.loc[lt == -1, "Station"].unique()
        raise ValueError(f"No UTC offset mapping for stations: {unknown}")
    return lt


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PREPARING ANN EVALUATION DATASETS")
print("=" * 65)
print()

print("Station standard-time UTC offsets (no DST):")
for stn, off in STATION_UTC_OFFSET.items():
    print(f"  {stn:<14}  UTC{off:+d}")
print()

# 1. Parse quiet/disturbed days
print(f"Parsing: {QD_TXT.name}")
quiet_doys, disturbed_doys = parse_quiet_disturbed(QD_TXT)
print(f"  Quiet DOYs      : {len(quiet_doys):3d}  {sorted(quiet_doys)}")
print(f"  Disturbed DOYs  : {len(disturbed_doys):3d}  {sorted(disturbed_doys)}")
print()

# 2. Load master CSV
print(f"Loading: {MASTER_CSV.name}")
df = pd.read_csv(MASTER_CSV)
print(f"  {len(df):,} rows  |  stations: {sorted(df['Station'].unique())}")
print()

# 3. Standard local-time column (Scenario 4 — no DST)
print("Computing standard local time (fixed UTC offsets, no DST)...")
df["LT"] = compute_standard_lt(df)
print(f"  LT range: {df['LT'].min()} - {df['LT'].max()}")
print()

MIDDAY_HOURS   = {11, 12, 13}
MIDNIGHT_HOURS = {23, 0, 1}

# 4. Boolean masks
mask_quiet     = df["DayOfYear"].isin(quiet_doys)
mask_disturbed = df["DayOfYear"].isin(disturbed_doys)
mask_midday    = df["LT"].isin(MIDDAY_HOURS)
mask_midnight  = df["LT"].isin(MIDNIGHT_HOURS)
mask_mid       = mask_midday | mask_midnight

print(f"  Quiet rows              : {mask_quiet.sum():>7,}")
print(f"  Disturbed rows          : {mask_disturbed.sum():>7,}")
print(f"  Midday rows (std LT)    : {mask_midday.sum():>7,}")
print(f"  Midnight rows (std LT)  : {mask_midnight.sum():>7,}")
print()

# 5. Build save frames
# quiet/disturbed CSVs keep UTC Hour unchanged
df_utc = df.drop(columns=["LT"])

# midday/midnight CSVs replace Hour with standard local time
df_lt = df.copy()
df_lt["Hour"] = df_lt["LT"]
df_lt = df_lt.drop(columns=["LT"])

# 6. Write CSVs
outputs = {
    "master_no_quiet.csv"           : (df_utc, ~mask_quiet),
    "master_no_disturbed.csv"       : (df_utc, ~mask_disturbed),
    "master_no_midday_midnight.csv" : (df_lt,  ~mask_mid),
    "quiet_days_only.csv"           : (df_utc,  mask_quiet),
    "disturbed_days_only.csv"       : (df_utc,  mask_disturbed),
    "midday_midnight_only.csv"      : (df_lt,   mask_mid),
}

for fname, (frame, mask) in outputs.items():
    path   = data_dir / fname
    subset = frame[mask]
    subset.to_csv(path, index=False)
    hour_type = "local-std" if "midday" in fname or "midnight" in fname else "UTC"
    print(f"  {fname:<40} {len(subset):>7,} rows  Hour={hour_type}  ->  saved")

print()
print("All datasets written to:", data_dir)
print("=" * 65)
