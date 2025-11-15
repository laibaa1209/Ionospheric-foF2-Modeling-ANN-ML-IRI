import pandas as pd
import os
import glob

# ==== CONFIG ====
data_folder = r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs"
master_file = os.path.join(data_folder, "Master_FOF2_F107_2019.csv")
f107_file = os.path.join(data_folder, "f10.7.csv")

expected_days = set(range(1, 366))   # 2019 → 365 days
expected_hours = set(range(0, 24))   # 0–23 hours
records_per_station = 365 * 24       # 8760


def check_file(df, filename, is_master=False):

    print(f"\n==============================")
    print(f"Checking: {filename}")
    print(f"==============================\n")

    # --- Basic shape ---
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")

    # --- DayOfYear issues ---
    days = set(df["DayOfYear"].unique())
    missing_days = expected_days - days
    extra_days = days - expected_days

    if missing_days:
        print(f"❌ Missing {len(missing_days)} days, e.g.: {sorted(list(missing_days))[:5]}")
    else:
        print("✔ All 365 days present")

    if extra_days:
        print(f"❌ Found invalid day numbers: {sorted(list(extra_days))[:5]}")

    # --- Hour issues ---
    if "Hour" in df.columns:
        hours = set(df["Hour"].unique())
        missing_hours = expected_hours - hours
        extra_hours = hours - expected_hours

        if missing_hours:
            print(f"❌ Missing hour values: {sorted(list(missing_hours))}")
        else:
            print("✔ Hours 0–23 are present")

        if extra_hours:
            print(f"❌ Invalid hour values detected: {sorted(list(extra_hours))}")

        # Check day-by-day
        days_missing_hours = 0
        for d in expected_days:
            hrs = set(df[df["DayOfYear"] == d]["Hour"])
            if hrs != expected_hours:
                days_missing_hours += 1

        if days_missing_hours > 0:
            print(f"❌ {days_missing_hours} days do NOT have complete 24 hours")
        else:
            print("✔ Every day has complete hourly data")

    # --- Duplicate Day-Hour pairs ---
    if "Hour" in df.columns:
        dup = df.duplicated(subset=["DayOfYear", "Hour"]).sum()
        if dup > 0:
            print(f"❌ Duplicate (Day,Hour) rows: {dup}")
        else:
            print("✔ No duplicate DayOfYear–Hour rows")

    # --- Check for NaN values ---
    nan_counts = df.isna().sum()
    nan_total = nan_counts.sum()

    if nan_total > 0:
        print(f"❌ Missing values found:\n{nan_counts}")
    else:
        print("✔ No missing (NaN) values")

    # --- Master CSV station checks ---
    if is_master:
        if "Station" not in df.columns:
            print("❌ Master CSV missing 'Station' column!")
        else:
            stations = df["Station"].unique()
            print(f"\nStations found: {stations}")

            for s in stations:
                count = len(df[df["Station"] == s])
                if count != records_per_station:
                    print(f"❌ Station {s} has wrong number of rows: {count}")
                else:
                    print(f"✔ Station {s} has 8760 rows")

    print("\nCheck complete.")
    print("----------------------------------------------------")


# ==== CHECK F10.7 CSV ====
f107 = pd.read_csv(f107_file)
check_file(f107, "f10.7.csv", is_master=False)

# ==== CHECK ALL REGIONAL CSV FILES ====
print("\n\n### CHECKING REGIONAL CSV FILES ###\n")

region_files = [
    f for f in glob.glob(os.path.join(data_folder, "*_2019.csv"))
    if "f10.7" not in f.lower()
]

for file in region_files:
    df = pd.read_csv(file)
    check_file(df, os.path.basename(file), is_master=False)

# ==== CHECK MASTER CSV ====
print("\n\n### CHECKING MASTER CSV FILE ###\n")

master = pd.read_csv(master_file)
check_file(master, "Master_FOF2_F107_2019.csv", is_master=True)
