import pandas as pd
import os
import glob

# Folder paths
data_folder = r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs"
f107_file = os.path.join(data_folder, "f10.7.csv")

# Load daily F10.7 values
f107 = pd.read_csv(f107_file)[["DayOfYear", "F10.7"]]

# Region CSVs (exclude f10.7.csv)
region_files = [
    f for f in glob.glob(os.path.join(data_folder, "*_2019.csv"))
    if "f10.7" not in f.lower()
]

merged_frames = []

for file in region_files:
    df = pd.read_csv(file)

    # Remove any existing F10.7 columns from region files
    for col in df.columns:
        if "f10" in col.lower():
            df = df.drop(columns=[col])

    # Reset hourly values to 0–23 for each day
    df["Hour"] = df.groupby("DayOfYear").cumcount()
    
    # Merge daily F10.7 → automatically repeats for 24 hours
    merged = df.merge(f107, on="DayOfYear", how="left")

    # Add station name
    station_name = os.path.basename(file).replace("_2019.csv", "")
    merged["Station"] = station_name

    merged_frames.append(merged)

# Combine all stations
master = pd.concat(merged_frames, ignore_index=True)

# Save final master CSV
output_file = os.path.join(data_folder, "Master_FOF2_F107_2019.csv")
master.to_csv(output_file, index=False)

print("Master CSV created successfully with Hours = 0–23 and clean F10.7 merge.")
