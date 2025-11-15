import pandas as pd
from datetime import datetime
import os

# === FOLDER PATHS ===
input_folder = r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\raw_txt_fof2_files"
output_folder = r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs"

# create output folder if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# === REGION METADATA ===
regions = {
    # "ElginAB": (30.5, 273.0),
    # "Jicamarca": (-12.0, 283.2),
    # "CP": (-22.7, 315.0),
    "MilstonHill": (42.6, 288.5),
    # "Ramey": (18.5, 292.9)
}

# === PROCESS EACH REGION ===
for region, (lat, lon) in regions.items():
    files = [
        os.path.join(input_folder, f"{region}_2019_part1.txt"),
        os.path.join(input_folder, f"{region}_2019_part2.txt")
    ]
    data = []

    for file in files:
        if not os.path.exists(file):
            print(f"⚠️ File not found: {file}")
            continue
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                timestamp = parts[0]
                fof2 = float(parts[2])
                dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.000Z")
                day_of_year = dt.timetuple().tm_yday
                hour = dt.hour + 1  # 1–24 format
                data.append((day_of_year, hour, fof2, lon, lat, None))

    # Create DataFrame
    df = pd.DataFrame(data, columns=["DayOfYear", "Hour", "foF2", "Longitude", "Latitude", "F10.7"])

    # Remove duplicate hours for the same day
    df = df.drop_duplicates(subset=["DayOfYear", "Hour"], keep="first").sort_values(["DayOfYear", "Hour"])

    # Save CSV
    output_path = os.path.join(output_folder, f"{region}_2019.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

print("\nAll region CSVs created successfully!")
