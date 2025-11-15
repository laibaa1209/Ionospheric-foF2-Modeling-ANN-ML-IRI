import pandas as pd
import os

# Paths
f107_path = r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\f10.7.csv"  # Daily F10.7
regions_folder = r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs"
output_folder = os.path.join(os.path.dirname(regions_folder), "Regional_Dataset_F107")
os.makedirs(output_folder, exist_ok=True)

# Load daily F10.7 CSV
f107_df = pd.read_csv(f107_path)
f107_df = f107_df[['DayOfYear', 'F10.7']]  # Keep only needed columns

# Loop through all region CSVs
for region_file in os.listdir(regions_folder):
    if region_file.endswith(".csv"):
        region_name = os.path.splitext(region_file)[0]
        region_path = os.path.join(regions_folder, region_file)

        # Load region CSV
        region_df = pd.read_csv(region_path)

        # Merge F10.7 values on DayOfYear
        merged_df = pd.merge(region_df, f107_df, on='DayOfYear', how='left', suffixes=('', '_F107'))
        merged_df['F10.7'] = merged_df['F10.7_F107']
        merged_df.drop(columns=['F10.7_F107'], inplace=True)

        # Save merged CSV to the subfolder
        output_path = os.path.join(output_folder, f"{region_name}_merged.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Merged F10.7 into {region_name}: {output_path}")
