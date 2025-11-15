import pandas as pd
import matplotlib.pyplot as plt

def remove_equinox_solstice(master_path, cleaned_path, removed_path):
    print("\n---------- PROCESS STARTED ----------\n")
    
    # Load data
    df = pd.read_csv(master_path)
    original_rows = len(df)
    print(f"Original rows in master CSV: {original_rows}")
    
    # Days to remove
    remove_doy = [79, 172, 266, 356]
    print(f"Removing these DayOfYear values: {remove_doy}\n")

    # Extract removed rows
    removed_rows = df[df['DayOfYear'].isin(remove_doy)]
    removed_count = len(removed_rows)
    print(f"Rows removed (should be 480 if all 4 days exist): {removed_count}")

    # Save removed values
    removed_rows.to_csv(removed_path, index=False)
    print(f"Saved removed rows to: {removed_path}")

    # Remaining rows
    cleaned_df = df[~df['DayOfYear'].isin(remove_doy)]
    cleaned_count = len(cleaned_df)
    print(f"Rows remaining after removal: {cleaned_count}")

    # Save cleaned values
    cleaned_df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned CSV to: {cleaned_path}")

    # Visualization (before/after bar chart)
    plt.figure(figsize=(7, 5))
    plt.bar(["Original", "Removed", "Remaining"], 
            [original_rows, removed_count, cleaned_count])
    plt.title("Row Count Comparison")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Rows")
    plt.tight_layout()
    plt.show()

    print("\n---------- PROCESS COMPLETED ----------\n")

# ---- RUN FUNCTION ----
remove_equinox_solstice(
    master_path=r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\Master_FOF2_F107_2019.csv",
    cleaned_path=r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\FINAL_MASTER.csv",
    removed_path=r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\equinox_solstice.csv"
)


df = pd.read_csv(r"C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\Master_FOF2_F107_2019.csv")

remove_doy = [79, 172, 266, 356]

print("---- Rows per DayOfYear (Original CSV) ----")
for d in remove_doy:
    count = len(df[df["DayOfYear"] == d])
    print(f"DOY {d}: {count} rows (expected 5*24 = 120)")