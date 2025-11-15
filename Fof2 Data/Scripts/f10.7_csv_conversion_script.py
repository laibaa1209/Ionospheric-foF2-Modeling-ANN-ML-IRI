import pandas as pd
from datetime import datetime

input_file = r'C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\raw_txt_fof2_files\F10.7.txt'
output_file = r'C:\Laiba\Uni\5th Sem\TBW\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\f10.7.csv'

rows = []

with open(input_file, "r") as f:
    for line in f:
        parts = line.split()

        #year, month, day
        year = int(parts[0]) + 2000
        month = int(parts[1])
        day = int(parts[2])

        #f10.7 in third last position
        F107 = float(parts[-3])

        #dya of yr conversion
        date = datetime(year, month, day)
        day_of_year = date.timetuple().tm_yday

        rows.append([day_of_year, F107])

#save csv
df = pd.DataFrame(rows, columns=["DayOfYear", "F10.7"])
df.to_csv(output_file, index=False)

print("Created f10.7.csv successfully!")
