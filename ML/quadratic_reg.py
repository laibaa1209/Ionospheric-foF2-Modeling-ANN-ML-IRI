import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np


df = pd.read_csv(r'C:\Ali\Programming\Ionospheric-foF2-Modeling-ANN-ML-IRI\Fof2 Data\preprocessed_fof2_csvs\FINAL_MASTER.csv')
print("Shape of the loaded Data:", df.shape)

x = df.drop(['foF2', 'Station'],axis=1)
y = df['foF2']

model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())   
])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE:", mse)
print(f"R²:", r2, "\n")

# Predictions for the entire dataset (so the regression line overlays the full foF2 series)
y_pred_all = model.predict(x)  # predict on full X

# Use a DOY (day-of-year) column for the x-axis if available, otherwise fall back to index
if 'DOY' in df.columns:
    x_axis = df['DOY'].values
elif 'doy' in df.columns:
    x_axis = df['doy'].values
elif 'DayOfYear' in df.columns:
    x_axis = df['DayOfYear'].values
elif 'date' in df.columns:
    dates = pd.to_datetime(df['date'])
    x_axis = dates.dt.dayofyear.values
elif 'Date' in df.columns:
    dates = pd.to_datetime(df['Date'])
    x_axis = dates.dt.dayofyear.values
else:
    # fallback: use sample index if no DOY/date column found
    x_axis = np.arange(len(y))

# If DOY repeats (multiple years), sorting will group by DOY; keep this behavior but preserve pairing
order = np.argsort(x_axis)
x_sorted = x_axis[order]
y_sorted = y.values[order]
y_pred_all_sorted = y_pred_all[order]

plt.figure(figsize=(12, 6))
plt.scatter(x_sorted, y_sorted, alpha=0.6, s=30, label='Actual foF2 Values')
plt.plot(x_sorted, y_pred_all_sorted, 'r-', lw=2, label='Regression Line (predicted on full X)')
plt.xlabel('Day of Year (DOY)', fontsize=12)
plt.ylabel('foF2 Values', fontsize=12)
plt.title(f'foF2 Data Points and Regression Line (R² on test = {r2:.4f})', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
# limit x-ticks to 1..365 if DOY values are in that range
if x_sorted.min() >= 1 and x_sorted.max() <= 366:
    plt.xticks(np.arange(1, 367, 30))
plt.tight_layout()
plt.show()