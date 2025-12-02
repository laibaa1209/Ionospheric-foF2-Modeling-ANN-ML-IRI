import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Fof2 Data", "preprocessed_fof2_csvs", "FINAL_MASTER.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "fof2_ann")
os.makedirs(MODEL_DIR, exist_ok=True)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["foF2"])  # ensure target present

# Feature engineering: encode cyclic variables (DayOfYear, Hour)
# DayOfYear column in dataset is "DayOfYear"
doy = df["DayOfYear"].astype(float).values
hour = df["Hour"].astype(float).values

# normalize periodic ranges
doy_cycle = 365.0
hour_cycle = 24.0
doy_rad = 2 * np.pi * (doy % doy_cycle) / doy_cycle
hour_rad = 2 * np.pi * (hour % hour_cycle) / hour_cycle

df["doy_sin"] = np.sin(doy_rad)
df["doy_cos"] = np.cos(doy_rad)
df["hour_sin"] = np.sin(hour_rad)
df["hour_cos"] = np.cos(hour_rad)

# Select input features (adjust if you want more/less)
feature_cols = ["doy_sin", "doy_cos", "hour_sin", "hour_cos", "Longitude", "Latitude", "F10.7"]
X = df[feature_cols].values.astype(np.float32)
y = df["foF2"].values.astype(np.float32).reshape(-1, 1)

# Train / val / test split (70/15/15)
n = len(X)
indices = np.arange(n)
np.random.shuffle(indices)
train_end = int(0.70 * n)
val_end = train_end + int(0.15 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Normalization layer (TF) - adapt on training data
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

# Build model
def build_model():
    inputs = tf.keras.Input(shape=(X_train.shape[1],), name="inputs")
    x = normalizer(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="fof2")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse",
                  metrics=["mae"])
    return model

model = build_model()
model.summary()

# Prepare tf.data datasets
BATCH_SIZE = 128
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2048, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Callbacks
early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"),
                                                monitor="val_loss", save_best_only=True)

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[early, checkpoint], verbose=2)

# Save final model (SavedModel)
model.save(os.path.join(MODEL_DIR, "saved_model"), include_optimizer=False)

# Evaluate on test set
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_loss.png"))
plt.close()

# Plot actual vs predicted by DOY (use DayOfYear from test rows)
doy_test = doy[test_idx]
order = np.argsort(doy_test)
plt.figure(figsize=(12,5))
plt.scatter(doy_test[order], y_test.flatten()[order], s=15, alpha=0.6, label="actual")
plt.plot(doy_test[order], y_pred_test[order], "r-", linewidth=1.5, label="predicted")
plt.xlabel("Day of Year (DOY)")
plt.ylabel("foF2")
plt.title(f"foF2 actual vs predicted (R²={r2:.3f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "pred_vs_actual_by_doy.png"))
plt.close()

# quick sample predictions csv
out = pd.DataFrame({
    "DayOfYear": doy[test_idx],
    "Hour": hour[test_idx],
    "foF2_actual": y_test.flatten(),
    "foF2_pred": y_pred_test
})
out.to_csv(os.path.join(MODEL_DIR, "predictions_test_sample.csv"), index=False)

print("Model and plots saved to:", MODEL_DIR)