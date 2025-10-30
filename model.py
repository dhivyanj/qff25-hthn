import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# --- 1. Load Data ---
try:
    df_full = pd.read_csv('weatherHistory.csv')
except FileNotFoundError:
    print("Error: Could not find the weatherHistory.csv file.")
    raise

# --- 2. Clean and Set Index ---
# Convert date column to datetime objects
df_full['Formatted Date'] = pd.to_datetime(df_full['Formatted Date'], utc=True)

# Set the date as the index and sort the data
df_full = df_full.set_index('Formatted Date').sort_index()

# --- Handle duplicate index entries before resampling ---
# Keep the first occurrence of each timestamp
df_full = df_full[~df_full.index.duplicated(keep='first')]

# The data is hourly, but we resample to 'h' to fill any missing time steps
# We use 'interpolate' to fill gaps (e.g., missing hours)
# Using 'h' as 'H' is deprecated
df_full = df_full.resample('h').interpolate(method='time')

# --- 3. Select Features (X) and Target (Y) ---
N_FEATURES = 3
features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)']
target = 'Temperature (C)'

# Drop rows where our target or features are missing
df_clean = df_full[features + [target]].dropna()

X = df_clean[features].values
Y = df_clean[target].values.reshape(-1, 1)

print(f"Original data shape: {X.shape}")

# --- 4. Scale Data ---
# Standard practice for LSTMs is to scale features to [0, 1]
feature_scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = feature_scaler.fit_transform(X)

# Scale the target (Temperature) to [0, 1]
target_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaled = target_scaler.fit_transform(Y)

# --- 5. Create Train/Test Split ---
# We must split time-series data sequentially
test_percent = 0.2
split_idx = int(len(Y_scaled) * (1 - test_percent))

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
Y_train, Y_test = Y_scaled[:split_idx], Y_scaled[split_idx:]

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error

# --- 1. Create Sequences for LSTM ---
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24 # Use 24 hours of history
X_seq_train, Y_seq_train = create_sequences(X_train, Y_train, TIME_STEPS)
X_seq_test, Y_seq_test = create_sequences(X_test, Y_test, TIME_STEPS)

# The shape will now be [samples, 24, 3]
print(f"Training sequences shape: {X_seq_train.shape}")
print(f"Test sequences shape: {X_seq_test.shape}")

# --- 2. Build the LSTM Model ---
model_lstm_simple = Sequential([
    # This Input shape is now (24, 3) automatically
    Input(shape=(TIME_STEPS, N_FEATURES)), 
    LSTM(units=50, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, activation='relu'),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1) 
])

# --- 3. Compile and Train ---
print("\n--- Training LSTM Model on SELECTED Features ---")
model_lstm_simple.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

history = model_lstm_simple.fit(
    X_seq_train, Y_seq_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_seq_test, Y_seq_test),
    shuffle=False,
    verbose=1
)
print("--- Model Training Complete ---")

# --- 4. Evaluate the LSTM Model ---
predictions_scaled = model_lstm_simple.predict(X_seq_test)
predictions_real = target_scaler.inverse_transform(predictions_scaled)
Y_test_real = target_scaler.inverse_transform(Y_seq_test)

rmse = np.sqrt(mean_squared_error(Y_test_real, predictions_real))
print(f"\n--- Model Evaluation ---")
print(f"Test RMSE: {rmse:.4f} (degrees C)")

# --- BEGIN: Comparison operators / diagnostics (drop-in) ---
# Ensure directories
out_dir = "model_comparison_outputs"
os.makedirs(out_dir, exist_ok=True)

# 1) Prepare ground-truth and predictions as 1D arrays (real Celsius)
y_true = Y_test_real.flatten()
y_pred = predictions_real.flatten()

# 2) Additional numeric metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) != 0 else np.nan

# 3) Baseline predictors for context
#  - Persistence (last observed value as prediction)
#    For sequence-model test set the true target for time t corresponds to the input ending at t-1.
#    We can approximate persistence by shifting y_true by 1 (first sample uses same as true[0])
persistence_pred = np.roll(y_true, 1)
persistence_pred[0] = y_true[0]  # fallback for the first sample

#  - Mean predictor
mean_pred = np.full_like(y_true, y_true.mean())

# Compute baseline metrics
mae_persist = mean_absolute_error(y_true, persistence_pred)
rmse_persist = np.sqrt(mean_squared_error(y_true, persistence_pred))
mae_mean = mean_absolute_error(y_true, mean_pred)
rmse_mean = np.sqrt(mean_squared_error(y_true, mean_pred))

# 4) Model size and timing
param_count = model_lstm_simple.count_params()
# Measure predict runtime (wall time) for the test set
t0 = time.time()
_ = model_lstm_simple.predict(X_seq_test, verbose=0)
predict_time = time.time() - t0
avg_predict_ms = (predict_time / len(X_seq_test)) * 1000.0 if len(X_seq_test) else np.nan

# 5) Summary print + csv save
summary = {
    "model": "LSTM_simple",
    "test_samples": int(len(y_true)),
    "param_count": int(param_count),
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "nrmse": float(nrmse),
    "predict_total_s": float(predict_time),
    "predict_avg_ms": float(avg_predict_ms),
    "mae_persistence": float(mae_persist),
    "rmse_persistence": float(rmse_persist),
    "mae_mean_baseline": float(mae_mean),
    "rmse_mean_baseline": float(rmse_mean)
}

print("\n--- Extended Evaluation Summary ---")
for k, v in summary.items():
    print(f"{k}: {v}")

# Save summary
pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "evaluation_summary.csv"), index=False)

# 6) Diagnostic plots
# Create a timestamp index for plotting if original timestamps are available:
# We'll attempt to reconstruct index from df_clean — align with the test split and TIME_STEPS offset
try:
    # df_clean index is the timestamps (df_clean was created earlier)
    # The test portion of df_clean corresponds to indices after split_idx
    df_index = df_clean.index[split_idx + TIME_STEPS : split_idx + TIME_STEPS + len(y_true)]
    time_index = df_index
except Exception:
    # fallback to a simple range index
    time_index = np.arange(len(y_true))

# a) time series: truth vs prediction (plot a subset if long)
plt.figure(figsize=(12, 4))
subset_n = min(1000, len(y_true))  # show up to 1000 samples to keep plot readable
plt.plot(time_index[:subset_n], y_true[:subset_n], label='Actual', linewidth=1)
plt.plot(time_index[:subset_n], y_pred[:subset_n], label='LSTM Predicted', linewidth=1)
plt.plot(time_index[:subset_n], persistence_pred[:subset_n], label='Persistence Baseline', linewidth=0.8, alpha=0.7)
plt.legend()
plt.title('Temperature: Actual vs LSTM Prediction (subset)')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.tight_layout()
ts_plot_path = os.path.join(out_dir, "time_series_comparison.png")
plt.savefig(ts_plot_path, dpi=150)
plt.close()

# b) error histogram
errors = y_pred - y_true
plt.figure(figsize=(6,4))
plt.hist(errors, bins=50)
plt.title('Prediction Error Distribution (pred - true)')
plt.xlabel('Error (C)')
plt.ylabel('Count')
plt.tight_layout()
err_hist_path = os.path.join(out_dir, "error_histogram.png")
plt.savefig(err_hist_path, dpi=150)
plt.close()

# c) rolling RMSE (windowed) to inspect stability
window = min(100, len(y_true)//10) if len(y_true)>0 else 1
if window >= 2:
    df_err = pd.DataFrame({"error_sq": (errors**2)})
    rolling_rmse = np.sqrt(df_err["error_sq"].rolling(window=window, min_periods=1).mean())
    plt.figure(figsize=(12,4))
    plt.plot(time_index[:len(rolling_rmse)], rolling_rmse, label=f'Rolling RMSE (window={window})')
    plt.title('Rolling RMSE Over Test Set')
    plt.xlabel('Time')
    plt.ylabel('RMSE (C)')
    plt.tight_layout()
    rolling_path = os.path.join(out_dir, "rolling_rmse.png")
    plt.savefig(rolling_path, dpi=150)
    plt.close()

# d) Scatter: predicted vs actual (shows bias)
plt.figure(figsize=(5,5))
plt.scatter(y_true, y_pred, s=6, alpha=0.5)
mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], '--', linewidth=1, label='Ideal')
plt.xlabel('Actual (C)')
plt.ylabel('Predicted (C)')
plt.title('Predicted vs Actual')
plt.tight_layout()
scatter_path = os.path.join(out_dir, "pred_vs_actual_scatter.png")
plt.savefig(scatter_path, dpi=150)
plt.close()

# 7) Save full per-sample CSV (actual, pred, persistence, errors)
df_out = pd.DataFrame({
    "time": list(time_index) if not isinstance(time_index, np.ndarray) else list(time_index),
    "actual": y_true,
    "predicted": y_pred,
    "persistence": persistence_pred,
    "error": errors
})
df_out.to_csv(os.path.join(out_dir, "per_sample_predictions.csv"), index=False)

print(f"\nSaved plots and CSVs to '{out_dir}/'.")
print(f"Time-series plot: {ts_plot_path}")
print(f"Error histogram: {err_hist_path}")
print(f"Pred vs Actual scatter: {scatter_path}")
# --- END: Comparison operators / diagnostics ---

# --- BEGIN: Renewable Energy Source Recommendation Comparison ---

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n--- Renewable Energy Recommendation Comparison ---")

# Step 1: Define heuristic rules to map weather → energy source
def recommend_energy(temp, humidity, wind_speed):
    """
    Simple heuristic classifier:
    - Wind if wind_speed > 20 km/h
    - Solar if temperature > 20°C and humidity < 0.6
    - Hydro if humidity > 0.7
    - Otherwise choose best fit
    """
    if wind_speed > 20:
        return "Wind"
    elif humidity > 0.7:
        return "Hydro"
    elif temp > 20 and humidity < 0.6:
        return "Solar"
    else:
        # Fallback — compare which metric dominates
        if wind_speed > 15:
            return "Wind"
        elif humidity > 0.5:
            return "Hydro"
        else:
            return "Solar"

# Step 2: Get original feature values aligned with test set
# (the test data corresponds to the last split_idx portion)
X_test_df = df_clean[features].iloc[split_idx + TIME_STEPS : split_idx + TIME_STEPS + len(Y_seq_test)]

# Inverse-scale features for interpretability
X_test_real = feature_scaler.inverse_transform(X_test_df.values)
humidity_real = X_test_real[:, 0]
wind_speed_real = X_test_real[:, 1]
wind_bearing_real = X_test_real[:, 2]
temperature_pred_real = predictions_real.flatten()

# Step 3: Generate energy source predictions for LSTM test data
lstm_energy_pred = [
    recommend_energy(temp, hum, ws)
    for temp, hum, ws in zip(temperature_pred_real, humidity_real, wind_speed_real)
]

# Step 4: Generate “actual” labels heuristically from true weather (as proxy ground truth)
temperature_true_real = Y_test_real.flatten()
actual_energy_label = [
    recommend_energy(temp, hum, ws)
    for temp, hum, ws in zip(temperature_true_real, humidity_real, wind_speed_real)
]

# Step 5: Evaluate classification metrics
print("\nEnergy source classification report (LSTM heuristic):")
print(classification_report(actual_energy_label, lstm_energy_pred, digits=3))

acc = accuracy_score(actual_energy_label, lstm_energy_pred)
cm = confusion_matrix(actual_energy_label, lstm_energy_pred, labels=["Hydro", "Solar", "Wind"])

print(f"Accuracy: {acc:.3f}")
print("\nConfusion Matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=["Hydro", "Solar", "Wind"], columns=["Hydro", "Solar", "Wind"]))

# Optional: visualize confusion matrix
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title('LSTM-based Energy Source Recommendation')
plt.xticks(range(3), ["Hydro", "Solar", "Wind"])
plt.yticks(range(3), ["Hydro", "Solar", "Wind"])
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "energy_confusion_matrix.png"), dpi=150)
plt.close()

# Step 6: Compare against Quantum Model (if available)
# (optional if you have the CSV from your quantum run)
quantum_labels_path = "quantum_model_predictions.csv"  # or update this path
if os.path.exists(quantum_labels_path):
    df_q = pd.read_csv(quantum_labels_path)
    if 'predicted_label' in df_q.columns:
        q_labels = df_q['predicted_label'][:len(lstm_energy_pred)].values
        print("\n--- Comparing LSTM vs Quantum Model Recommendations ---")
        print(classification_report(q_labels, lstm_energy_pred, digits=3))
        acc_q = accuracy_score(q_labels, lstm_energy_pred)
        print(f"Agreement Accuracy (LSTM vs Quantum): {acc_q:.3f}")

# Step 7: Save per-sample recommendation results
df_reco = pd.DataFrame({
    "time": list(X_test_df.index[:len(lstm_energy_pred)]),
    "temp_true": temperature_true_real,
    "temp_pred": temperature_pred_real,
    "humidity": humidity_real,
    "wind_speed": wind_speed_real,
    "true_label": actual_energy_label,
    "lstm_predicted_label": lstm_energy_pred
})
df_reco.to_csv(os.path.join(out_dir, "energy_recommendations.csv"), index=False)

print(f"\n✅ Saved energy classification results and plots in '{out_dir}/'")

# --- END: Renewable Energy Source Recommendation Comparison ---

