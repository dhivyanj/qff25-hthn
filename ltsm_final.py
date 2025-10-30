"""
Hybrid LSTM Regression + Renewable Energy Recommendation Evaluation
--------------------------------------------------------------------
This script:
1. Trains an LSTM on weather data to predict temperature.
2. Evaluates regression metrics (RMSE, MAE, R², etc.).
3. Derives renewable energy source recommendations (Hydro/Solar/Wind)
   from weather + temperature predictions.
4. Computes classification metrics (Accuracy, Precision, Recall, F1).
5. Optionally compares to Quantum Model outputs.
--------------------------------------------------------------------
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# =====================================================
# 1. LOAD & CLEAN WEATHER DATA
# =====================================================
print("\n--- Loading and Preparing Data ---")
df = pd.read_csv("weatherHistory.csv")
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df = df.set_index('Formatted Date').sort_index()
df = df[~df.index.duplicated(keep='first')]
df = df.resample('h').interpolate(method='time')

features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)']
target = 'Temperature (C)'
df_clean = df[features + [target]].dropna()

X = df_clean[features].values
Y = df_clean[target].values.reshape(-1, 1)

print(f"Cleaned dataset: {X.shape[0]} samples | Features: {len(features)}")

# =====================================================
# 2. SCALE DATA & TRAIN/TEST SPLIT
# =====================================================
feature_scaler = MinMaxScaler((0, 1))
target_scaler = MinMaxScaler((0, 1))
X_scaled = feature_scaler.fit_transform(X)
Y_scaled = target_scaler.fit_transform(Y)

split_ratio = 0.8
split_idx = int(len(Y_scaled) * split_ratio)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
Y_train, Y_test = Y_scaled[:split_idx], Y_scaled[split_idx:]

# =====================================================
# 3. CREATE SEQUENCES FOR LSTM
# =====================================================
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24
X_seq_train, Y_seq_train = create_sequences(X_train, Y_train, TIME_STEPS)
X_seq_test, Y_seq_test = create_sequences(X_test, Y_test, TIME_STEPS)

print(f"Training sequences: {X_seq_train.shape}, Test sequences: {X_seq_test.shape}")

# =====================================================
# 4. BUILD & TRAIN LSTM MODEL
# =====================================================
model = Sequential([
    Input(shape=(TIME_STEPS, len(features))),
    LSTM(50, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("\n--- Training LSTM Model ---")
history = model.fit(
    X_seq_train, Y_seq_train,
    epochs=20, batch_size=32,
    validation_data=(X_seq_test, Y_seq_test),
    shuffle=False, verbose=1
)

# =====================================================
# 5. REGRESSION EVALUATION
# =====================================================
print("\n--- Evaluating LSTM Regression ---")
pred_scaled = model.predict(X_seq_test)
pred_real = target_scaler.inverse_transform(pred_scaled)
Y_test_real = target_scaler.inverse_transform(Y_seq_test)

y_true = Y_test_real.flatten()
y_pred = pred_real.flatten()

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.3f}°C | MAE: {mae:.3f}°C | R²: {r2:.3f}")

# =====================================================
# 6. RENEWABLE ENERGY RECOMMENDATION (HEURISTIC)
# =====================================================
print("\n--- Renewable Energy Recommendation ---")

def recommend_energy(temp, humidity, wind_speed):
    if wind_speed > 20:
        return "Wind"
    elif humidity > 0.7:
        return "Hydro"
    elif temp > 20 and humidity < 0.6:
        return "Solar"
    elif wind_speed > 15:
        return "Wind"
    elif humidity > 0.5:
        return "Hydro"
    else:
        return "Solar"

# Extract test features (real-world scale)
X_test_df = df_clean[features].iloc[split_idx + TIME_STEPS : split_idx + TIME_STEPS + len(Y_seq_test)]
X_test_real = feature_scaler.inverse_transform(X_test_df.values)
humidity_real = X_test_real[:, 0]
wind_speed_real = X_test_real[:, 1]

temperature_pred_real = y_pred
temperature_true_real = y_true

# Derive energy source predictions
pred_labels = [recommend_energy(t, h, w) for t, h, w in zip(temperature_pred_real, humidity_real, wind_speed_real)]
true_labels = [recommend_energy(t, h, w) for t, h, w in zip(temperature_true_real, humidity_real, wind_speed_real)]

# =====================================================
# 7. CLASSIFICATION METRICS
# =====================================================
print("\n--- Energy Source Classification Metrics ---")
labels = ["Hydro", "Solar", "Wind"]
report = classification_report(true_labels, pred_labels, labels=labels, digits=3, output_dict=True)
report_df = pd.DataFrame(report).transpose()
acc = accuracy_score(true_labels, pred_labels)
prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, labels=labels, average='weighted')

print(f"\nOverall Accuracy: {acc:.3f}")
print(f"Weighted Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
print("\nDetailed per-class report:\n", report_df.loc[labels, ["precision", "recall", "f1-score", "support"]])

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix — Energy Source Recommendation')
plt.xticks(range(3), labels)
plt.yticks(range(3), labels)
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()

out_dir = "model_comparison_outputs"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "energy_confusion_matrix.png"), dpi=150)
plt.close()

# =====================================================
# 8. OPTIONAL: COMPARE WITH QUANTUM MODEL
# =====================================================
quantum_labels_path = "quantum_model_predictions.csv"
if os.path.exists(quantum_labels_path):
    df_q = pd.read_csv(quantum_labels_path)
    if "predicted_label" in df_q.columns:
        q_labels = df_q["predicted_label"][:len(pred_labels)]
        print("\n--- Comparing LSTM vs Quantum Model ---")
        print(classification_report(q_labels, pred_labels, labels=labels, digits=3))
        print(f"Agreement Accuracy: {accuracy_score(q_labels, pred_labels):.3f}")

# =====================================================
# 9. SAVE RESULTS
# =====================================================
pd.DataFrame({
    "time": X_test_df.index[:len(pred_labels)],
    "temp_true": temperature_true_real,
    "temp_pred": temperature_pred_real,
    "humidity": humidity_real,
    "wind_speed": wind_speed_real,
    "true_label": true_labels,
    "predicted_label": pred_labels
}).to_csv(os.path.join(out_dir, "energy_recommendations.csv"), index=False)

report_df.to_csv(os.path.join(out_dir, "classification_report.csv"))
cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

print(f"\n✅ Metrics, confusion matrix, and CSVs saved in '{out_dir}/'\n")
