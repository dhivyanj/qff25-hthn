import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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