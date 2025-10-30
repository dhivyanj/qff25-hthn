import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Weather Prediction App", layout="wide")

st.title("🌦️ Weather Prediction with LSTMs")
st.write("This app uses an LSTM model to predict temperature based on historical weather data.")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    try:
        df_full = pd.read_csv('weatherHistory.csv')
    except FileNotFoundError:
        st.error("Error: Could not find the weatherHistory.csv file.")
        return None

    # --- 2. Clean and Set Index ---
    df_full['Formatted Date'] = pd.to_datetime(df_full['Formatted Date'], utc=True)
    df_full = df_full.set_index('Formatted Date').sort_index()
    df_full = df_full[~df_full.index.duplicated(keep='first')]
    df_full = df_full.resample('h').interpolate(method='time')
    return df_full

df_full = load_data()

if df_full is not None:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df_full.head())

    # --- 3. Select Features and Target ---
    N_FEATURES = 3
    features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)']
    target = 'Temperature (C)'

    df_clean = df_full[features + [target]].dropna()
    X = df_clean[features].values
    Y = df_clean[target].values.reshape(-1, 1)

    st.write(f"Original data shape: `{X.shape}`")

    # --- 4. Scale Data ---
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = feature_scaler.fit_transform(X)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    Y_scaled = target_scaler.fit_transform(Y)

    # --- 5. Create Train/Test Split ---
    test_percent = 0.2
    split_idx = int(len(Y_scaled) * (1 - test_percent))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    Y_train, Y_test = Y_scaled[:split_idx], Y_scaled[split_idx:]

    st.write(f"Training data shape: `{X_train.shape}`")
    st.write(f"Test data shape: `{X_test.shape}`")

    # --- 6. Create Sequences for LSTM ---
    def create_sequences(X, y, time_steps=24):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    TIME_STEPS = 24
    X_seq_train, Y_seq_train = create_sequences(X_train, Y_train, TIME_STEPS)
    X_seq_test, Y_seq_test = create_sequences(X_test, Y_test, TIME_STEPS)

    st.write(f"Training sequences shape: `{X_seq_train.shape}`")
    st.write(f"Test sequences shape: `{X_seq_test.shape}`")

    # --- 7. Build and Train the LSTM Model ---
    st.subheader("🤖 LSTM Model Training")

    if st.button("Train LSTM Model"):
        # Lazy-import TensorFlow / Keras so the app can still start if TF is not installed
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        except Exception as e:
            st.error(
                "TensorFlow / Keras not available in this environment. "
                "Install it with `pip install tensorflow` to train the LSTM. "
                f"({e})"
            )
            st.stop()

        model_lstm_simple = Sequential([
            Input(shape=(TIME_STEPS, N_FEATURES)),
            LSTM(units=50, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])

        model_lstm_simple.compile(optimizer='adam', loss='mean_squared_error')

        with st.spinner('Training model... This may take a moment.'):
            history = model_lstm_simple.fit(
                X_seq_train, Y_seq_train,
                epochs=20,
                batch_size=32,
                validation_data=(X_seq_test, Y_seq_test),
                shuffle=False,
                verbose=0  # Set to 0 for cleaner output in Streamlit
            )
        st.success("✅ Model training complete!")

        # --- 8. Evaluate the Model ---
        st.subheader("📈 Model Evaluation")
        predictions_scaled = model_lstm_simple.predict(X_seq_test)
        predictions_real = target_scaler.inverse_transform(predictions_scaled)
        Y_test_real = target_scaler.inverse_transform(Y_seq_test)

        rmse = np.sqrt(mean_squared_error(Y_test_real, predictions_real))
        st.write(f"**Test RMSE:** `{rmse:.4f}` (degrees C)")

        # --- 9. Visualize Predictions ---
        st.subheader("📊 Prediction Visualization")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(Y_test_real, label='Actual Temperature', color='blue')
        ax.plot(predictions_real, label='Predicted Temperature', color='red', linestyle='--')
        ax.set_title('Temperature Prediction vs. Actual')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (C)')
        ax.legend()
        st.pyplot(fig)

else:
    st.info("Ensure `weatherHistory.csv` is in the same directory.")