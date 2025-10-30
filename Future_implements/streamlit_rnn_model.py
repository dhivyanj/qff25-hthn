import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- Machine Learning & Data Science ---
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import shap

# --- Quantum (PennyLane) ---
try:
    import pennylane as qml
    from pennylane.templates.layers import StronglyEntanglingLayers
    from pennylane.templates.state_preparations import AmplitudeEmbedding
except ImportError:
    st.error("PennyLane not found. Please run 'pip install pennylane'")

# --- Deep Learning (TensorFlow) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from sklearn.metrics import mean_squared_error
except ImportError:
    st.warning("TensorFlow not found. The LSTM Forecaster tab will be disabled. Please 'pip install tensorflow'")

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Hybrid Quantum-Classical AI Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🤖 Hybrid Quantum-Classical AI Laboratory")
st.write("An integrated app to compare Quantum and Classical classifiers and run time-series forecasts.")

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def check_libs():
    """Check if all required libraries are installed and return status."""
    libs = {
        "pennylane": "pennylane" in globals(),
        "tensorflow": "tensorflow" in globals(),
        "shap": "shap" in globals()
    }
    return libs

lib_status = check_libs()

# ==============================================================================
# --- TAB 1: QUANTUM VS CLASSICAL CLASSIFICATION ---
# ==============================================================================

def render_classification_tab():
    st.header("🔬 Quantum vs. Classical Classification")
    st.write("Compare a PennyLane Quantum Support Vector Classifier (QSVC) against a classical Random Forest on the same dataset.")

    # --- Data Source Selection ---
    st.subheader("1. Data Source")
    data_source = st.radio(
        "Choose data for classification",
        ("Use Default Mock Data", "Upload your own CSV"),
        horizontal=True,
        help="The default data is a 3-class mock dataset (Solar, Wind, Hydro) generated per your notebook."
    )

    X, y, df, labels, N_SAMPLES, N_FEATURES = None, None, None, None, 0, 0

    if data_source == "Use Default Mock Data":
        with st.spinner("Generating mock data..."):
            X, y, labels = generate_mock_data(days=150) # Smaller for speed
            N_SAMPLES, N_FEATURES = X.shape
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(N_FEATURES)])
            df['target'] = y
            st.dataframe(df.head())

    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                target_col = st.selectbox(
                    "Select the target/label column",
                    df.columns,
                    index=len(df.columns)-1
                )
                
                feature_cols = st.multiselect(
                    "Select the feature columns",
                    [col for col in df.columns if col != target_col],
                    default=[col for col in df.columns if col != target_col and pd.api.types.is_numeric_dtype(df[col])]
                )

                if not feature_cols:
                    st.warning("Please select at least one feature column.")
                elif not target_col:
                    st.warning("Please select a target column.")
                else:
                    # Pre-process uploaded data
                    X_raw = df[feature_cols].values
                    y_raw = df[target_col].values
                    
                    # Scale features
                    X = MinMaxScaler().fit_transform(X_raw)
                    
                    # Encode labels
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)
                    labels = le.classes_
                    
                    N_SAMPLES, N_FEATURES = X.shape
                    
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                return

    if df is None:
        st.info("Please generate or upload data to proceed.")
        return

    st.success(f"Data loaded: {N_SAMPLES} samples, {N_FEATURES} features, {len(np.unique(y))} classes.")

    # --- Model Parameters ---
    st.subheader("2. Model Parameters")
    n_qubits = N_FEATURES
    
    # Ensure n_qubits is at least 1, even if 0 features
    if n_qubits == 0:
        st.error("No features selected. Cannot run models.")
        return
        
    n_layers = st.slider("Number of Quantum Layers (for QRC)", 1, 5, 2)
    
    # Qubit check for quantum models
    if N_FEATURES > 10:
        st.warning(f"Warning: {N_FEATURES} features requires {N_FEATURES} qubits. This will be *very* slow. Consider feature reduction (PCA).")
        n_qubits = 10 # Cap qubits for performance
        
    q_dev = qml.device("default.qubit", wires=n_qubits)

    # --- Run Analysis ---
    st.subheader("3. Run Analysis")
    if st.button("Train Models & Compare", type="primary"):
        if X is None or y is None:
            st.error("No data available to train.")
            return

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        quantum_results = {}
        classical_results = {}
        
        with st.status("🚀 Running Hybrid Analysis...", expanded=True) as status:
            try:
                # --- 1. Quantum Model (QRC + QSVC) ---
                status.update(label="Running Quantum Reservoir Computer (QRC)...")
                st.write("Step 1/5: Running Quantum Reservoir Computer (QRC) for feature extraction...")
                qrc_train_features = run_qrc_model(X_train, n_qubits, n_layers, q_dev)
                qrc_test_features = run_qrc_model(X_test, n_qubits, n_layers, q_dev)

                status.update(label="Running PCA on Quantum features...")
                st.write("Step 2/5: Running PCA on Quantum features...")
                X_train_pca, X_test_pca, pca_model = run_pca(qrc_train_features, qrc_test_features, n_components=2)
                
                status.update(label="Training Quantum SVC...")
                st.write("Step 3/5: Training Quantum SVC...")
                qsvc = SVC(kernel='rbf', probability=True) # Use classical RBF kernel on QRC features
                qsvc.fit(X_train_pca, y_train)
                y_pred_q = qsvc.predict(X_test_pca)

                quantum_results = {
                    "model": qsvc,
                    "y_pred": y_pred_q,
                    "accuracy": accuracy_score(y_test, y_pred_q),
                    "f1": f1_score(y_test, y_pred_q, average='weighted'),
                    "pca_model": pca_model,
                    "X_test_pca": X_test_pca,
                    "labels": labels
                }

                # --- 2. Classical Model (Random Forest) ---
                status.update(label="Training Classical Random Forest...")
                st.write("Step 4/5: Training Classical Random Forest...")
                # We train the classical model on the *original* features for a fair comparison
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                y_pred_c = rf.predict(X_test)
                
                status.update(label="Generating SHAP analysis...")
                st.write("Step 5/5: Generating SHAP analysis...")
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_test)
                
                classical_results = {
                    "model": rf,
                    "y_pred": y_pred_c,
                    "accuracy": accuracy_score(y_test, y_pred_c),
                    "f1": f1_score(y_test, y_pred_c, average='weighted'),
                    "explainer": explainer,
                    "shap_values": shap_values,
                    "X_test": X_test,
                    "feature_names": [f'feature_{i}' for i in range(N_FEATURES)]
                }
                
                status.update(label="Analysis Complete!", state="complete")

            except Exception as e:
                status.update(label="Error during analysis", state="error")
                st.exception(e)
                return

        # --- 4. Display Results ---
        st.subheader("4. Results Dashboard")
        
        col1, col2 = st.columns(2)

        with col1:
            st.header("⚛️ Quantum Model (QRC+SVC)")
            if quantum_results:
                st.metric("Accuracy", f"{quantum_results['accuracy']:.2%}")
                st.metric("F1 Score (Weighted)", f"{quantum_results['f1']:.3f}")
                
                st.subheader("Quantum Clustering (PCA)")
                fig_pca = plot_pca(
                    quantum_results['X_test_pca'],
                    y_test,
                    quantum_results['labels']
                )
                st.pyplot(fig_pca)
                
                st.subheader("Confusion Matrix")
                fig_cm_q, ax_cm_q = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(
                    y_test, quantum_results['y_pred'],
                    display_labels=quantum_results['labels'],
                    ax=ax_cm_q,
                    cmap='Blues'
                )
                st.pyplot(fig_cm_q)

        with col2:
            st.header("💻 Classical Model (Random Forest)")
            if classical_results:
                st.metric("Accuracy", f"{classical_results['accuracy']:.2%}")
                st.metric("F1 Score (Weighted)", f"{classical_results['f1']:.3f}")

                st.subheader("SHAP Analysis (Feature Importance)")
                if "shap" in globals():
                    fig_shap, ax_shap = plt.subplots()
                    shap.summary_plot(
                        classical_results['shap_values'],
                        classical_results['X_test'],
                        plot_type="bar",
                        class_names=labels,
                        feature_names=classical_results['feature_names'],
                        show=False
                    )
                    st.pyplot(fig_shap)
                else:
                    st.info("SHAP library not installed. Skipping plot.")
                
                st.subheader("Confusion Matrix")
                fig_cm_c, ax_cm_c = plt.subplots()
                ConfusionMatrixDisplay.from_predictions(
                    y_test, classical_results['y_pred'],
                    display_labels=labels,
                    ax=ax_cm_c,
                    cmap='Greens'
                )
                st.pyplot(fig_cm_c)


# --- Helper Functions for Classification Tab ---

@st.cache_data
def generate_mock_data(days=365, n_features=4):
    """Generates mock weather/sensor data for classification."""
    np.random.seed(42)
    X = np.random.rand(days, n_features)
    
    # Create somewhat separable classes
    def get_label(row):
        if row[0] > 0.6 and row[1] > 0.6:
            return "Solar" # High feature 0 & 1
        elif row[2] > 0.7 or row[3] < 0.3:
            return "Wind" # High feature 2 or low feature 3
        else:
            return "Hydro" # Everything else
            
    y = np.array([get_label(row) for row in X])
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    X_scaled = MinMaxScaler().fit_transform(X)
    
    return X_scaled, y_encoded, le.classes_

# --- QRC (Quantum Reservoir Computer) Functions ---

def qrc_circuit(params, wires, features):
    """The QRC circuit template."""
    qml.broadcast(qml.RX, pattern=[[i] for i in wires], parameters=features, wires=wires)
    StronglyEntanglingLayers(params, wires=wires)

@st.cache_resource(show_spinner="Creating QRC Node...")
def get_qrc_node(n_qubits, n_layers, _q_dev):
    """
    Returns a QNode for the QRC.
    We cache this function based on the _q_dev,
    which is itself cached.
    """
    wires = list(range(n_qubits))
    
    @qml.qnode(_q_dev, interface='autograd')
    def qrc_qnode(inputs, params):
        qrc_circuit(params, wires=wires, features=inputs)
        # Return expectation value for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in wires]
        
    return qrc_qnode, wires

@st.cache_data(show_spinner="Running QRC model...")
def run_qrc_model(_features, n_qubits, n_layers, _q_dev):
    """
    Processes features through the Quantum Reservoir Computer.
    """
    # Initialize random parameters for the QRC layers
    np.random.seed(42)
    shape = StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    params = np.random.random(shape, requires_grad=False)
    
    qrc_qnode, wires = get_qrc_node(n_qubits, n_layers, _q_dev)
    
    # Pad features if n_features < n_qubits
    n_features = _features.shape[1]
    if n_features < n_qubits:
        padding = np.zeros((_features.shape[0], n_qubits - n_features))
        _features = np.hstack([_features, padding])
    # Truncate features if n_features > n_qubits
    elif n_features > n_qubits:
        _features = _features[:, :n_qubits]

    # Run each sample through the QRC
    reservoir_states = []
    for x in _features:
        state = qrc_qnode(x, params)
        reservoir_states.append(state)
        
    return np.array(reservoir_states)

@st.cache_data(show_spinner="Running PCA...")
def run_pca(X_train, X_test, n_components=2):
    """Applies PCA to the data."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

def plot_pca(X_pca, y, labels):
    """Generates a scatter plot of the PCA components."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ax.scatter(
            X_pca[y == i, 0],
            X_pca[y == i, 1],
            alpha=0.7,
            label=label
        )
    ax.set_title("Quantum Feature Clustering (PCA)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    return fig


# ==============================================================================
# --- TAB 2: LSTM TIME SERIES FORECASTER ---
# ==============================================================================

def render_lstm_tab():
    st.header("📈 LSTM Time Series Forecaster")
    st.write("This app uses an LSTM model to predict temperature based on historical weather data (your original `model.py` and `streamlit_app.py`).")
    
    # --- 1. Load Data ---
    try:
        df_full = load_weather_data()
    except FileNotFoundError:
        st.error(f"Error: Could not find `weatherHistory.csv`. Please make sure it's in the same directory.")
        return
    except Exception as e:
        st.error(f"Error loading `weatherHistory.csv`: {e}")
        return

    if df_full is not None:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df_full.head())

        # --- 3. Select Features and Target ---
        N_FEATURES = 3
        features = ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)']
        target = 'Temperature (C)'

        # Drop rows where our target or features are missing
        df_clean = df_full[features + [target]].dropna()

        # --- 4. Prepare Data for LSTM ---
        if st.checkbox("Show data preparation details?", value=False):
            st.write(f"**Features (X):** {features}")
            st.write(f"**Target (Y):** {target}")
            st.write("Data is split 80/20 for training/testing.")
            st.write("Features and Target are scaled independently using MinMaxScaler.")
        
        # --- 5. Split Data ---
        split_ratio = 0.8
        split_index = int(len(df_clean) * split_ratio)
        df_train = df_clean.iloc[:split_index]
        df_test = df_clean.iloc[split_index:]

        # --- 6. Scale Data ---
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        X_train_scaled = feature_scaler.fit_transform(df_train[features])
        X_test_scaled = feature_scaler.transform(df_test[features])
        
        Y_train_scaled = target_scaler.fit_transform(df_train[[target]])
        Y_test_scaled = target_scaler.transform(df_test[[target]])

        # --- 7. Create Sequences ---
        time_steps = st.slider("Select Time Steps (History)", 1, 48, 24,
                               help="How many hours of history to use for each prediction.")

        X_seq_train, Y_seq_train = create_sequences(X_train_scaled, Y_train_scaled, time_steps)
        X_seq_test, Y_seq_test = create_sequences(X_test_scaled, Y_test_scaled, time_steps)
        
        st.write(f"Training sequences shape: `{X_seq_train.shape}`")
        st.write(f"Test sequences shape: `{X_seq_test.shape}`")

        # --- 8. Build and Train Model ---
        st.subheader("🧠 Model Training")
        epochs = st.number_input("Select number of epochs", 1, 100, 10)

        if st.button("Train LSTM Model", type="primary"):
            
            model_lstm = get_lstm_model(time_steps, N_FEATURES)
            
            with st.spinner('Training model... This may take a moment.'):
                history = model_lstm.fit(
                    X_seq_train, Y_seq_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_seq_test, Y_seq_test),
                    shuffle=False,
                    verbose=0, # Set to 0 for cleaner output
                    callbacks=[StreamlitCallback(st.empty())] # Custom callback for progress
                )
            st.success("✅ Model training complete!")

            # --- 9. Evaluate the Model ---
            st.subheader("📈 Model Evaluation")
            predictions_scaled = model_lstm.predict(X_seq_test)
            predictions_real = target_scaler.inverse_transform(predictions_scaled)
            Y_test_real = target_scaler.inverse_transform(Y_seq_test)

            rmse = np.sqrt(mean_squared_error(Y_test_real, predictions_real))
            st.metric("Test Root Mean Squared Error (RMSE)", f"{rmse:.4f} °C")

            # --- 10. Visualize Predictions ---
            st.subheader("📊 Prediction Visualization")
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(Y_test_real, label='Actual Temperature', color='blue')
            ax.plot(predictions_real, label='Predicted Temperature', color='red', linestyle='--')
            ax.set_title('Temperature Prediction vs. Actual')
            ax.set_xlabel('Time (Test Set samples)')
            ax.set_ylabel('Temperature (C)')
            ax.legend()
            st.pyplot(fig)

            # --- 11. Visualize Loss ---
            st.subheader("📉 Training Loss")
            fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
            ax_loss.plot(history.history['loss'], label='Training Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.set_title('Model Loss Over Epochs')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss (MSE)')
            ax_loss.legend()
            st.pyplot(fig_loss)

# --- Helper Functions for LSTM Tab ---

class StreamlitCallback(tf.keras.callbacks.Callback):
    """Custom callback to update Streamlit during training."""
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        epoch_str = f"Epoch {epoch+1}/{self.params['epochs']}"
        loss_str = f"Loss: {loss:.4f}"
        val_loss_str = f"Val Loss: {val_loss:.4f}"
        self.progress_bar.info(f"{epoch_str} | {loss_str} | {val_loss_str}")

@st.cache_data(show_spinner="Loading weather data...")
def load_weather_data():
    """Loads and pre-processes the weatherHistory.csv file."""
    df_full = pd.read_csv('weatherHistory.csv')
    df_full['Formatted Date'] = pd.to_datetime(df_full['Formatted Date'], utc=True)
    df_full = df_full.set_index('Formatted Date').sort_index()
    df_full = df_full[~df_full.index.duplicated(keep='first')]
    df_full = df_full.resample('h').interpolate(method='time')
    return df_full

@st.cache_data
def create_sequences(_X, _Y, _time_steps=1):
    """Converts data into sequences for LSTM."""
    Xs, ys = [], []
    for i in range(len(_X) - _time_steps):
        v = _X[i:(i + _time_steps)]
        Xs.append(v)
        ys.append(_Y[i + _time_steps])
    return np.array(Xs), np.array(ys)

@st.cache_resource(show_spinner="Building LSTM model...")
def get_lstm_model(_time_steps, _n_features):
    """Builds and compiles the LSTM model."""
    model_lstm_simple = Sequential([
        Input(shape=(_time_steps, _n_features)),
        LSTM(units=50, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, activation='relu'),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model_lstm_simple.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model_lstm_simple

# ==============================================================================
# --- MAIN APP LAYOUT (TABS) ---
# ==============================================================================

# Create tabs
tab_list = []
if lib_status["pennylane"] and lib_status["shap"]:
    tab_list.append("Quantum vs. Classical Classification")
if lib_status["tensorflow"]:
    tab_list.append("LSTM Temperature Forecaster")

if not tab_list:
    st.error("All major libraries (PennyLane, TensorFlow, SHAP) are missing. Please install them.")
else:
    tabs = st.tabs(tab_list)
    
    # Assign content to tabs
    tab_map = {
        "Quantum vs. Classical Classification": (render_classification_tab, "pennylane" in globals() and "shap" in globals()),
        "LSTM Temperature Forecaster": (render_lstm_tab, "tensorflow" in globals())
    }
    
    tab_index = 0
    if "Quantum vs. Classical Classification" in tab_list:
        with tabs[tab_index]:
            render_classification_tab()
        tab_index += 1
        
    if "LSTM Temperature Forecaster" in tab_list:
        with tabs[tab_index]:
            render_lstm_tab()
        tab_index += 1

# Sidebar with library info
st.sidebar.title("🛠️ Environment Status")
st.sidebar.info(
    f"PennyLane (Quantum): {'✅ Found' if lib_status['pennylane'] else '❌ Not Found'}\n\n"
    f"TensorFlow (LSTM): {'✅ Found' if lib_status['tensorflow'] else '❌ Not Found'}\n\n"
    f"SHAP (Analysis): {'✅ Found' if lib_status['shap'] else '❌ Not Found'}\n\n"
)
if not all(lib_status.values()):
    st.sidebar.warning("Some tabs may be disabled. Please install all required libraries from `requirements.txt`.")
