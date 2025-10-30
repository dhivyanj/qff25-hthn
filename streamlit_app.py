# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Page setup ---
st.set_page_config(page_title="Kaggle Dashboard", layout="wide")

st.title("🚀 Kaggle ML Dashboard")
st.write("A minimal Streamlit dashboard for quick EDA and prediction.")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --- Basic EDA ---
    st.subheader("🔍 Quick EDA")
    st.write(df.describe())

    # --- Visualization ---
    col = st.selectbox("Select a column to visualize", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    # --- Model training (example) ---
    st.subheader("🤖 Quick ML Demo (Random Forest)")

    target = st.selectbox("Select target column", df.columns)
    features = st.multiselect(
        "Select feature columns", [c for c in df.columns if c != target]
    )

    if st.button("Train Model"):
        X = df[features].select_dtypes(include=['number']).fillna(0)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"✅ Model trained! Accuracy: {acc:.2f}")

else:
    st.info("👆 Upload a CSV file to get started.")
