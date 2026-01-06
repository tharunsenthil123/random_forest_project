import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")

st.title("🎗️ Breast Cancer Prediction")
st.write("Random Forest Classifier")

# -------------------------------------------------
# Upload CSV (SAFE for Streamlit Cloud)
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload breast-cancer.csv", type=["csv"])

if uploaded_file is None:
    st.info("Please upload the breast cancer dataset to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Target column (adjust if needed)
# -------------------------------------------------
TARGET = "diagnosis"   # common column: M / B

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Encode target if categorical (M/B)
if y.dtype == "object":
    y = y.map({"M": 1, "B": 0})

# -------------------------------------------------
# Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# Train Random Forest Classifier
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# -------------------------------------------------
# Accuracy
# -------------------------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {acc:.2f}")

# -------------------------------------------------
# User Input Section
# -------------------------------------------------
st.subheader("Enter Patient Feature Values")

input_data = []

for col in X.columns:
    value = st.number_input(f"{col}", value=float(X[col].mean()))
    input_data.append(value)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Cancer"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Malignant (Cancerous)")
    else:
        st.success("✅ Prediction: Benign (Non-Cancerous)")
