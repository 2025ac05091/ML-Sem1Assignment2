import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    matthews_corrcoef
)

# Set page configuration
st.set_page_config(page_title="2025AC05091 Multi Model App", layout="wide")
st.title("Multi-Model Classification Dashboard")
st.write("2025AC05091 Machine Learning Assignment 2")

# Load trained components (Contains models and scaler)
@st.cache_resource
def load_assets():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

assets = load_assets()
models = assets["models"]
scaler = assets["scaler"]

# Load test data to compute evaluation metrics live
@st.cache_data
def load_test_data():
    test_df = pd.read_csv("test_data.csv")
    X_test_raw = test_df.drop(columns=['Heart Disease'])
    y_test = test_df['Heart Disease']
    return X_test_raw, y_test

try:
    X_test_raw, y_test = load_test_data()
    
    # Transform test data using the saved training scaler
    # Passing a pure DataFrame with correct column headers to prevent UserWarning
    X_test_scaled = scaler.transform(X_test_raw)

    # Calculate evaluation metrics live across all models
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        # Get probabilities for the positive class (Presence = 1) to calculate AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc_val = roc_auc_score(y_test, y_prob)
        else:
            auc_val = np.nan

        metrics[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": auc_val,
            "Precision": precision_score(y_test, y_pred, average='binary'),
            "Recall": recall_score(y_test, y_pred, average='binary'),
            "F1-Score": f1_score(y_test, y_pred, average='binary'),
            "MCC Score": matthews_corrcoef(y_test, y_pred)
        }
except FileNotFoundError:
    st.error("Missing 'test_data.csv' in the directory. Please run your training script first.")
    st.stop()

# Sidebar for Model Selection
st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Choose a Classification Model", list(models.keys()))

# Section 1: Display Evaluation Metrics
st.subheader(f"Performance Metrics: {selected_model_name}")
col1, col2, col3, col4, col5, col6 = st.columns(6)
model_metrics = metrics[selected_model_name]

col1.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
col2.metric("AUC Score", f"{model_metrics['AUC Score']:.4f}" if not np.isnan(model_metrics['AUC Score']) else "N/A")
col3.metric("Precision", f"{model_metrics['Precision']:.4f}")
col4.metric("Recall", f"{model_metrics['Recall']:.4f}")
col5.metric("F1-Score", f"{model_metrics['F1-Score']:.4f}")
col6.metric("MCC Score", f"{model_metrics['MCC Score']:.4f}")

# Comparative Table
st.subheader("Model Comparison Summary")
st.dataframe(pd.DataFrame(metrics).T.style.highlight_max(axis=0, color="#d4edda"))

# Section 2: Interactive Real-Time Prediction Input
st.subheader("Live Test-Data Prediction")
st.write("Adjust the features below to view the selected model's instant prediction.")

feature_labels = list(X_test_raw.columns)
user_inputs = {}
cols = st.columns(4) 

for i, label in enumerate(feature_labels):
    with cols[i % 4]:
        default_val = float(X_test_raw[label].median())
        
        # Keep numeric values cleanly separated
        val = st.number_input(f"{label}", value=default_val, key=f"feat_{label}")
        user_inputs[label] = val

# Run Prediction Logic
if st.button("Run Prediction Inference"):

    input_df = pd.DataFrame([user_inputs])
    input_df = input_df[feature_labels] # Forces exact order match with training schema
    
    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_df) 
    
    chosen_model = models[selected_model_name]
    prediction = chosen_model.predict(scaled_input)
    probabilities = chosen_model.predict_proba(scaled_input)
    
    # Map numerical values back to explicit assignment labels
    outcome_mapped = "Presence (Heart Disease)" if prediction[0] == 1 else "Absence (Healthy)"
    
    st.success(f"Predicted Class Target Outcome: **{outcome_mapped}**")
    
    # Format prediction confidences cleanly
    prob_df = pd.DataFrame(probabilities, columns=["Absence Probability", "Presence Probability"])
    st.write("Prediction Confidence Probabilities:", prob_df)
