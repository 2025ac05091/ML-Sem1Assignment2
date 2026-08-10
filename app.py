import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# Set page configuration
st.set_page_config(page_title="2025AC05091 Heart Disease Prediction App", layout="wide")
st.title("Heart Disease Prediction Dashboard")
st.write("2025AC05091 Machine Learning Assignment 2")

# Load trained components (Contains models and scaler)
@st.cache_resource
def load_assets():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
    models = assets["models"]
    scaler = assets["scaler"]
except FileNotFoundError:
    st.error("Missing 'models.pkl' file. Please run your training script first.")
    st.stop()

# Main Page Configuration Setup (Stacked Layout)
st.subheader("Dashboard Configuration")

dropdown_col, _ = st.columns([1, 3])
with dropdown_col:
    selected_model_name = st.selectbox("1. Choose a Classification Model", list(models.keys()))

config_container_col, _ = st.columns([3, 2])
with config_container_col:
    # 1. Constrained width file uploader element
    uploaded_file = st.file_uploader("2. Upload Test Dataset (CSV) [Optional - Defaults to preloaded data]", type=["csv"])

# Extract feature names from scaler to align data frames
feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None

# Pre-loading Logic: Use uploaded file if present, otherwise fall back to local default file
test_df = None
is_using_default = False

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
else:
    try:
        # Automatically load the default test file from the local path
        test_df = pd.read_csv("test_data.csv")
        is_using_default = True
    except FileNotFoundError:
        with config_container_col:
            st.warning("No custom file uploaded, and default 'test_data.csv' was not found locally.")

# Process data if any dataframe is successfully loaded (either uploaded or default)
if test_df is not None:
    try:
        # Show status indicator for dataset context
        with config_container_col:
            if is_using_default:
                st.info("Running dashboard using pre-loaded default 'test_data.csv'.")
            else:
                st.success("Running dashboard using your uploaded dataset.")
                
        # Verify target variable existence
        target_col = 'Heart Disease'
        if target_col not in test_df.columns:
            st.error(f"The test dataset must contain the target column: '{target_col}'")
            st.stop()
            
        X_test_raw = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Explicit target mapping if it contains string values
        if y_test.dtype == 'object':
            y_test = y_test.map({'Presence': 1, 'Absence': 0})
        
        # Calculate dynamic evaluation metrics for all models
        metrics = {}
        for name, model in models.items():
            # Align features using exact name order to suppress scikit-learn warnings
            X_test_scaled = scaler.transform(X_test_raw[feature_names] if feature_names is not None else X_test_raw)
            y_pred = model.predict(X_test_scaled)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                auc_val = roc_auc_score(y_test, y_prob)
            else:
                auc_val = np.nan
                
            metrics[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC Score": auc_val,
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "MCC Score": matthews_corrcoef(y_test, y_pred)
            }
            
        st.markdown("---")
        # Display Comparative Table Summary
        st.subheader("Model Comparison Summary")
        st.dataframe(pd.DataFrame(metrics).T.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)
        
        st.subheader(f"Confusion Matrix: {selected_model_name}")
        chosen_model = models[selected_model_name]
        X_test_scaled = scaler.transform(X_test_raw[feature_names] if feature_names is not None else X_test_raw)
        y_pred_selected = chosen_model.predict(X_test_scaled)
        
        cm = confusion_matrix(y_test, y_pred_selected)
        
        fig, ax = plt.subplots(figsize=(2.5, 2))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Absence', 'Presence'], 
                    yticklabels=['Absence', 'Presence'], 
                    annot_kws={"size": 9}, ax=ax)
        

        plt.ylabel('Actual', fontsize=8)
        plt.xlabel('Predicted', fontsize=8)
        ax.tick_params(labelsize=8)
        
        matrix_col, _ = st.columns([1, 3])
        with matrix_col:
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error processing dataset: {e}")


# Interactive Real-Time Prediction Input (Always available)
st.markdown("---")
st.subheader("Live Test-Data Prediction")
st.write(f"Adjust the features below to view the selected model's ({selected_model_name}) instant prediction.")

# Dynamic input form generation matching exactly 12 attributes
features = []
cols = st.columns(4)

labels = feature_names if feature_names is not None else [f"Feature {i+1}" for i in range(12)]

for i, label in enumerate(labels):
    with cols[i % 4]:
        val = st.number_input(f"{label}", min_value=0.0, max_value=5000.0, value=10.0 * (i+1), key=f"input_{i}")
        features.append(val)

# Run Prediction Logic
if st.button("Run Prediction Inference"):
    # Convert input list directly to DataFrame with original training feature headers
    input_df = pd.DataFrame([features], columns=labels)
    scaled_input = scaler.transform(input_df)
    
    chosen_model = models[selected_model_name]
    prediction = chosen_model.predict(scaled_input)
    
    # Class outcome interpretation
    outcome_text = "Presence (1)" if prediction == 1 else "Absence (0)"
    st.success(f"Predicted Class Target Outcome: **{outcome_text}**")
    
    if hasattr(chosen_model, "predict_proba"):
        probabilities = chosen_model.predict_proba(scaled_input)
        st.write("Prediction Confidence Probabilities:")
        prob_df = pd.DataFrame(probabilities, columns=['Absence (0)', 'Presence (1)'])
        st.dataframe(prob_df)
