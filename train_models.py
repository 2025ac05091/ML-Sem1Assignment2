import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load raw heart disease dataset 
file_path = r"/Users/shriramsankaran/Documents/Documents - Shriram’s Laptop/BITS-Pilani_Docs/Semester1/Assignments/MachineLearning/Assignment2/Heart_Disease_Prediction.csv"
df = pd.read_csv(file_path)
df = df.dropna()
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns after dropping missing values.")
# Separate Features and Target
X = df.drop(columns=['Heart Disease']) 

y = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# Split raw data BEFORE scaling 
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save isolated test data to a CSV
test_df = X_test_raw.copy()
test_df['Heart Disease'] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved 'test_data.csv' successfully.")

# Fit scaler ONLY on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

# Setup the 6 mandatory assignment models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    # "Support Vector Machine": SVC(probability=True, random_state=42)
}

# Train models on training partition only
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# Save only the models and the scaler object)
with open("models.pkl", "wb") as f:
    pickle.dump({"models": models, "scaler": scaler}, f)

print("Model training complete. File 'models.pkl' saved.")
