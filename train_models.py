import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load raw heart disease dataset 
file_path = r"/Users/shriramsankaran/Documents/Documents - Shriram’s Laptop/BITS-Pilani_Docs/Semester1/Assignments/MachineLearning/Assignment2/Heart_Disease_Prediction.csv"
df = pd.read_csv(file_path)
df = df.dropna()
print(f"Loaded dataset with {df.shape} rows and {df.shape} columns after dropping missing values.")

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

# Fit scaler ONLY on training data to avoid data leakage for final Streamlit inference
scaler = StandardScaler()
scaler.fit(X_train_raw)

# Setup Hyperparameter Tuning Dictionaries for GridSearchCV
param_grids = {
    "Logistic Regression": {
        "model__solver": ["lbfgs", "liblinear", "newton-cg"],
        "model__C": [0.001, 0.005, 0.01, 0.1, 1.0, 2, 3, 4, 5, 10]
},
    "Decision Tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [10, 15, 20], 
        "model__ccp_alpha": [0.0, 0.01, 0.02]
    },
    "K-Nearest Neighbor": {
        "model__n_neighbors": [3, 5, 7, 9],  # Fixed: Populated empty list
        "model__weights": ["uniform", "distance"]
    },
    "Naive Bayes": {
        "model__var_smoothing": np.logspace(0, -9, num=10)
    },
    "Random Forest": {
        "model__n_estimators": [20, 50, 100, 200, 300, 400],
        "model__max_depth": [5, 10, 20, None],
        "model__min_samples_split": [5, 10, 15, 20],
        "model__ccp_alpha": [0.0, 0.01, 0.02]
    }
}

# Base instantiated classifiers
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

final_optimized_models = {}

print("\n--- Starting 5-Fold Cross-Validation Parameter Search ---")

for name, model in base_models.items():
    # Build internal leakage-safe cross-validation pipeline 
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Run grid search optimized over accuracy score
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Fit across cross-validation groups using non-pre-scaled train features
    grid_search.fit(X_train_raw, y_train)
    
    print(f"\n🌟 {name} Optimized Details:")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV Accuracy Score: {grid_search.best_score_:.4f}")
    
    # Extract the optimal tuned inner model layer 
    final_optimized_models[name] = grid_search.best_estimator_.named_steps['model']

# Save only the optimized models and the reference scaler object
with open("models.pkl", "wb") as f:
    pickle.dump({"models": final_optimized_models, "scaler": scaler}, f)

print("\nOptimized model training complete. File 'models.pkl' saved.")
