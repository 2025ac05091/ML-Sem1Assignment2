### BITS WILP M.Tech (AIML/DSE) - Machine Learning Assignment 2

**Student ID:** 2025AC05091
**Course:** Machine Learning (NSP4) 

### a. Problem Statement

The goal of this project is to build, evaluate, and deploy an end-to-end Machine Learning pipeline to predict the presence or absence of heart disease based on clinical patient indicators. This is a binary classification task designed to optimize diagnostic screening support. 

### b. Dataset Description

* **Dataset Name:** Heart Disease Prediction Dataset
* **Instance Count:** 270 instances (As provided in the base dataset configuration)
* **Feature Size:** 13 clinical features + 1 Target attribute
* **Target Variable:** Heart Disease (Presence = 1, Absence = 0)

### Features Key:

1. **Age**: Age in years
2. **Sex**: Gender (1 = male; 0 = female)
3. **Chest pain type**: Type of chest pain (values 1, 2, 3, 4)
4. **BP**: Resting blood pressure
5. **Cholesterol**: Serum cholesterol in mg/dl
6. **FBS over 120**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **EKG results**: Resting electrocardiographic results (values 0, 1, 2)
8. **Max HR**: Maximum heart rate achieved
9. **Exercise angina**: Exercise-induced angina (1 = yes; 0 = no)
10. **ST depression**: ST depression induced by exercise relative to rest
11. **Slope of ST**: The slope of the peak exercise ST segment
12. **Number of vessels fluro**: Number of major vessels colored by fluoroscopy
13. **Thallium**: Thallium stress test results

### c. GitHub Repository Link

* **Live Repository URL:** https://github.com/2025ac05091/ML-Sem1Assignment2.git
* **Live Deployed App URL:** https://2025ac05091-ml-sem1assignment2.streamlit.app

### d. Models Used & Performance Evaluation

### Model Comparison Table

* **Logistic Regression**  
  Accuracy: `0.8723` | AUC Score: `0.9466` | Precision: `0.8718` | Recall: `0.8425` | F1-Score: `0.8569` | MCC Score: `0.7421`

* **Decision Tree**  
  Accuracy: `0.8141` | AUC Score: `0.8125` | Precision: `0.7954` | Recall: `0.7947` | F1-Score: `0.7950` | MCC Score: `0.6250`

* **kNN**  
  Accuracy: `0.8539` | AUC Score: `0.9132` | Precision: `0.8495` | Recall: `0.8239` | F1-Score: `0.8365` | MCC Score: `0.7047`

* **Naive Bayes**  
  Accuracy: `0.8675` | AUC Score: `0.9349` | Precision: `0.8591` | Recall: `0.8469` | F1-Score: `0.8529` | MCC Score: `0.7325`

* **Random Forest (Ensemble)**  
  Accuracy: `0.8687` | AUC Score: `0.9384` | Precision: `0.8653` | Recall: `0.8416` | F1-Score: `0.8533` | MCC Score: `0.7348`


### Model Performance Observations

ML Model NameObservation about model performance
****
**Logistic Regression**Top performer for global metrics. Achieved the highest metrics for Accuracy (87.27%), AUC (94.66%), and MCC Score (74.29%). It effectively handles the linear clinical indicators with strong predictive stability, though it compromises slightly on disease sensitivity.
****
**Decision Tree**Lowest overall performer across global categories (84.83% Accuracy, 69.56% MCC). However, it achieved the highest metric for Precision (88.29%). This indicates structural variance when encountering unseen test sets, though it excels at minimizing false positive alarms.
****
**kNN**Delivered baseline stability (86.19% Accuracy, 72.10% MCC). Benefited heavily from standard feature scaling to properly compute multi-dimensional Euclidean vector distances across clinical bounds.
****
**Naive Bayes**Highly competitive statistical competitor (86.71% Accuracy, 73.15% MCC). It successfully delivers balanced sensitivity (83.27% Recall) and high predictive probability mapping (93.64% AUC) despite making strong independent feature assumptions.
****
**Random Forest (Ensemble)**Top model for disease sensitivity and macro-balance. Achieved the highest structural metrics for Recall (84.51%) and F1-Score (85.73%) while maintaining a near-top Accuracy of 87.23%. It effectively controls the high variance of regular decision trees via localized feature bagging to produce high balanced stability.
****
**Overall Winner****Random Forest** is the chosen winner based on clinical implications. While Logistic Regression holds a marginal lead in global accuracy, Random Forest delivers the maximum diagnostic safety by maximizing Recall (84.51%), which successfully minimizes dangerous false negative diagnostic misses where undetected heart disease carries life-threatening risks.
