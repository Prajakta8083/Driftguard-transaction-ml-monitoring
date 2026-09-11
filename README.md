DriftGuard** is an end-to-end **fraud detection and machine learning monitoring platform** designed to detect suspicious credit card transactions and monitor model performance in real-world usage.

The system uses a **Random Forest classifier with SMOTE** to handle the severe class imbalance in the credit card fraud dataset. Users can submit transaction details such as **Time, Amount, and anonymized V1–V28 features**, after which the backend generates a fraud probability and classification.

Key Features
* Live Fraud Scoring** — Score transactions in real time.
* Random Forest + SMOTE** — Handles highly imbalanced fraud data.
* Prediction History** — Tracks previous predictions, probabilities, timestamps, and model versions.
* Review Queue** — Allows analysts to review and confirm flagged transactions.
* Model Health** — Monitors Precision, Recall and F1 after confirmed outcomes become available.
* Explainability** — Shows global feature importance and supports transaction-level explanations.
* Data Drift Detection** — Uses **Population Stability Index (PSI)** to identify changes between training and recent transaction distributions.
* Model Report** — Displays dataset statistics and the currently deployed model.
* API Integration** — Connects the frontend dashboard with the ML backend.

ML Pipeline

Credit Card Dataset
        ↓
Data Preprocessing
        ↓
Class Imbalance Handling
        ↓
SMOTE / Class Weighting
        ↓
Model Training & Comparison
        ↓
PR-AUC Evaluation
        ↓
Random Forest + SMOTE
        ↓
Backend API
        ↓
DriftGuard Dashboard
        ↓
Fraud Probability + Decision
        ↓
Monitoring & Explainability

Tech Stack
Machine Learning:** Python, Scikit-learn, Random Forest, SMOTE
Backend: Python, REST API
Frontend: Web Dashboard
Monitoring: Precision, Recall, F1, PSI
Dataset: Credit Card Fraud Detection Dataset
One-line tagline for GitHub:

DriftGuard — Real-time fraud detection with explainable ML, model health monitoring, and data drift detection.
