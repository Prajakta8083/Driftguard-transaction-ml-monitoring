"""
DriftGuard - Phase 1: Baseline ML Model Comparison
====================================================
Pure ML experimentation, deliberately separate from the backend app.
No FastAPI, no database here — just: can we build a model comparison
story we can defend in an interview?

Expects train.csv / val.csv / test.csv (from the Phase 0 data script,
which used a TIME-BASED split — see the note in section 2 below for why
that decision carries forward here instead of re-splitting randomly).

Run: python model_comparison.py
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

feature_cols = [c for c in train_df.columns if c not in ("Class",)]
X_train, y_train = train_df[feature_cols], train_df["Class"]
X_val, y_val = val_df[feature_cols], val_df["Class"]
X_test, y_test = test_df[feature_cols], test_df["Class"]

print(f"Train fraud rate: {y_train.mean():.4%} ({y_train.sum()} / {len(y_train)})")
print(f"Val   fraud rate: {y_val.mean():.4%} ({y_val.sum()} / {len(y_val)})")
print(f"Test  fraud rate: {y_test.mean():.4%} ({y_test.sum()} / {len(y_test)})")

# ---------------------------------------------------------------------
# 2. Correlation with target (quick, since V1-V28 are already anonymized
#    PCA components with no real-world meaning to explore further)
# ---------------------------------------------------------------------
corr_with_target = train_df[feature_cols + ["Class"]].corr()["Class"].drop("Class")
top_corr = corr_with_target.abs().sort_values(ascending=False).head(5)
print("\nTop 5 features by |correlation| with Class:")
print(top_corr)

# ---------------------------------------------------------------------
# NOTE on stratification vs. time-based split (read before touching this)
# ---------------------------------------------------------------------
# "Stratified split" and "time-based split" solve DIFFERENT problems and
# cannot both be fully satisfied at once:
#   - Stratification (train_test_split(..., stratify=y)) guarantees each
#     split has the SAME class ratio, by randomly assigning rows —
#     which requires ignoring time order.
#   - Time-based split guarantees no future information leaks into
#     training, by preserving time order — which means class ratio per
#     split is whatever it happens to be in that time window.
#
# For fraud specifically, temporal leakage is the more dangerous failure
# mode (Phase 0's whole premise), so this project keeps the time-based
# split from the data-prep phase as the split of record. We do NOT
# re-shuffle it into a stratified split here.
#
# What stratification IS still used for below: SMOTE is applied only
# inside the training fold (never touching val/test), which is the
# imbalance-handling technique that actually matters at this stage —
# stratified splitting and SMOTE address different problems (split
# composition vs. training signal), and only the latter is compatible
# with keeping time order intact.
print(f"\nUsing existing time-based split "
      f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}).")
print("Not re-stratifying: would reintroduce temporal leakage. See comment above.")

# ---------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------
# V1-V28 are already PCA components (roughly comparable scale). Amount
# and Time are raw and need scaling for distance/gradient-based models
# (LogReg, Isolation Forest). Tree models (RF, XGBoost) don't need this,
# but we scale consistently so ONE preprocessing pipeline serves all
# four models — simpler to save/reuse in the backend later.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

# ---------------------------------------------------------------------
# 4. Imbalance handling — TWO approaches, kept separate for comparison
# ---------------------------------------------------------------------
# Approach A: class weighting — tell the model's loss function that
# minority-class errors cost more. No synthetic data, no data duplication.
# Approach B: SMOTE — synthesize new minority-class examples by
# interpolating between real fraud examples' feature vectors, so the
# model sees a less skewed class ratio during training.
#
# CRITICAL: SMOTE is fit only on X_train_scaled — never on val/test.
# Applying SMOTE before splitting, or to val/test, leaks synthetic
# near-duplicates across the split boundary and silently inflates
# reported performance. This is a very common, very serious bug.
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"\nAfter SMOTE: {y_train_smote.sum()} fraud / {len(y_train_smote)} total "
      f"(was {y_train.sum()} / {len(y_train)})")

# ---------------------------------------------------------------------
# 5. Model definitions (weighted variant + SMOTE variant per model,
#    where applicable — Isolation Forest is unsupervised and unaffected
#    by either technique, included here purely as an unsupervised
#    contrast point)
# ---------------------------------------------------------------------
fraud_rate = y_train.mean()

model_configs = {
    "LogReg (class-weighted)": (
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        X_train_scaled, y_train,
    ),
    "LogReg (SMOTE)": (
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        X_train_smote, y_train_smote,
    ),
    "RandomForest (class-weighted)": (
        RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=RANDOM_STATE),
        X_train_scaled, y_train,
    ),
    "RandomForest (SMOTE)": (
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        X_train_smote, y_train_smote,
    ),
    "XGBoost (class-weighted)": (
        XGBClassifier(
            scale_pos_weight=(1 - fraud_rate) / fraud_rate,
            eval_metric="aucpr", random_state=RANDOM_STATE,
        ),
        X_train_scaled, y_train,
    ),
    "XGBoost (SMOTE)": (
        XGBClassifier(eval_metric="aucpr", random_state=RANDOM_STATE),
        X_train_smote, y_train_smote,
    ),
    # Unsupervised — trained WITHOUT labels, contamination set to the
    # known training fraud rate as its prior on anomaly proportion.
    "IsolationForest (unsupervised)": (
        IsolationForest(contamination=fraud_rate, random_state=RANDOM_STATE),
        X_train_scaled, None,
    ),
}

# ---------------------------------------------------------------------
# 6. Train + evaluate on the held-out TEST set (touched exactly once)
# ---------------------------------------------------------------------
results = []
fitted_models = {}

for name, (model, X_fit, y_fit) in model_configs.items():
    if y_fit is not None:
        model.fit(X_fit, y_fit)
        y_score = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
    else:
        # Isolation Forest: fit unsupervised, score_samples gives higher
        # values for "more normal" points, so we flip sign for an
        # anomaly score comparable in direction to a fraud probability.
        model.fit(X_fit)
        y_score = -model.score_samples(X_test_scaled)
        y_pred = (model.predict(X_test_scaled) == -1).astype(int)

    fitted_models[name] = model
    results.append({
        "model": name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_test, y_score), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    })

comparison_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n" + "=" * 70)
print("MODEL COMPARISON (sorted by PR-AUC)")
print("=" * 70)
print(comparison_df[["model", "precision", "recall", "f1", "pr_auc"]].to_string(index=False))
print("\nConfusion matrices ([[TN, FP], [FN, TP]]):")
for r in results:
    print(f"  {r['model']}: {r['confusion_matrix']}")

# ---------------------------------------------------------------------
# 7. Pick the winner and save it + the preprocessing pipeline together
# ---------------------------------------------------------------------
best_name = comparison_df.iloc[0]["model"]
best_model = fitted_models[best_name]
print(f"\nBest model by PR-AUC: {best_name}")

pipeline_artifact = {
    "model": best_model,
    "scaler": scaler,
    "feature_cols": feature_cols,
    "model_name": best_name,
}
joblib.dump(pipeline_artifact, "fraud_model_pipeline.joblib")

with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved: fraud_model_pipeline.joblib, model_comparison_results.json")
