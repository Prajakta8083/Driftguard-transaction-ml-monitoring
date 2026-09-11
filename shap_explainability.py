"""
DriftGuard - Phase 2: Explainability (SHAP)
=============================================
Loads the winning model + preprocessing pipeline saved by model_comparison.py,
and produces:
  1. Global feature importance (which features matter across ALL predictions)
  2. Local explanation for one specific flagged transaction (why THIS one)
  3. A summary (beeswarm) plot + a waterfall plot, saved as PNGs

Run: python shap_explainability.py
Expects: fraud_model_pipeline.joblib (from model_comparison.py) and test.csv
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load the saved pipeline + test data
# ---------------------------------------------------------------------
artifact = joblib.load("fraud_model_pipeline.joblib")
model = artifact["model"]
scaler = artifact["scaler"]
feature_cols = artifact["feature_cols"]
model_name = artifact["model_name"]

print(f"Loaded model: {model_name}")

test_df = pd.read_csv("test.csv")
X_test = test_df[feature_cols]
y_test = test_df["Class"]
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

# Use a manageable sample for SHAP computation — SHAP is expensive per row,
# and for a demo/interview artifact you don't need all of the test set,
# just enough to make the global pattern trustworthy.
sample_size = min(500, len(X_test_scaled))
X_sample = X_test_scaled.sample(sample_size, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. Pick the right SHAP explainer for the model type
# ---------------------------------------------------------------------
# TreeExplainer is exact and fast for tree ensembles (RF, XGBoost,
# IsolationForest). LinearExplainer is exact and fast for linear models
# (Logistic Regression). Using the generic, slow KernelExplainer for
# either of these would be both slower and less exact than necessary.
model_class = type(model).__name__

if model_class in ("XGBClassifier", "RandomForestClassifier", "IsolationForest"):
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    # SHAP's return shape for binary classifiers has changed across
    # versions, so handle BOTH formats rather than assuming one:
    #   - older shap: list of two (n_samples, n_features) arrays,
    #     one per class
    #   - newer shap: single (n_samples, n_features, n_classes) array
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
        expected_value = explainer.expected_value[1]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]  # positive ("fraud") class
        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        shap_values = shap_values_raw
        expected_value = explainer.expected_value
elif model_class == "LogisticRegression":
    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
else:
    raise ValueError(f"No explainer configured for model type: {model_class}")

# ---------------------------------------------------------------------
# 3. GLOBAL explanation — which features matter across ALL predictions
# ---------------------------------------------------------------------
# mean(|SHAP value|) per feature = average magnitude of that feature's
# push on the prediction, across every row in the sample. This answers
# "which features does the model rely on in general" — a portfolio-level
# question, not a per-transaction one.
mean_abs_shap = np.abs(shap_values).mean(axis=0)
global_importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)

print("\nTop 10 features by global importance (mean |SHAP value|):")
print(global_importance.head(10).to_string())

plt.figure()
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("shap_global_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_global_summary.png")

# ---------------------------------------------------------------------
# 4. LOCAL explanation — why THIS specific transaction was flagged
# ---------------------------------------------------------------------
# Find a transaction the model actually flagged as fraud, so the local
# explanation is meaningful (explaining a confident "not fraud" case is
# a less interesting story for a fraud analyst than explaining an alert).
if hasattr(model, "predict_proba"):
    scores = model.predict_proba(X_sample)[:, 1]
else:
    scores = -model.score_samples(X_sample)  # IsolationForest fallback

flagged_idx = int(np.argmax(scores))
print(f"\nExplaining sample row {flagged_idx} (model score: {scores[flagged_idx]:.4f})")
print("Feature values for this transaction:")
print(X_sample.iloc[flagged_idx].sort_values(ascending=False).head(10).to_string())

local_shap = shap_values[flagged_idx]
top_local = pd.Series(local_shap, index=feature_cols).sort_values(key=np.abs, ascending=False)
print("\nTop 5 features driving THIS prediction (signed SHAP value):")
print(top_local.head(5).to_string())

plt.figure()
explanation = shap.Explanation(
    values=local_shap,
    base_values=expected_value,
    data=X_sample.iloc[flagged_idx].values,
    feature_names=feature_cols,
)
shap.plots.waterfall(explanation, show=False)
plt.tight_layout()
plt.savefig("shap_local_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_local_waterfall.png")

print("\nDone. Global plot = model-wide behavior. Local plot = one transaction's story.")