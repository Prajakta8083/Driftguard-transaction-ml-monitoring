"""
DriftGuard - Phase 1: Data & Baseline Establishment
=====================================================
Purpose: Load the Credit Card Fraud dataset, characterize the imbalance,
split the data in a way that respects time (fraud is not IID over time),
and save a "training distribution baseline" — a small JSON of per-feature
statistics that Phase 5 (drift detection) will diff live data against.

Run locally: python phase1_eda_baseline.py
Expects creditcard.csv in the same directory (download from Kaggle:
"Credit Card Fraud Detection" by mlg-ulb).
"""

import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------
df = pd.read_csv("creditcard.csv")
print(f"Shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean():.5%}")
print(f"Fraud count: {df['Class'].sum()} / {len(df)}")

# ---------------------------------------------------------------------
# 2. Imbalance-aware EDA
# ---------------------------------------------------------------------
# Don't just report accuracy-style stats. Report what actually matters
# for a 0.17% positive-rate problem: how separable are the classes,
# and does that separability hold up feature-by-feature?
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

feature_cols = [c for c in df.columns if c not in ("Class", "Time")]

separation_report = {}
for col in feature_cols:
    # Simple, interpretable separability signal: standardized mean difference
    pooled_std = np.sqrt((fraud[col].var() + normal[col].var()) / 2)
    smd = (fraud[col].mean() - normal[col].mean()) / (pooled_std + 1e-9)
    separation_report[col] = round(float(smd), 3)

top_features = sorted(separation_report.items(), key=lambda x: -abs(x[1]))[:5]
print("\nTop 5 features by class separation (standardized mean diff):")
for name, val in top_features:
    print(f"  {name}: {val}")

# ---------------------------------------------------------------------
# 3. Time-based split (NOT random split)
# ---------------------------------------------------------------------
# Why: 'Time' = seconds elapsed since the first transaction in the dataset.
# Fraud patterns drift over time (that's the whole premise of this project).
# A random split lets future fraud patterns leak into training, which
# silently inflates your reported metrics and is a classic mistake that
# interviewers specifically probe for.
df = df.sort_values("Time").reset_index(drop=True)
n = len(df)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

print(f"\nTrain: {len(train_df)} rows (fraud rate {train_df['Class'].mean():.5%})")
print(f"Val:   {len(val_df)} rows (fraud rate {val_df['Class'].mean():.5%})")
print(f"Test:  {len(test_df)} rows (fraud rate {test_df['Class'].mean():.5%})")

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# ---------------------------------------------------------------------
# 4. Save the training distribution baseline (drift detection input)
# ---------------------------------------------------------------------
# This is the artifact that makes Phase 5 possible. PSI and KS-tests both
# need a reference distribution to compare live data against — this IS
# that reference, computed once, frozen, and never silently recomputed.
baseline = {"features": {}}
for col in feature_cols:
    values = train_df[col].values
    # Store both raw stats (for KS-test, which needs distributional shape)
    # and binned histogram (for PSI, which is bucket-based by design)
    hist, bin_edges = np.histogram(values, bins=10)
    baseline["features"][col] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
        },
        "histogram_counts": hist.tolist(),
        "histogram_bin_edges": bin_edges.tolist(),
    }

baseline["n_train_rows"] = len(train_df)
baseline["train_fraud_rate"] = float(train_df["Class"].mean())

with open("training_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("\nSaved: train.csv, val.csv, test.csv, training_baseline.json")
