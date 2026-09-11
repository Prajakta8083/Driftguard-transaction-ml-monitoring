"""
Model Loader
============
Loading a joblib pipeline (and building a SHAP explainer) is expensive —
disk I/O plus deserialization, and for tree models SHAP's TreeExplainer
does non-trivial setup work walking the tree structure. If this happened
inside the request handler, every single POST /transactions call would
pay that cost again, which is a real, measurable latency and CPU problem
under any real traffic. Loading once at startup and reusing the same
in-memory object for every request is the standard fix.

FastAPI's `app.state` is used to hold this singleton, populated once in
main.py's startup event, so every route accesses the SAME loaded model —
not a fresh one per request.
"""

import os

import joblib
import numpy as np
import pandas as pd
import shap

from app.config import settings


class ModelBundle:
    def __init__(self, pipeline_path: str):
        artifact = joblib.load(pipeline_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_cols = artifact["feature_cols"]
        self.model_version = artifact["model_name"]
        self.explainer = self._build_explainer()
        self.global_importance = self._compute_global_importance()

    def _build_explainer(self):
        model_class = type(self.model).__name__
        if model_class in ("XGBClassifier", "RandomForestClassifier", "IsolationForest"):
            return shap.TreeExplainer(self.model)
        elif model_class == "LogisticRegression":
            # LinearExplainer needs a background dataset; in a real deploy
            # this would be a stored sample of scaled training data. Kept
            # simple here — swap in a persisted background sample if the
            # winning model turns out to be Logistic Regression.
            return None
        return None

    def _compute_global_importance(self):
        """
        Computed ONCE at startup (not per-request) against a sample of the
        original test set, so the Explainability page has real, cheap-to-
        serve global feature importance instead of nothing, and instead of
        recomputing SHAP over a full dataset on every request.
        """
        if self.explainer is None or not os.path.exists(settings.original_test_data_path):
            return []
        test_df = pd.read_csv(settings.original_test_data_path)
        sample = test_df[self.feature_cols].sample(
            min(300, len(test_df)), random_state=42
        )
        sample_scaled = pd.DataFrame(self.scaler.transform(sample), columns=self.feature_cols)
        shap_out = self.explainer.shap_values(sample_scaled)
        if isinstance(shap_out, list):
            shap_values = shap_out[1]
        elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
            shap_values = shap_out[:, :, 1]
        else:
            shap_values = shap_out
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = sorted(
            zip(self.feature_cols, mean_abs.tolist()), key=lambda p: p[1], reverse=True
        )
        return [{"feature": f, "importance": round(v, 4)} for f, v in ranked]


_model_bundle: ModelBundle | None = None


def load_model_bundle() -> ModelBundle:
    """Call once, at app startup."""
    global _model_bundle
    _model_bundle = ModelBundle(settings.model_pipeline_path)
    return _model_bundle


def get_model_bundle() -> ModelBundle:
    """FastAPI dependency: returns the already-loaded singleton."""
    if _model_bundle is None:
        raise RuntimeError("Model not loaded — was startup event run?")
    return _model_bundle
