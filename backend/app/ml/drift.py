"""
Drift Detection
===============
Compares LIVE incoming feature distributions against the FROZEN training
baseline saved in Phase 0 (training_baseline.json). Two methods, because
they catch different kinds of shift:

  - PSI (Population Stability Index): bucket-based, works directly off
    the histogram we already saved — cheap, standard in credit-risk/fraud
    industry monitoring, gives one interpretable number per feature.
  - KS-test (Kolmogorov-Smirnov): compares full distribution shape, more
    sensitive to subtle shifts PSI's fixed buckets might miss.

Design note: the baseline stores histogram bin COUNTS, not raw individual
training values (deliberate — avoids retaining a full copy of training
data in a monitoring artifact). For the KS-test, which technically wants
raw samples, we reconstruct an approximate sample by repeating each bin's
midpoint according to its stored count. This is a documented approximation,
not a fudge — worth stating plainly in an interview rather than glossing
over.
"""

import json
import numpy as np
from scipy import stats


def load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _psi(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    # Convert to proportions; add epsilon to avoid log(0) / divide-by-zero
    # for empty bins, which would otherwise crash on real, sparse data.
    eps = 1e-6
    expected_pct = expected_counts / (expected_counts.sum() + eps) + eps
    actual_pct = actual_counts / (actual_counts.sum() + eps) + eps
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def _reconstruct_sample_from_histogram(counts: list, bin_edges: list) -> np.ndarray:
    counts = np.array(counts)
    bin_edges = np.array(bin_edges)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Cap total reconstructed points to keep KS-test cheap even if the
    # original training set was large.
    scale = min(1.0, 2000 / max(counts.sum(), 1))
    scaled_counts = np.maximum((counts * scale).astype(int), 0)
    return np.repeat(midpoints, scaled_counts)


def compute_drift_for_feature(
    feature_name: str,
    live_values: np.ndarray,
    baseline: dict,
    psi_threshold: float,
    ks_pvalue_threshold: float,
) -> dict:
    feat_baseline = baseline["features"][feature_name]
    hist_counts = np.array(feat_baseline["histogram_counts"])
    bin_edges = np.array(feat_baseline["histogram_bin_edges"])

    # Bucket live values into the SAME bin edges as training — required
    # for PSI to be a fair comparison (different bucketing = apples to oranges).
    live_counts, _ = np.histogram(live_values, bins=bin_edges)
    psi_score = _psi(hist_counts, live_counts)

    reconstructed_baseline_sample = _reconstruct_sample_from_histogram(
        feat_baseline["histogram_counts"], feat_baseline["histogram_bin_edges"]
    )
    if len(reconstructed_baseline_sample) > 1 and len(live_values) > 1:
        ks_stat, ks_pvalue = stats.ks_2samp(reconstructed_baseline_sample, live_values)
    else:
        ks_pvalue = None

    is_drifted = psi_score > psi_threshold or (
        ks_pvalue is not None and ks_pvalue < ks_pvalue_threshold
    )

    return {
        "feature_name": feature_name,
        "psi_score": round(psi_score, 4),
        "ks_pvalue": round(float(ks_pvalue), 4) if ks_pvalue is not None else None,
        "is_drifted": bool(is_drifted),
    }


def compute_drift_report(
    live_feature_matrix: dict,  # {feature_name: np.ndarray of recent live values}
    baseline: dict,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
) -> list:
    return [
        compute_drift_for_feature(
            name, values, baseline, psi_threshold, ks_pvalue_threshold
        )
        for name, values in live_feature_matrix.items()
    ]
