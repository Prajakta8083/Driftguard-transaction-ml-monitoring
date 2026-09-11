"""
Feature Contract
================
THE most important file in the backend, conceptually — everything else
just calls this.

The problem this solves: a trained model doesn't know feature NAMES, it
only knows POSITIONS. When you called model.fit(X_train), X_train's
columns were in some fixed order (Time, V1, V2, ..., V28, Amount), and
the model learned weights/splits tied to those positions. If, at serving
time, you build a vector where Amount ends up in position 1 instead of
position 30, the model will not error out — it will run happily, produce
a confident-looking probability, and be completely wrong. This is the
single most dangerous class of bug in production ML: silent, not loud.

The "feature contract" is the explicit, versioned, single-source-of-truth
definition of:
  1. exactly which features the model expects
  2. in exactly what order
  3. built from exactly which fields of the incoming request

Every other part of the system (ingestion, drift detection, retraining)
must import FEATURE_ORDER from here rather than re-deriving it, so there
is never a second, possibly-inconsistent definition floating around.
"""

import numpy as np
from app.schemas import TransactionIn

# This list must match, in this exact order, the column order used when
# the model was trained (see feature_cols in model_comparison.py). If the
# training pipeline and this list ever diverge, every prediction after
# that point is silently corrupted — not crashed, corrupted.
FEATURE_ORDER = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)


def transaction_to_feature_vector(txn: TransactionIn) -> np.ndarray:
    """
    The ONLY function in the codebase allowed to construct a model input
    vector from a TransactionIn. Centralizing this means there is exactly
    one place to fix if the contract ever needs to change, and exactly
    one place to unit-test.
    """
    ordered_values = [
        txn.time,
        txn.v1, txn.v2, txn.v3, txn.v4, txn.v5, txn.v6, txn.v7,
        txn.v8, txn.v9, txn.v10, txn.v11, txn.v12, txn.v13, txn.v14,
        txn.v15, txn.v16, txn.v17, txn.v18, txn.v19, txn.v20, txn.v21,
        txn.v22, txn.v23, txn.v24, txn.v25, txn.v26, txn.v27, txn.v28,
        txn.amount,
    ]
    assert len(ordered_values) == len(FEATURE_ORDER), (
        "Feature contract violation: vector length does not match "
        "FEATURE_ORDER length. This should never happen — if it does, "
        "the contract definition and the vector-building logic have "
        "drifted apart and MUST be fixed before serving another request."
    )
    return np.array(ordered_values, dtype=float).reshape(1, -1)
