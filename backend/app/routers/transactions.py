import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.config import settings
from app.ml.feature_contract import transaction_to_feature_vector, FEATURE_ORDER
from app.ml.model_loader import get_model_bundle, ModelBundle

router = APIRouter()


@router.get("/transactions/sample")
def sample_transaction(label: int = 1):
    """
    Returns ONE real, genuinely-labeled row from the held-out test set —
    label=1 for a real fraud example, label=0 for a real legitimate one.

    This exists specifically because hand-typing arbitrary large numbers
    into V1-V28 does NOT reliably produce a fraud prediction: those are
    PCA components, and fraud correlates with a specific DIRECTION per
    feature (often large-negative for V14/V12/V10), not with magnitude in
    general. An out-of-distribution value just lands in whatever leaf
    node the model happens to route extreme inputs to — not necessarily
    the fraud leaf. Feeding the model a real row sidesteps that entirely.
    """
    if label not in (0, 1):
        raise HTTPException(status_code=400, detail="label must be 0 or 1")
    if not os.path.exists(settings.original_test_data_path):
        raise HTTPException(
            status_code=400,
            detail=f"{settings.original_test_data_path} not found next to the backend — copy test.csv in.",
        )
    df = pd.read_csv(settings.original_test_data_path)
    subset = df[df["Class"] == label]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No rows with Class={label} in test.csv")
    row = subset.sample(1).iloc[0]
    return {feat: float(row[feat]) for feat in FEATURE_ORDER}


@router.post("/transactions", response_model=schemas.TransactionResponse)
def ingest_transaction(
    txn_in: schemas.TransactionIn,
    db: Session = Depends(get_db),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    # Step 1: contract — turn validated JSON into the exact ordered vector
    # the model was trained on. This is the only place this happens.
    raw_vector = transaction_to_feature_vector(txn_in)
    scaled_vector = bundle.scaler.transform(raw_vector)

    # Step 2: persist the transaction, including BOTH the raw payload and
    # the exact vector used for scoring — so a wrong prediction can later
    # be traced back to "was the contract applied correctly for this row".
    txn_row = models.Transaction(
        raw_payload=txn_in.model_dump(by_alias=True),
        feature_vector_json=dict(zip(FEATURE_ORDER, raw_vector.flatten().tolist())),
    )
    db.add(txn_row)
    db.commit()
    db.refresh(txn_row)

    # Step 3: score
    if hasattr(bundle.model, "predict_proba"):
        fraud_probability = float(bundle.model.predict_proba(scaled_vector)[0, 1])
    else:
        fraud_probability = float(-bundle.model.score_samples(scaled_vector)[0])
    predicted_label = int(fraud_probability >= 0.5)

    # Step 4: local SHAP explanation for this one transaction
    top_drivers = []
    if bundle.explainer is not None:
        shap_out = bundle.explainer.shap_values(scaled_vector)
        # Handle both SHAP output formats (see shap_explainability.py for
        # the full explanation): older versions return a list of per-class
        # arrays, newer versions return one (n_samples, n_features, n_classes)
        # array. scaled_vector has exactly 1 row here, so index that row.
        if isinstance(shap_out, list):
            shap_row = shap_out[1][0]
        elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
            shap_row = shap_out[0, :, 1]
        else:
            shap_row = shap_out[0]
        driver_pairs = sorted(
            zip(bundle.feature_cols, shap_row.tolist()),
            key=lambda p: abs(p[1]),
            reverse=True,
        )[:5]
        top_drivers = [
            schemas.ShapDriver(feature=f, shap_value=round(v, 4)) for f, v in driver_pairs
        ]

    # Step 5: persist the prediction, including the SHAP snapshot — so
    # GET /model/explain/{id} later doesn't need to re-run SHAP against
    # whatever the CURRENT model happens to be, which could differ from
    # the model that actually made this prediction after a retrain.
    pred_row = models.Prediction(
        transaction_id=txn_row.id,
        fraud_probability=fraud_probability,
        predicted_label=predicted_label,
        model_version=bundle.model_version,
        shap_top_drivers=[d.model_dump() for d in top_drivers],
    )
    db.add(pred_row)
    db.commit()

    return schemas.TransactionResponse(
        transaction_id=txn_row.id,
        fraud_probability=round(fraud_probability, 4),
        predicted_label=predicted_label,
        model_version=bundle.model_version,
        top_shap_drivers=top_drivers,
    )
