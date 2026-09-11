from datetime import datetime, timezone
import json
import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.config import settings
from app.routers.monitoring import live_performance, current_drift
from app.ml.retrain import retrain_model
from app.ml.model_loader import get_model_bundle, ModelBundle

router = APIRouter()


@router.get("/model/global-importance")
def global_importance(bundle: ModelBundle = Depends(get_model_bundle)):
    """Cached at startup (see ModelBundle) — cheap to serve, not recomputed per request."""
    return {"model_version": bundle.model_version, "features": bundle.global_importance}


@router.get("/model/report")
def model_report(bundle: ModelBundle = Depends(get_model_bundle)):
    """
    Serves the Phase 1 model_comparison_results.json (all 8 model/imbalance-
    technique variants and their PR-AUC/precision/recall/F1) alongside which
    one is currently active. This is the 'analysis report' — the evidence
    that the active model was picked by comparison, not by assumption.
    """
    comparison = []
    if os.path.exists("model_comparison_results.json"):
        with open("model_comparison_results.json") as f:
            comparison = json.load(f)
    return {
        "active_model": bundle.model_version,
        "comparison": comparison,
        "dataset_summary": {
            "source": "Kaggle: Credit Card Fraud Detection (mlg-ulb)",
            "total_transactions": 284807,
            "fraud_transactions": 492,
            "fraud_rate_pct": 0.173,
        },
    }


@router.get("/model/explain/{transaction_id}", response_model=schemas.TransactionResponse)
def explain_prediction(transaction_id: int, db: Session = Depends(get_db)):
    """
    Returns the SHAP explanation that was computed and stored AT PREDICTION
    TIME (see routers/transactions.py) rather than recomputing it now.
    This matters: if the model has been retrained since this prediction
    was made, recomputing SHAP now would explain a DIFFERENT model's
    reasoning, not the one that actually made this decision. For a
    compliance-style "why was this transaction flagged" answer, you must
    explain the model that actually made the call.
    """
    pred = (
        db.query(models.Prediction)
        .filter(models.Prediction.transaction_id == transaction_id)
        .order_by(models.Prediction.created_at.desc())
        .first()
    )
    if pred is None:
        raise HTTPException(status_code=404, detail="No prediction found for this transaction")

    return schemas.TransactionResponse(
        transaction_id=pred.transaction_id,
        fraud_probability=pred.fraud_probability,
        predicted_label=pred.predicted_label,
        model_version=pred.model_version,
        top_shap_drivers=[schemas.ShapDriver(**d) for d in (pred.shap_top_drivers or [])],
    )


@router.post("/model/retrain-check", response_model=schemas.RetrainCheckResponse)
def retrain_check(db: Session = Depends(get_db)):
    """
    Deliberately simple and honest: this is NOT an ML model deciding to
    retrain itself. It's a small set of named, inspectable rules over two
    signals we already compute elsewhere (drift + live performance).
    A human reads `reasons` and decides whether to hit "retrain" — this
    endpoint recommends, it does not act.
    """
    reasons = []

    perf = live_performance(db=db)
    if perf.recall is not None:
        # Compare live recall against the ORIGINAL test-set recall stored
        # in model_versions.metrics_json for the currently active model.
        active_version = (
            db.query(models.ModelVersion)
            .filter(models.ModelVersion.is_active.is_(True))
            .first()
        )
        if active_version is not None:
            baseline_recall = active_version.metrics_json.get("recall")
            if baseline_recall:
                relative_drop = (baseline_recall - perf.recall) / baseline_recall
                if relative_drop > settings.recall_drop_threshold:
                    reasons.append(
                        f"Live recall ({perf.recall}) has dropped "
                        f"{relative_drop:.1%} vs. training-time recall "
                        f"({baseline_recall}), exceeding the "
                        f"{settings.recall_drop_threshold:.0%} threshold."
                    )

    drift = current_drift(db=db)
    if drift.n_drifted >= settings.min_drifted_features_for_alert:
        drifted_names = [r.feature_name for r in drift.results if r.is_drifted]
        reasons.append(
            f"{drift.n_drifted} features show significant drift "
            f"(>= {settings.min_drifted_features_for_alert} threshold): "
            f"{', '.join(drifted_names)}."
        )

    return schemas.RetrainCheckResponse(
        retrain_recommended=len(reasons) > 0,
        reasons=reasons or ["No thresholds breached — model appears stable."],
        checked_at=datetime.now(timezone.utc),
    )


@router.post("/model/retrain")
def trigger_retrain(db: Session = Depends(get_db)):
    """
    The 'manual retrain button.' Deliberately NOT triggered automatically
    by retrain_check() above — a human decides to pull this trigger after
    reading the recommendation and reasons. Returns the new version's
    metrics so it can be compared against the currently active version
    BEFORE anyone promotes it (see /model/activate below).
    """
    try:
        result = retrain_model(db, triggered_by="manual_button")
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@router.post("/model/activate/{version}")
def activate_version(version: str, db: Session = Depends(get_db)):
    """
    Explicit promotion step. Deactivates whatever is currently active and
    activates the named version. Does NOT hot-swap the in-memory model
    used by /transactions — that still requires a process restart (see
    app/ml/model_loader.py), which is called out there as a known,
    intentional limitation rather than something silently glossed over.
    """
    version_row = db.query(models.ModelVersion).filter(
        models.ModelVersion.version == version
    ).first()
    if version_row is None:
        raise HTTPException(status_code=404, detail="Model version not found")

    db.query(models.ModelVersion).filter(models.ModelVersion.is_active.is_(True)).update(
        {"is_active": False}
    )
    version_row.is_active = True
    db.commit()

    return {
        "activated_version": version,
        "note": (
            "Marked active in the database. The running API process still "
            "needs a restart (pointed at this version's artifact file) to "
            "actually serve predictions with it — this endpoint does not "
            "hot-swap the in-memory model."
        ),
    }
