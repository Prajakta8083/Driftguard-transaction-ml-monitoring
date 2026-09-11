import numpy as np
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sklearn.metrics import precision_score, recall_score, f1_score

from app.database import get_db
from app import models, schemas
from app.config import settings
from app.ml.drift import load_baseline, compute_drift_report
from app.ml.feature_contract import FEATURE_ORDER

router = APIRouter()


@router.get("/monitoring/performance", response_model=schemas.PerformanceResponse)
def live_performance(db: Session = Depends(get_db)):
    """
    Joins predictions to feedback on transaction_id — this join is the
    whole mechanism. Feedback (ground truth) arrives LATE and only for a
    subset of transactions (only ones that got flagged / disputed / audited
    eventually get a confirmed label), so this metric is necessarily
    computed over whatever subset has BOTH a prediction AND feedback so
    far — not over all predictions. That's real, and worth stating rather
    than hiding: a small matched sample gives you a noisy metric, and the
    response says so explicitly via `n_matched`.
    """
    window = settings.performance_window_size

    matched = (
        db.query(models.Prediction.predicted_label, models.Feedback.actual_label)
        .join(models.Feedback, models.Feedback.transaction_id == models.Prediction.transaction_id)
        .order_by(models.Feedback.created_at.desc())
        .limit(window)
        .all()
    )

    if len(matched) < 5:
        return schemas.PerformanceResponse(
            window_size=window,
            n_matched=len(matched),
            precision=None,
            recall=None,
            f1=None,
            note="Not enough matched feedback yet for a stable metric (need >= 5 rows).",
        )

    y_pred = [m[0] for m in matched]
    y_true = [m[1] for m in matched]

    return schemas.PerformanceResponse(
        window_size=window,
        n_matched=len(matched),
        precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
        recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
        f1=round(f1_score(y_true, y_pred, zero_division=0), 4),
    )


@router.get("/monitoring/drift", response_model=schemas.DriftResponse)
def current_drift(db: Session = Depends(get_db), recent_n: int = 500):
    """
    Pulls the most recent N transactions' feature_vector_json (the exact
    vectors that were actually scored — not re-derived from raw_payload,
    so this stays consistent with whatever the contract produced at the
    time), reshapes into per-feature arrays, and diffs each one against
    the frozen training baseline.
    """
    baseline = load_baseline(settings.training_baseline_path)

    recent_txns = (
        db.query(models.Transaction)
        .order_by(models.Transaction.created_at.desc())
        .limit(recent_n)
        .all()
    )

    if len(recent_txns) < 30:
        return schemas.DriftResponse(
            checked_at=datetime.now(timezone.utc),
            n_features_checked=0,
            n_drifted=0,
            results=[],
        )

    live_feature_matrix = {
        feat: np.array([t.feature_vector_json[feat] for t in recent_txns])
        for feat in FEATURE_ORDER
    }

    results = compute_drift_report(
        live_feature_matrix,
        baseline,
        psi_threshold=settings.psi_drift_threshold,
        ks_pvalue_threshold=settings.ks_pvalue_threshold,
    )

    # Persist each check as a row, so drift has a history over time
    # (this is what a "drift over time" dashboard chart will read from).
    for r in results:
        db.add(models.DriftReport(
            feature_name=r["feature_name"],
            drift_score=r["psi_score"],
            method="PSI",
            is_drifted=r["is_drifted"],
        ))
    db.commit()

    n_drifted = sum(1 for r in results if r["is_drifted"])
    return schemas.DriftResponse(
        checked_at=datetime.now(timezone.utc),
        n_features_checked=len(results),
        n_drifted=n_drifted,
        results=[schemas.DriftFeatureResult(**r) for r in results],
    )


@router.get("/monitoring/drift/history", response_model=schemas.DriftHistoryResponse)
def drift_history(db: Session = Depends(get_db), limit: int = 500):
    """
    Raw drift_reports rows, oldest-to-newest, for charting. Every call to
    GET /monitoring/drift (and every scheduled job run) inserts new rows
    here — this endpoint just reads that accumulated history back out.
    The frontend groups these by feature_name to draw one line per feature.
    """
    rows = (
        db.query(models.DriftReport)
        .order_by(models.DriftReport.created_at.asc())
        .limit(limit)
        .all()
    )
    return schemas.DriftHistoryResponse(
        points=[
            schemas.DriftHistoryPoint(
                feature_name=r.feature_name,
                drift_score=r.drift_score,
                method=r.method,
                is_drifted=r.is_drifted,
                created_at=r.created_at,
            )
            for r in rows
        ]
    )


@router.get("/monitoring/performance/history", response_model=schemas.PerformanceHistoryResponse)
def performance_history(db: Session = Depends(get_db), limit: int = 200):
    rows = (
        db.query(models.PerformanceSnapshot)
        .order_by(models.PerformanceSnapshot.created_at.asc())
        .limit(limit)
        .all()
    )
    return schemas.PerformanceHistoryResponse(
        points=[
            schemas.PerformanceHistoryPoint(
                precision=r.precision,
                recall=r.recall,
                f1=r.f1,
                n_matched=r.n_matched,
                created_at=r.created_at,
            )
            for r in rows
        ]
    )
