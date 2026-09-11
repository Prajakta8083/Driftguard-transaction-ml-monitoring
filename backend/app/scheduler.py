"""
Scheduler
=========
Runs the drift check and performance check on a recurring timer, using
APScheduler's BackgroundScheduler. This is the piece that makes DriftGuard
a "living system" rather than a set of endpoints someone has to remember
to call: monitoring happens whether or not anyone is looking at a
dashboard right now.

Design note: scheduled jobs run OUTSIDE any HTTP request, so they can't
use FastAPI's `Depends(get_db)` — that dependency only exists inside a
request's lifecycle. Instead, each job opens and closes its own DB
session directly from SessionLocal. This is a common gotcha: code that
works fine in a route handler will NOT work unchanged inside a
scheduled job, because the two have different execution contexts.
"""

import logging
from datetime import datetime, timezone

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

from app.database import SessionLocal
from app import models
from app.config import settings
from app.ml.drift import load_baseline, compute_drift_report
from app.ml.feature_contract import FEATURE_ORDER

logger = logging.getLogger("driftguard.scheduler")
logger.setLevel(logging.INFO)


def run_drift_check_job():
    db = SessionLocal()
    try:
        baseline = load_baseline(settings.training_baseline_path)
        recent_txns = (
            db.query(models.Transaction)
            .order_by(models.Transaction.created_at.desc())
            .limit(500)
            .all()
        )
        if len(recent_txns) < 30:
            logger.info("Drift check skipped: fewer than 30 recent transactions.")
            return

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
        for r in results:
            db.add(models.DriftReport(
                feature_name=r["feature_name"],
                drift_score=r["psi_score"],
                method="PSI",
                is_drifted=r["is_drifted"],
            ))
        db.commit()

        n_drifted = sum(1 for r in results if r["is_drifted"])
        if n_drifted >= settings.min_drifted_features_for_alert:
            drifted_names = [r["feature_name"] for r in results if r["is_drifted"]]
            logger.warning(
                f"DRIFT ALERT: {n_drifted} features drifted: {drifted_names}"
            )
        else:
            logger.info(f"Drift check complete: {n_drifted} feature(s) drifted (below alert threshold).")
    except FileNotFoundError:
        logger.warning("Drift check skipped: training_baseline.json not found.")
    finally:
        db.close()


def run_performance_check_job():
    from sklearn.metrics import precision_score, recall_score, f1_score

    db = SessionLocal()
    try:
        matched = (
            db.query(models.Prediction.predicted_label, models.Feedback.actual_label)
            .join(models.Feedback, models.Feedback.transaction_id == models.Prediction.transaction_id)
            .order_by(models.Feedback.created_at.desc())
            .limit(settings.performance_window_size)
            .all()
        )
        if len(matched) < 5:
            logger.info("Performance check skipped: fewer than 5 matched feedback rows.")
            return

        y_pred = [m[0] for m in matched]
        y_true = [m[1] for m in matched]
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logger.info(
            f"Live performance (n={len(matched)}): "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
        )

        db.add(models.PerformanceSnapshot(
            window_size=settings.performance_window_size,
            n_matched=len(matched),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
        ))
        db.commit()

        active_version = (
            db.query(models.ModelVersion)
            .filter(models.ModelVersion.is_active.is_(True))
            .first()
        )
        if active_version is not None:
            baseline_recall = active_version.metrics_json.get("recall")
            if baseline_recall:
                relative_drop = (baseline_recall - recall) / baseline_recall
                if relative_drop > settings.recall_drop_threshold:
                    logger.warning(
                        f"PERFORMANCE ALERT: recall dropped {relative_drop:.1%} "
                        f"vs. training-time baseline ({baseline_recall})."
                    )
    finally:
        db.close()


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=timezone.utc)
    scheduler.add_job(
        run_drift_check_job,
        "interval",
        minutes=settings.monitoring_interval_minutes,
        id="drift_check",
        next_run_time=datetime.now(timezone.utc),  # run once immediately, then on interval
    )
    scheduler.add_job(
        run_performance_check_job,
        "interval",
        minutes=settings.monitoring_interval_minutes,
        id="performance_check",
        next_run_time=datetime.now(timezone.utc),
    )
    scheduler.start()
    logger.info(
        f"Monitoring scheduler started: checks run every "
        f"{settings.monitoring_interval_minutes} minute(s)."
    )
    return scheduler
