from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter()


@router.get("/predictions/history", response_model=schemas.PredictionHistoryResponse)
def prediction_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    total = db.query(models.Prediction).count()
    total_fraud = db.query(models.Prediction).filter(models.Prediction.predicted_label == 1).count()
    items = (
        db.query(models.Prediction)
        .order_by(models.Prediction.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return schemas.PredictionHistoryResponse(
        items=[schemas.PredictionHistoryItem.model_validate(i) for i in items],
        page=page,
        page_size=page_size,
        total=total,
        total_fraud=total_fraud,
        total_legitimate=total - total_fraud,
    )


@router.get("/predictions/review-queue", response_model=schemas.ReviewQueueResponse)
def review_queue(db: Session = Depends(get_db), limit: int = 50):
    """
    Every prediction flagged as fraud (predicted_label=1) that has NO
    feedback row yet — i.e. a real analyst hasn't confirmed the outcome.
    This is the app's actual actionable workflow: Live Scoring produces a
    recommendation, but nothing previously let anyone DO anything with it.
    Resolving an item here (via the existing POST /feedback) removes it
    from the queue automatically, since the filter is "no feedback yet".
    """
    subquery = db.query(models.Feedback.transaction_id).subquery()
    pending = (
        db.query(models.Prediction)
        .filter(models.Prediction.predicted_label == 1)
        .filter(models.Prediction.transaction_id.notin_(db.query(subquery.c.transaction_id)))
        .order_by(models.Prediction.fraud_probability.desc())
        .limit(limit)
        .all()
    )
    total_pending = (
        db.query(models.Prediction)
        .filter(models.Prediction.predicted_label == 1)
        .filter(models.Prediction.transaction_id.notin_(db.query(subquery.c.transaction_id)))
        .count()
    )
    return schemas.ReviewQueueResponse(
        items=[
            schemas.ReviewQueueItem(
                prediction_id=p.id,
                transaction_id=p.transaction_id,
                fraud_probability=p.fraud_probability,
                top_shap_drivers=[schemas.ShapDriver(**d) for d in (p.shap_top_drivers or [])],
                created_at=p.created_at,
            )
            for p in pending
        ],
        total_pending=total_pending,
    )
