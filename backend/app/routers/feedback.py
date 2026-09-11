from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas

router = APIRouter()


@router.post("/feedback", response_model=schemas.FeedbackResponse)
def submit_feedback(feedback_in: schemas.FeedbackIn, db: Session = Depends(get_db)):
    txn = db.query(models.Transaction).filter(
        models.Transaction.id == feedback_in.transaction_id
    ).first()
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")

    fb_row = models.Feedback(
        transaction_id=feedback_in.transaction_id,
        actual_label=feedback_in.actual_label,
    )
    db.add(fb_row)
    db.commit()
    db.refresh(fb_row)
    return schemas.FeedbackResponse.model_validate(fb_row)
