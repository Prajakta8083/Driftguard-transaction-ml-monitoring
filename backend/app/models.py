from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    # Raw incoming JSON exactly as received — kept verbatim for audit/replay,
    # even though it also gets parsed into structured columns below.
    raw_payload = Column(JSON, nullable=False)
    # The EXACT feature vector, in EXACT order, that was actually fed to the
    # model at prediction time. This is the feature-contract audit trail:
    # if a prediction ever looks wrong, we can check whether the contract
    # mapping (raw JSON -> ordered vector) was applied correctly for this
    # specific row, instead of guessing after the fact.
    feature_vector_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    predictions = relationship("Prediction", back_populates="transaction")
    feedback = relationship("Feedback", back_populates="transaction")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    fraud_probability = Column(Float, nullable=False)
    predicted_label = Column(Integer, nullable=False)  # 0 or 1
    model_version = Column(String, nullable=False, index=True)
    shap_top_drivers = Column(JSON, nullable=True)  # cached at prediction time
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    transaction = relationship("Transaction", back_populates="predictions")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=False, index=True)
    actual_label = Column(Integer, nullable=False)  # 0 or 1, arrives late (e.g. post-chargeback)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    transaction = relationship("Transaction", back_populates="feedback")


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String, nullable=False, index=True)
    drift_score = Column(Float, nullable=False)
    method = Column(String, nullable=False)  # "PSI" or "KS"
    is_drifted = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, nullable=False, index=True)
    trained_at = Column(DateTime(timezone=True), nullable=False)
    metrics_json = Column(JSON, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)


class PerformanceSnapshot(Base):
    """
    Added when building the frontend (Phase 5): a 'trend over time' chart
    needs a time series, and the original schema only ever computed
    performance live, on demand — nothing persisted it. Rather than fake
    a trend client-side from repeated polling, this table gives the
    scheduled job (app/scheduler.py) somewhere to write a snapshot each
    time it runs, so the chart reflects real history.
    """
    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    window_size = Column(Integer, nullable=False)
    n_matched = Column(Integer, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1 = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
