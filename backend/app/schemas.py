from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# This class IS the feature contract's entry point. Field names here
# must match the dataset's column names exactly (Time, V1..V28, Amount),
# because app/ml/feature_contract.py builds the model's input vector by
# reading these exact attribute names in a fixed order. If a client sends
# JSON with a typo'd or missing field, Pydantic rejects it at the API
# boundary with a clear 422 error — BEFORE it can silently become a
# wrong-shaped feature vector.
# ---------------------------------------------------------------------
class TransactionIn(BaseModel):
    time: float = Field(..., alias="Time")
    v1: float = Field(..., alias="V1")
    v2: float = Field(..., alias="V2")
    v3: float = Field(..., alias="V3")
    v4: float = Field(..., alias="V4")
    v5: float = Field(..., alias="V5")
    v6: float = Field(..., alias="V6")
    v7: float = Field(..., alias="V7")
    v8: float = Field(..., alias="V8")
    v9: float = Field(..., alias="V9")
    v10: float = Field(..., alias="V10")
    v11: float = Field(..., alias="V11")
    v12: float = Field(..., alias="V12")
    v13: float = Field(..., alias="V13")
    v14: float = Field(..., alias="V14")
    v15: float = Field(..., alias="V15")
    v16: float = Field(..., alias="V16")
    v17: float = Field(..., alias="V17")
    v18: float = Field(..., alias="V18")
    v19: float = Field(..., alias="V19")
    v20: float = Field(..., alias="V20")
    v21: float = Field(..., alias="V21")
    v22: float = Field(..., alias="V22")
    v23: float = Field(..., alias="V23")
    v24: float = Field(..., alias="V24")
    v25: float = Field(..., alias="V25")
    v26: float = Field(..., alias="V26")
    v27: float = Field(..., alias="V27")
    v28: float = Field(..., alias="V28")
    amount: float = Field(..., alias="Amount")

    class Config:
        populate_by_name = True


class ShapDriver(BaseModel):
    feature: str
    shap_value: float


class TransactionResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    predicted_label: int
    model_version: str
    top_shap_drivers: List[ShapDriver]


class PredictionHistoryItem(BaseModel):
    id: int
    transaction_id: int
    fraud_probability: float
    predicted_label: int
    model_version: str
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
    page: int
    page_size: int
    total: int
    total_fraud: int
    total_legitimate: int


class ReviewQueueItem(BaseModel):
    prediction_id: int
    transaction_id: int
    fraud_probability: float
    top_shap_drivers: List[ShapDriver]
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewQueueResponse(BaseModel):
    items: List[ReviewQueueItem]
    total_pending: int


class FeedbackIn(BaseModel):
    transaction_id: int
    actual_label: int = Field(..., ge=0, le=1)


class FeedbackResponse(BaseModel):
    id: int
    transaction_id: int
    actual_label: int
    created_at: datetime

    class Config:
        from_attributes = True


class PerformanceResponse(BaseModel):
    window_size: int
    n_matched: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    note: Optional[str] = None


class DriftFeatureResult(BaseModel):
    feature_name: str
    psi_score: float
    ks_pvalue: Optional[float]
    is_drifted: bool


class DriftResponse(BaseModel):
    checked_at: datetime
    n_features_checked: int
    n_drifted: int
    results: List[DriftFeatureResult]


class RetrainCheckResponse(BaseModel):
    retrain_recommended: bool
    reasons: List[str]
    checked_at: datetime


class DriftHistoryPoint(BaseModel):
    feature_name: str
    drift_score: float
    method: str
    is_drifted: bool
    created_at: datetime


class DriftHistoryResponse(BaseModel):
    points: List[DriftHistoryPoint]


class PerformanceHistoryPoint(BaseModel):
    precision: float
    recall: float
    f1: float
    n_matched: int
    created_at: datetime


class PerformanceHistoryResponse(BaseModel):
    points: List[PerformanceHistoryPoint]
