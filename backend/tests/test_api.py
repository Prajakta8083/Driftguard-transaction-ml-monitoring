"""
DriftGuard backend test suite.

Run from inside backend/, with the venv active and fraud_model_pipeline.joblib
/ training_baseline.json / test.csv already in this folder:

    pytest tests/ -v

These test REAL code paths (the actual FastAPI app, actual SQLAlchemy models,
actual feature contract) against a disposable SQLite DB — not mocks.
"""
from tests.conftest import make_transaction_payload


# ---------------------------------------------------------------------
# Health & startup
# ---------------------------------------------------------------------
def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------
# Feature contract / ingestion
# ---------------------------------------------------------------------
def test_transaction_rejects_missing_field(client):
    """The Pydantic schema IS the feature contract's enforcement point —
    a missing V-field must be rejected at the API boundary (422), not
    silently accepted and fed to the model as a malformed vector."""
    payload = make_transaction_payload(seed=1)
    del payload["V14"]
    r = client.post("/transactions", json=payload)
    assert r.status_code == 422


def test_transaction_rejects_wrong_type(client):
    payload = make_transaction_payload(seed=1)
    payload["Amount"] = "not-a-number"
    r = client.post("/transactions", json=payload)
    assert r.status_code == 422


def test_valid_transaction_returns_full_shape(client):
    """A valid transaction must return everything the frontend depends on:
    a usable probability, a binary label, the model version, and SHAP
    drivers with the expected fields."""
    payload = make_transaction_payload(seed=2)
    r = client.post("/transactions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["predicted_label"] in (0, 1)
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["top_shap_drivers"], list)
    if body["top_shap_drivers"]:
        driver = body["top_shap_drivers"][0]
        assert "feature" in driver and "shap_value" in driver


def test_same_input_gives_same_output(client):
    """A deterministic model must score identical inputs identically —
    if this ever fails, something non-deterministic (uninitialized
    random_state, a stateful preprocessing bug) has leaked into scoring."""
    payload = make_transaction_payload(seed=3)
    r1 = client.post("/transactions", json=payload)
    r2 = client.post("/transactions", json=payload)
    assert r1.json()["fraud_probability"] == r2.json()["fraud_probability"]


# ---------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------
def test_history_counts_match_reality(client):
    for i in range(5):
        client.post("/transactions", json=make_transaction_payload(seed=i))
    r = client.get("/predictions/history?page=1&page_size=20")
    body = r.json()
    assert body["total"] == 5
    assert body["total_fraud"] + body["total_legitimate"] == 5


def test_history_pagination_does_not_overlap(client):
    for i in range(7):
        client.post("/transactions", json=make_transaction_payload(seed=i))
    page1 = client.get("/predictions/history?page=1&page_size=5").json()["items"]
    page2 = client.get("/predictions/history?page=2&page_size=5").json()["items"]
    ids_page1 = {item["id"] for item in page1}
    ids_page2 = {item["id"] for item in page2}
    assert ids_page1.isdisjoint(ids_page2)


# ---------------------------------------------------------------------
# Feedback loop + review queue
# ---------------------------------------------------------------------
def test_feedback_rejects_unknown_transaction(client):
    r = client.post("/feedback", json={"transaction_id": 999999, "actual_label": 1})
    assert r.status_code == 404


def test_feedback_rejects_invalid_label(client):
    txn = client.post("/transactions", json=make_transaction_payload(seed=4)).json()
    r = client.post("/feedback", json={"transaction_id": txn["transaction_id"], "actual_label": 2})
    assert r.status_code == 422


def test_resolving_feedback_removes_item_from_review_queue(client):
    """This is the core of the Review Queue feature: resolving feedback
    for a flagged transaction must make it disappear from the pending
    queue. If this breaks, the queue would show already-resolved cases
    forever, which defeats its entire purpose."""
    from app.database import SessionLocal
    from app import models

    db = SessionLocal()
    txn = models.Transaction(raw_payload={}, feature_vector_json={})
    db.add(txn); db.commit(); db.refresh(txn)
    txn_id = txn.id  # capture before closing the session, or accessing it
                      # later raises DetachedInstanceError
    pred = models.Prediction(
        transaction_id=txn_id, fraud_probability=0.95, predicted_label=1,
        model_version="test", shap_top_drivers=[],
    )
    db.add(pred); db.commit()
    db.close()

    before = client.get("/predictions/review-queue").json()
    assert before["total_pending"] == 1

    client.post("/feedback", json={"transaction_id": txn_id, "actual_label": 1})

    after = client.get("/predictions/review-queue").json()
    assert after["total_pending"] == 0


# ---------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------
def test_drift_check_skips_below_minimum_sample_size(client):
    """Drift should NOT be computed on too few transactions — a tiny
    sample produces a statistically meaningless PSI/KS result, and
    reporting one anyway would be misleading."""
    for i in range(5):
        client.post("/transactions", json=make_transaction_payload(seed=i))
    r = client.get("/monitoring/drift")
    assert r.json()["n_features_checked"] == 0


def test_drift_check_runs_above_minimum_sample_size(client):
    for i in range(35):
        client.post("/transactions", json=make_transaction_payload(seed=i))
    r = client.get("/monitoring/drift")
    body = r.json()
    assert body["n_features_checked"] == 30  # Time + V1-28 + Amount
    for feature_result in body["results"]:
        assert feature_result["psi_score"] >= 0  # PSI is never negative


# ---------------------------------------------------------------------
# Model report
# ---------------------------------------------------------------------
def test_model_report_returns_active_model_name(client):
    r = client.get("/model/report")
    assert r.status_code == 200
    assert r.json()["active_model"]  # non-empty string


# ---------------------------------------------------------------------
# Real-sample endpoint (the fix from the "big numbers don't work" issue)
# ---------------------------------------------------------------------
def test_sample_endpoint_rejects_invalid_label(client):
    r = client.get("/transactions/sample?label=5")
    assert r.status_code == 400


def test_sample_endpoint_returns_scoreable_row(client):
    r = client.get("/transactions/sample?label=0")
    if r.status_code == 400:
        return  # test.csv not present in this environment — acceptable skip
    sample = r.json()
    score_response = client.post("/transactions", json=sample)
    assert score_response.status_code == 200
