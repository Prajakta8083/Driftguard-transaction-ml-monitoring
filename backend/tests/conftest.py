"""
Shared pytest fixtures. Each test gets a fresh SQLite DB (never touches
your real Postgres data) and a TestClient with the actual app — same
code path as production, just a disposable database underneath.
"""
import os
import random

os.environ["DATABASE_URL"] = "sqlite:///./test_suite.db"

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="function")
def client():
    # IMPORTANT: don't delete the .db FILE between tests. app/database.py
    # creates the SQLAlchemy engine ONCE at import time and caches it at
    # module level (by design — see model_loader.py's caching rationale).
    # If a test deletes the underlying file while that cached engine still
    # holds pooled connections open against it, SQLite gets into a broken
    # state on the next write ("attempt to write a readonly database").
    # The fix: keep the same file + engine alive for the whole test run,
    # and reset state by dropping/recreating tables ON that engine instead.
    from app.database import Base, engine
    from app import models  # noqa
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    from app.main import app
    with TestClient(app) as c:
        yield c


def make_transaction_payload(seed=None, feature_overrides=None):
    rng = random.Random(seed)
    payload = {"Time": rng.uniform(0, 100000), "Amount": rng.uniform(1, 300)}
    for i in range(1, 29):
        payload[f"V{i}"] = rng.gauss(0, 1)
    if feature_overrides:
        payload.update(feature_overrides)
    return payload
