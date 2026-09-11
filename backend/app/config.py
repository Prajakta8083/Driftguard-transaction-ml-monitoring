from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Defaults to a local Postgres; override via env var in real deployment.
    database_url: str = "postgresql://driftguard:driftguard@localhost:5432/driftguard"
    model_pipeline_path: str = "fraud_model_pipeline.joblib"
    training_baseline_path: str = "training_baseline.json"

    # Retrain-trigger thresholds — deliberately explicit, named constants,
    # not magic numbers buried in logic. Interviewers will ask "why these
    # numbers", and the honest answer is: they're a starting point, tuned
    # against business risk tolerance, not derived from a formula.
    psi_drift_threshold: float = 0.2          # PSI > 0.2 = "significant" shift (industry convention)
    ks_pvalue_threshold: float = 0.05         # standard significance level
    min_drifted_features_for_alert: int = 3   # a handful of noisy features isn't a system-wide signal
    recall_drop_threshold: float = 0.15       # relative drop vs. training-time recall that triggers concern
    performance_window_size: int = 200        # rolling window size (in # of feedback-matched predictions)
    monitoring_interval_minutes: int = 5      # how often the scheduled drift+performance job runs

    # Retrain pipeline inputs — the ORIGINAL split files from Phase 0,
    # kept around specifically so retraining always has this fixed
    # reference point, not just whatever recent data happens to exist.
    original_train_data_path: str = "train.csv"
    original_test_data_path: str = "test.csv"

    class Config:
        env_file = ".env"


settings = Settings()
