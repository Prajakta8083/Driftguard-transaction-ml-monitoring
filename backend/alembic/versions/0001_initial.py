"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-01-01
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("raw_payload", sa.JSON, nullable=False),
        sa.Column("feature_vector_json", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("transaction_id", sa.Integer, sa.ForeignKey("transactions.id"), nullable=False, index=True),
        sa.Column("fraud_probability", sa.Float, nullable=False),
        sa.Column("predicted_label", sa.Integer, nullable=False),
        sa.Column("model_version", sa.String, nullable=False, index=True),
        sa.Column("shap_top_drivers", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("transaction_id", sa.Integer, sa.ForeignKey("transactions.id"), nullable=False, index=True),
        sa.Column("actual_label", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("feature_name", sa.String, nullable=False, index=True),
        sa.Column("drift_score", sa.Float, nullable=False),
        sa.Column("method", sa.String, nullable=False),
        sa.Column("is_drifted", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("version", sa.String, unique=True, nullable=False, index=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metrics_json", sa.JSON, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.false()),
    )


def downgrade():
    op.drop_table("model_versions")
    op.drop_table("drift_reports")
    op.drop_table("feedback")
    op.drop_table("predictions")
    op.drop_table("transactions")
