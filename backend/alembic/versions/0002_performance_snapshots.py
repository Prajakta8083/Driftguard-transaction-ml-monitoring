"""add performance_snapshots

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "performance_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, index=True),
        sa.Column("window_size", sa.Integer, nullable=False),
        sa.Column("n_matched", sa.Integer, nullable=False),
        sa.Column("precision", sa.Float, nullable=False),
        sa.Column("recall", sa.Float, nullable=False),
        sa.Column("f1", sa.Float, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )


def downgrade():
    op.drop_table("performance_snapshots")
