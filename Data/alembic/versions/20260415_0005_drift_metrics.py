"""add drift_metrics table

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-15
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drift_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version_tag", sa.Text(), nullable=False),
        sa.Column("reference_version_tag", sa.Text(), nullable=True),
        sa.Column("manifest_kind", sa.Text(), nullable=False),   # 'semantic' | 'aesthetic'
        sa.Column("column_name", sa.Text(), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=True),
        sa.Column("drift_detected", sa.Boolean(), nullable=True),
        sa.Column("stattest_name", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("drift_metrics")
