"""add manifest_type column to dataset_snapshots

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "dataset_snapshots",
        sa.Column("manifest_type", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("dataset_snapshots", "manifest_type")
