"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-03-21 00:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("image_id", sa.String(), nullable=False),
        sa.Column("image_uri", sa.String(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("source_dataset", sa.String(), nullable=True),
        sa.Column("split", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("image_id"),
    )

    op.create_table(
        "image_metadata",
        sa.Column("image_id", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("source_dataset", sa.String(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("format", sa.String(), nullable=True),
        sa.Column("exif_json", sa.Text(), nullable=True),
        sa.Column("tags", sa.Text(), nullable=True),
        sa.Column("captured_at", sa.DateTime(), nullable=True),
        sa.Column("normalized_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("image_id"),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("image_id", sa.String(), nullable=False),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("job_id"),
    )

    op.create_table(
        "feedback_events",
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("query_id", sa.String(), nullable=True),
        sa.Column("image_id", sa.String(), nullable=False),
        sa.Column("shown_rank", sa.Integer(), nullable=True),
        sa.Column("clicked", sa.Boolean(), nullable=True),
        sa.Column("favorited", sa.Boolean(), nullable=True),
        sa.Column("semantic_score", sa.Float(), nullable=True),
        sa.Column("aesthetic_score", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("event_id"),
    )

    op.create_table(
        "dataset_snapshots",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("version_tag", sa.String(), nullable=True),
        sa.Column("manifest_path", sa.String(), nullable=True),
        sa.Column("record_count", sa.Integer(), nullable=True),
        sa.Column("split_strategy", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )


def downgrade() -> None:
    op.drop_table("dataset_snapshots")
    op.drop_table("feedback_events")
    op.drop_table("processing_jobs")
    op.drop_table("image_metadata")
    op.drop_table("images")
