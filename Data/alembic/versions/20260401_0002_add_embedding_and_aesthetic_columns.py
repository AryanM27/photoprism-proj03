"""add embedding columns to images and aesthetic_score to image_metadata

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-01
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    # images: track embedding lifecycle per image
    op.add_column("images", sa.Column("embedding_status", sa.String(), nullable=True, server_default="pending"))
    op.add_column("images", sa.Column("embedded_at", sa.DateTime(), nullable=True))
    op.add_column("images", sa.Column("model_version", sa.String(), nullable=True))

    # image_metadata: aesthetic score (0.0–1.0 raw; normalised to 0–10 in manifests)
    op.add_column("image_metadata", sa.Column("aesthetic_score", sa.Float(), nullable=True))


def downgrade():
    op.drop_column("image_metadata", "aesthetic_score")
    op.drop_column("images", "model_version")
    op.drop_column("images", "embedded_at")
    op.drop_column("images", "embedding_status")
