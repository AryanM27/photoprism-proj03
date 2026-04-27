"""add user_id to images

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-27
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("images", sa.Column("user_id", sa.String(), nullable=True))
    op.create_index("ix_images_user_id", "images", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_images_user_id", table_name="images")
    op.drop_column("images", "user_id")
