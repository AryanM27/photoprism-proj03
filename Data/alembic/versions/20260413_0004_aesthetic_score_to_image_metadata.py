"""move aesthetic_score from feedback_events to image_metadata

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-13
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_metadata",
        sa.Column("dataset_aesthetic_score", sa.Float(), nullable=True),
    )
    op.add_column(
        "image_metadata",
        sa.Column("aesthetic_model_version", sa.String(), nullable=True),
    )
    op.add_column(
        "image_metadata",
        sa.Column("aesthetic_score_date", sa.DateTime(), nullable=True),
    )

    op.execute("""
        UPDATE image_metadata im
        SET
            aesthetic_score = fe.aesthetic_score,
            dataset_aesthetic_score = fe.aesthetic_score,
            aesthetic_model_version = 'ava_ground_truth',
            aesthetic_score_date = NOW()
        FROM (
            SELECT image_id, aesthetic_score
            FROM feedback_events
            WHERE user_id = 'ava_ground_truth'
              AND aesthetic_score IS NOT NULL
        ) fe
        WHERE im.image_id = fe.image_id
    """)

    op.drop_column("feedback_events", "aesthetic_score")


def downgrade() -> None:
    op.add_column(
        "feedback_events",
        sa.Column("aesthetic_score", sa.Float(), nullable=True),
    )

    op.execute("""
        UPDATE feedback_events fe
        SET aesthetic_score = im.dataset_aesthetic_score
        FROM image_metadata im
        WHERE fe.image_id = im.image_id
          AND fe.user_id = 'ava_ground_truth'
          AND im.dataset_aesthetic_score IS NOT NULL
    """)

    op.execute("UPDATE image_metadata SET aesthetic_score = NULL")

    op.drop_column("image_metadata", "aesthetic_score_date")
    op.drop_column("image_metadata", "aesthetic_model_version")
    op.drop_column("image_metadata", "dataset_aesthetic_score")
