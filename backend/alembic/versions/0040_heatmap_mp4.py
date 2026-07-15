"""add heatmap MP4 metadata

Revision ID: 0040_heatmap_mp4
Revises: 0039_scheduler_queue_rank
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0040_heatmap_mp4"
down_revision: str | None = "0039_scheduler_queue_rank"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "heatmap_range_runs",
        sa.Column("fps", sa.Integer(), nullable=False, server_default="8"),
    )
    op.add_column("heatmap_range_runs", sa.Column("video_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("heatmap_range_runs", "video_path")
    op.drop_column("heatmap_range_runs", "fps")
