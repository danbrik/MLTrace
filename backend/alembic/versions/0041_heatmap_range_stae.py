"""add STAE view metadata to heatmap range runs

Revision ID: 0041_heatmap_range_stae
Revises: 0040_heatmap_mp4
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0041_heatmap_range_stae"
down_revision: str | None = "0040_heatmap_mp4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "heatmap_range_runs",
        sa.Column("stae_view", sa.String(length=32), nullable=False, server_default="reconstruction"),
    )
    op.add_column(
        "heatmap_range_runs",
        sa.Column("prediction_horizon", sa.Integer(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("heatmap_range_runs", "prediction_horizon")
    op.drop_column("heatmap_range_runs", "stae_view")
