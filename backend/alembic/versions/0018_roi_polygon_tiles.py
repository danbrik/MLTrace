"""add polygon ROI points and tile scores

Revision ID: 0018_roi_polygon_tiles
Revises: 0017_testing_run_queue_fields
Create Date: 2026-06-15 04:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0018_roi_polygon_tiles"
down_revision = "0017_testing_run_queue_fields"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "roi_definitions",
        sa.Column("geometry_type", sa.String(length=32), nullable=False, server_default="rectangle"),
    )
    op.add_column("roi_definitions", sa.Column("points", json_type, nullable=True))
    op.add_column("roi_definitions", sa.Column("tile_rows", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("roi_definitions", sa.Column("tile_cols", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("testing_run_results", sa.Column("tile_scores", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("testing_run_results", "tile_scores")
    op.drop_column("roi_definitions", "tile_cols")
    op.drop_column("roi_definitions", "tile_rows")
    op.drop_column("roi_definitions", "points")
    op.drop_column("roi_definitions", "geometry_type")
