"""version cached heatmap renderings

Revision ID: 0027_heatmap_render_version
Revises: 0026_warp_output_shape_mode
Create Date: 2026-06-22 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0027_heatmap_render_version"
down_revision: str | None = "0026_warp_output_shape_mode"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "heatmap_runs",
        sa.Column("render_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "heatmap_range_runs",
        sa.Column("render_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column("heatmap_range_runs", sa.Column("frame_max_errors", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("heatmap_range_runs", "frame_max_errors")
    op.drop_column("heatmap_range_runs", "render_version")
    op.drop_column("heatmap_runs", "render_version")
