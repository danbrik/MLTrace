"""heatmap runs

Revision ID: 0019_heatmap_runs
Revises: 0018_roi_polygon_tiles
Create Date: 2026-06-15 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0019_heatmap_runs"
down_revision: str | None = "0018_roi_polygon_tiles"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "heatmap_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("testing_run_id", sa.Integer(), nullable=False),
        sa.Column("testing_result_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="finished"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("channels", sa.Integer(), nullable=False),
        sa.Column("dtype", sa.String(length=64), nullable=False),
        sa.Column("max_error", sa.Float(), nullable=False),
        sa.Column("mean_error", sa.Float(), nullable=False),
        sa.Column("max_x", sa.Integer(), nullable=False),
        sa.Column("max_y", sa.Integer(), nullable=False),
        sa.Column("source_image_data_url", sa.Text(), nullable=False),
        sa.Column("heatmap_image_data_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["testing_run_id"], ["testing_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["testing_result_id"], ["testing_run_results.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("testing_run_id", "testing_result_id", name="uq_heatmap_result"),
    )
    op.create_index("ix_heatmap_runs_created_at", "heatmap_runs", ["created_at"])
    op.create_index("ix_heatmap_runs_status", "heatmap_runs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_heatmap_runs_status", table_name="heatmap_runs")
    op.drop_index("ix_heatmap_runs_created_at", table_name="heatmap_runs")
    op.drop_table("heatmap_runs")
