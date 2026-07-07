"""add inspect diagnostics artifacts

Revision ID: 0037_inspect_diagnostics
Revises: 0036_optimization_studies
Create Date: 2026-07-07
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0037_inspect_diagnostics"
down_revision: str | None = "0036_optimization_studies"
branch_labels: str | None = None
depends_on: str | None = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "inspect_runs",
        sa.Column("analysis_mode", sa.String(length=64), nullable=False, server_default="preprocessed_video"),
    )
    op.add_column("inspect_runs", sa.Column("analysis_config", _json_type(), nullable=True))
    op.add_column("inspect_runs", sa.Column("roi_id", sa.Integer(), nullable=True))
    op.add_column(
        "inspect_runs",
        sa.Column("generate_video", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column("inspect_runs", sa.Column("csv_path", sa.Text(), nullable=True))
    op.add_column("inspect_runs", sa.Column("summary_json_path", sa.Text(), nullable=True))
    op.add_column("inspect_runs", sa.Column("plot_preview_path", sa.Text(), nullable=True))
    op.add_column("inspect_runs", sa.Column("overlay_video_path", sa.Text(), nullable=True))
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.create_foreign_key(
            "fk_inspect_runs_roi_id",
            "inspect_runs",
            "roi_definitions",
            ["roi_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.execute(
            "UPDATE inspect_runs SET analysis_mode = 'contrast_enhanced' "
            "WHERE contrast_enabled = true AND analysis_mode = 'preprocessed_video'"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_constraint("fk_inspect_runs_roi_id", "inspect_runs", type_="foreignkey")
    op.drop_column("inspect_runs", "overlay_video_path")
    op.drop_column("inspect_runs", "plot_preview_path")
    op.drop_column("inspect_runs", "summary_json_path")
    op.drop_column("inspect_runs", "csv_path")
    op.drop_column("inspect_runs", "generate_video")
    op.drop_column("inspect_runs", "roi_id")
    op.drop_column("inspect_runs", "analysis_config")
    op.drop_column("inspect_runs", "analysis_mode")
