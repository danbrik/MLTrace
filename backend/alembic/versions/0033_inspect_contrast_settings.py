"""add inspect contrast enhancement settings

Revision ID: 0033_inspect_contrast_settings
Revises: 0032_analysis_layouts
Create Date: 2026-06-29 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0033_inspect_contrast_settings"
down_revision: str | None = "0032_analysis_layouts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "inspect_runs",
        sa.Column("contrast_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("inspect_runs", sa.Column("contrast_reference_frames", sa.Integer(), nullable=True))
    op.add_column("inspect_runs", sa.Column("contrast_shift", sa.Float(), nullable=True))
    op.add_column("inspect_runs", sa.Column("contrast_vmax", sa.Float(), nullable=True))
    op.add_column("inspect_runs", sa.Column("contrast_ma_radius", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("inspect_runs", "contrast_ma_radius")
    op.drop_column("inspect_runs", "contrast_vmax")
    op.drop_column("inspect_runs", "contrast_shift")
    op.drop_column("inspect_runs", "contrast_reference_frames")
    op.drop_column("inspect_runs", "contrast_enabled")
