"""add heatmap artifacts_dir (move heavy heatmap payloads to disk)

Revision ID: 0034_heatmap_artifacts_dir
Revises: 0033_inspect_contrast_settings
Create Date: 2026-07-01 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0034_heatmap_artifacts_dir"
down_revision: str | None = "0033_inspect_contrast_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("heatmap_runs", sa.Column("artifacts_dir", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("heatmap_runs", "artifacts_dir")
