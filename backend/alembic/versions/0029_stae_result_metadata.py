"""add testing result metadata for sequence methods

Revision ID: 0029_stae_result_metadata
Revises: 0028_heatmap_visualization_config
Create Date: 2026-06-28 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0029_stae_result_metadata"
down_revision: str | None = "0028_heatmap_visualization_config"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("testing_run_results", sa.Column("result_metadata", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("testing_run_results", "result_metadata")
