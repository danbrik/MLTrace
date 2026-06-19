"""add downsampled raw error matrix to heatmap runs

Revision ID: 0024_heatmap_error_matrix
Revises: 0023_heatmap_reconstruction_preview
Create Date: 2026-06-16 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0024_heatmap_error_matrix"
down_revision: str | None = "0023_heatmap_reconstruction_preview"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("heatmap_runs", sa.Column("error_matrix", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("heatmap_runs", "error_matrix")
