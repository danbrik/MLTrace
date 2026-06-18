"""heatmap reconstruction preview

Revision ID: 0023_heatmap_reconstruction_preview
Revises: 0022_heatmap_direct_lookup
Create Date: 2026-06-18 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0023_heatmap_reconstruction_preview"
down_revision: str | None = "0022_heatmap_direct_lookup"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "heatmap_runs",
        sa.Column("reconstruction_image_data_url", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("heatmap_runs", "reconstruction_image_data_url")
