"""testing run expected image count

Revision ID: 0020_testing_run_expected_image_count
Revises: 0019_heatmap_runs
Create Date: 2026-06-18 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0020_testing_run_expected_image_count"
down_revision: str | None = "0019_heatmap_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("testing_runs", sa.Column("expected_image_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("testing_runs", "expected_image_count")
