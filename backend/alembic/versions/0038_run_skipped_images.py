"""add skipped-image reporting to training and testing runs

Revision ID: 0038_run_skipped_images
Revises: 0037_inspect_diagnostics
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0038_run_skipped_images"
down_revision: str | None = "0037_inspect_diagnostics"
branch_labels: str | None = None
depends_on: str | None = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("skipped_image_count", sa.Integer(), nullable=True))
    op.add_column("training_runs", sa.Column("skipped_images", _json_type(), nullable=True))
    op.add_column("testing_runs", sa.Column("skipped_image_count", sa.Integer(), nullable=True))
    op.add_column("testing_runs", sa.Column("skipped_images", _json_type(), nullable=True))


def downgrade() -> None:
    op.drop_column("testing_runs", "skipped_images")
    op.drop_column("testing_runs", "skipped_image_count")
    op.drop_column("training_runs", "skipped_images")
    op.drop_column("training_runs", "skipped_image_count")
