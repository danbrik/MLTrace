"""add updated_at to training_datasets

Revision ID: 0035_training_dataset_updated_at
Revises: 0034_heatmap_artifacts_dir
Create Date: 2026-07-02 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0035_training_dataset_updated_at"
down_revision: str | None = "0034_heatmap_artifacts_dir"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_datasets",
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=True),
    )
    op.execute("UPDATE training_datasets SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)")


def downgrade() -> None:
    op.drop_column("training_datasets", "updated_at")
