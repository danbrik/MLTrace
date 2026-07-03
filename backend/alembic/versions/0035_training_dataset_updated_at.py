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
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("training_datasets")}
    if "updated_at" not in columns:
        # SQLite cannot add a column with a non-constant CURRENT_TIMESTAMP
        # default to an existing table. Add the nullable column first, then
        # backfill it explicitly below.
        op.add_column("training_datasets", sa.Column("updated_at", sa.DateTime(timezone=False), nullable=True))
    op.execute(
        "UPDATE training_datasets "
        "SET updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP) "
        "WHERE updated_at IS NULL"
    )


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("training_datasets")}
    if "updated_at" in columns:
        op.drop_column("training_datasets", "updated_at")
