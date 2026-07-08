"""add scheduler queue rank

Revision ID: 0039_scheduler_queue_rank
Revises: 0038_run_skipped_images
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0039_scheduler_queue_rank"
down_revision: str | None = "0038_run_skipped_images"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    for table in ("training_runs", "testing_runs", "heatmap_range_runs"):
        op.add_column(table, sa.Column("queue_rank", sa.Integer(), nullable=True))
        op.create_index(f"ix_{table}_status_queue_rank", table, ["status", "queue_rank"])


def downgrade() -> None:
    for table in ("heatmap_range_runs", "testing_runs", "training_runs"):
        op.drop_index(f"ix_{table}_status_queue_rank", table_name=table)
        op.drop_column(table, "queue_rank")
