"""add resumable testing run checkpoints

Revision ID: 0044_testing_run_checkpoints
Revises: 0043_event_threshold_detection
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0044_testing_run_checkpoints"
down_revision: str | None = "0043_event_threshold_detection"
branch_labels = None
depends_on = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("testing_runs") as batch:
        batch.add_column(sa.Column("checkpoint_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("checkpoint_input_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("checkpoint_result_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("checkpoint_state", _json_type(), nullable=True))
        batch.add_column(sa.Column("restart_mode", sa.String(length=32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("testing_runs") as batch:
        batch.drop_column("restart_mode")
        batch.drop_column("checkpoint_state")
        batch.drop_column("checkpoint_result_count")
        batch.drop_column("checkpoint_input_count")
        batch.drop_column("checkpoint_at")
