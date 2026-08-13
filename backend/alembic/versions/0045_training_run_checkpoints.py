"""add resumable training checkpoints and bounded lock retries

Revision ID: 0045_training_run_checkpoints
Revises: 0044_testing_run_checkpoints
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0045_training_run_checkpoints"
down_revision: str | None = "0044_testing_run_checkpoints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_runs") as batch:
        batch.add_column(sa.Column("checkpoint_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("checkpoint_epoch", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("checkpoint_phase", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("checkpoint_iteration", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("checkpoint_path", sa.Text(), nullable=True))
        batch.add_column(sa.Column("checkpoint_size_bytes", sa.BigInteger(), nullable=True))
        batch.add_column(sa.Column("checkpoint_signature", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("checkpoint_warning", sa.Text(), nullable=True))
        batch.add_column(sa.Column("restart_mode", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("resume_count", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("auto_retry_count", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("next_retry_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("training_runs") as batch:
        batch.drop_column("next_retry_at")
        batch.drop_column("auto_retry_count")
        batch.drop_column("resume_count")
        batch.drop_column("restart_mode")
        batch.drop_column("checkpoint_warning")
        batch.drop_column("checkpoint_signature")
        batch.drop_column("checkpoint_size_bytes")
        batch.drop_column("checkpoint_path")
        batch.drop_column("checkpoint_iteration")
        batch.drop_column("checkpoint_phase")
        batch.drop_column("checkpoint_epoch")
        batch.drop_column("checkpoint_at")
