"""add scheduler/queue fields to testing runs

Revision ID: 0017_testing_run_queue_fields
Revises: 0016_training_pipeline_config_signature
Create Date: 2026-06-15 03:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0017_testing_run_queue_fields"
down_revision = "0016_training_pipeline_config_signature"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("testing_runs", sa.Column("enqueued_at", sa.DateTime(timezone=False), nullable=True))
    op.add_column("testing_runs", sa.Column("gpu_index", sa.Integer(), nullable=True))
    op.add_column("testing_runs", sa.Column("device", sa.String(length=32), nullable=True))
    op.add_column("testing_runs", sa.Column("pid", sa.Integer(), nullable=True))
    op.add_column("testing_runs", sa.Column("log_path", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("testing_runs", "log_path")
    op.drop_column("testing_runs", "pid")
    op.drop_column("testing_runs", "device")
    op.drop_column("testing_runs", "gpu_index")
    op.drop_column("testing_runs", "enqueued_at")
