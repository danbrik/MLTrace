"""add device column to training runs

Revision ID: 0013_training_run_device
Revises: 0012_training_runs
Create Date: 2026-06-15 01:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0013_training_run_device"
down_revision = "0012_training_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("training_runs", sa.Column("device", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("training_runs", "device")
