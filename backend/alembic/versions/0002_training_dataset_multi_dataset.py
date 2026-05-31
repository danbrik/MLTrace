"""allow training datasets to span multiple datasets

Revision ID: 0002_training_dataset_multi_dataset
Revises: 0001_initial
Create Date: 2026-05-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_training_dataset_multi_dataset"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_datasets") as batch_op:
        batch_op.alter_column("dataset_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("training_datasets") as batch_op:
        batch_op.alter_column("dataset_id", existing_type=sa.Integer(), nullable=False)
