"""add train/test dataset usage label

Revision ID: 0014_train_test_dataset_label
Revises: 0013_training_run_device
Create Date: 2026-06-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0014_train_test_dataset_label"
down_revision = "0013_training_run_device"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("training_datasets")}
    if "usage_label" not in columns:
        op.add_column(
            "training_datasets",
            sa.Column("usage_label", sa.String(length=32), nullable=False, server_default="train"),
        )


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("training_datasets")}
    if "usage_label" in columns:
        op.drop_column("training_datasets", "usage_label")
