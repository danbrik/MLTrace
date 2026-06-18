"""persist train test dataset rule counts

Revision ID: 0021_training_dataset_rule_counts
Revises: 0020_testing_run_expected_image_count
Create Date: 2026-06-18 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0021_training_dataset_rule_counts"
down_revision: str | None = "0020_testing_run_expected_image_count"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("training_dataset_rules", sa.Column("matching_images", sa.Integer(), nullable=True))
    op.add_column("training_dataset_rules", sa.Column("selected_images", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("training_dataset_rules", "selected_images")
    op.drop_column("training_dataset_rules", "matching_images")
