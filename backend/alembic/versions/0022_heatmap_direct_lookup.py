"""heatmap direct timestamp lookup

Revision ID: 0022_heatmap_direct_lookup
Revises: 0021_training_dataset_rule_counts
Create Date: 2026-06-18 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0022_heatmap_direct_lookup"
down_revision: str | None = "0021_training_dataset_rule_counts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("dataset_folders", sa.Column("filename_template", sa.JSON(), nullable=True))
    with op.batch_alter_table("heatmap_runs") as batch:
        batch.alter_column("testing_result_id", existing_type=sa.Integer(), nullable=True)
    op.create_index(
        "ix_heatmap_runs_testing_run_timestamp",
        "heatmap_runs",
        ["testing_run_id", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_heatmap_runs_testing_run_timestamp", table_name="heatmap_runs")
    with op.batch_alter_table("heatmap_runs") as batch:
        batch.alter_column("testing_result_id", existing_type=sa.Integer(), nullable=False)
    op.drop_column("dataset_folders", "filename_template")
