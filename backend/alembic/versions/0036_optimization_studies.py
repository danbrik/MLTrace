"""add optimization studies and trials

Revision ID: 0036_optimization_studies
Revises: 0035_training_dataset_updated_at
Create Date: 2026-07-06
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0036_optimization_studies"
down_revision: str | None = "0035_training_dataset_updated_at"
branch_labels: str | None = None
depends_on: str | None = None


def _json_type():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "optimization_studies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="draft"),
        sa.Column("objective_name", sa.String(length=64), nullable=False, server_default="median_anomaly_minus_p95_normal"),
        sa.Column("direction", sa.String(length=16), nullable=False, server_default="maximize"),
        sa.Column("n_trials", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("max_parallel_trials", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("sampler", sa.String(length=64), nullable=False, server_default="tpe"),
        sa.Column("preprocessing_pipeline_id", sa.Integer(), sa.ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("method_configuration_ids", _json_type(), nullable=False),
        sa.Column("normal_train_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("normal_validation_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("anomaly_validation_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("normal_holdout_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("anomaly_holdout_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=True),
        sa.Column("search_space", _json_type(), nullable=False),
        sa.Column("split_config", _json_type(), nullable=False),
        sa.Column("objective_config", _json_type(), nullable=False),
        sa.Column("best_trial_id", sa.Integer(), nullable=True),
        sa.Column("best_value", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_optimization_studies_status", "optimization_studies", ["status"])
    op.create_index("ix_optimization_studies_created_at", "optimization_studies", ["created_at"])

    op.create_table(
        "optimization_trials",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("study_id", sa.Integer(), sa.ForeignKey("optimization_studies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="waiting"),
        sa.Column("phase", sa.String(length=64), nullable=False, server_default="waiting"),
        sa.Column("sampled_params", _json_type(), nullable=False),
        sa.Column("method_configuration_id", sa.Integer(), sa.ForeignKey("method_configurations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("training_pipeline_id", sa.Integer(), sa.ForeignKey("training_pipelines.id", ondelete="SET NULL"), nullable=True),
        sa.Column("training_run_id", sa.Integer(), sa.ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("normal_testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("anomaly_testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("normal_holdout_testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("anomaly_holdout_testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("objective_value", sa.Float(), nullable=True),
        sa.Column("metrics", _json_type(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("study_id", "number", name="uq_optimization_trial_number"),
    )
    op.create_index("ix_optimization_trials_study_status", "optimization_trials", ["study_id", "status"])
    op.create_index("ix_optimization_trials_value", "optimization_trials", ["objective_value"])


def downgrade() -> None:
    op.drop_index("ix_optimization_trials_value", table_name="optimization_trials")
    op.drop_index("ix_optimization_trials_study_status", table_name="optimization_trials")
    op.drop_table("optimization_trials")
    op.drop_index("ix_optimization_studies_created_at", table_name="optimization_studies")
    op.drop_index("ix_optimization_studies_status", table_name="optimization_studies")
    op.drop_table("optimization_studies")
