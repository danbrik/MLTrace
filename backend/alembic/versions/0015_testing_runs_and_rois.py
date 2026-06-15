"""add testing runs and roi definitions

Revision ID: 0015_testing_runs_and_rois
Revises: 0014_train_test_dataset_label
Create Date: 2026-06-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0015_testing_runs_and_rois"
down_revision = "0014_train_test_dataset_label"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "roi_definitions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("x", sa.Integer(), nullable=False),
        sa.Column("y", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "testing_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("training_run_id", sa.Integer(), sa.ForeignKey("training_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column(
            "training_dataset_id",
            sa.Integer(),
            sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("roi_id", sa.Integer(), sa.ForeignKey("roi_definitions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="running"),
        sa.Column("started_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("image_count", sa.Integer(), nullable=True),
        sa.Column("score_mean", sa.Float(), nullable=True),
        sa.Column("score_min", sa.Float(), nullable=True),
        sa.Column("score_max", sa.Float(), nullable=True),
        sa.Column("full_mse_mean", sa.Float(), nullable=True),
        sa.Column("roi_mse_mean", sa.Float(), nullable=True),
        sa.Column("results_path", sa.Text(), nullable=True),
        sa.Column("results_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("training_run_name", sa.String(length=255), nullable=False),
        sa.Column("training_pipeline_name", sa.String(length=255), nullable=False),
        sa.Column("training_dataset_name", sa.String(length=255), nullable=False),
        sa.Column("preprocessing_pipeline_name", sa.String(length=255), nullable=False),
        sa.Column("method_type", sa.String(length=128), nullable=False),
        sa.Column("method_family", sa.String(length=128), nullable=False),
        sa.Column("training_mode", sa.String(length=64), nullable=False),
        sa.Column("artifact_kind", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("roi_name", sa.String(length=255), nullable=True),
        sa.Column("roi_geometry", json_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_testing_runs_status", "testing_runs", ["status"])
    op.create_index("ix_testing_runs_created_at", "testing_runs", ["created_at"])
    op.create_index("ix_testing_runs_score_mean", "testing_runs", ["score_mean"])

    op.create_table(
        "testing_run_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("full_mse", sa.Float(), nullable=False),
        sa.Column("roi_mse", sa.Float(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_testing_run_results_run_position", "testing_run_results", ["testing_run_id", "position"])
    op.create_index("ix_testing_run_results_timestamp", "testing_run_results", ["timestamp"])
    op.create_index("ix_testing_run_results_score", "testing_run_results", ["score"])


def downgrade() -> None:
    op.drop_index("ix_testing_run_results_score", table_name="testing_run_results")
    op.drop_index("ix_testing_run_results_timestamp", table_name="testing_run_results")
    op.drop_index("ix_testing_run_results_run_position", table_name="testing_run_results")
    op.drop_table("testing_run_results")
    op.drop_index("ix_testing_runs_score_mean", table_name="testing_runs")
    op.drop_index("ix_testing_runs_created_at", table_name="testing_runs")
    op.drop_index("ix_testing_runs_status", table_name="testing_runs")
    op.drop_table("testing_runs")
    op.drop_table("roi_definitions")
