"""add inspect runs

Revision ID: 0031_inspect_runs
Revises: 0030_testing_run_inference_config
Create Date: 2026-06-29 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision: str = "0031_inspect_runs"
down_revision: str | None = "0030_testing_run_inference_config"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "inspect_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column(
            "preprocessing_pipeline_id",
            sa.Integer(),
            sa.ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("enqueued_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("device", sa.String(length=32), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("log_path", sa.Text(), nullable=True),
        sa.Column("start_timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("stride", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("fps", sa.Integer(), nullable=False, server_default="12"),
        sa.Column("content_mode", sa.String(length=64), nullable=False, server_default="final_preprocessed_output"),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("done_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frames_dir", sa.Text(), nullable=True),
        sa.Column("video_path", sa.Text(), nullable=True),
        sa.Column("training_dataset_name", sa.String(length=255), nullable=False),
        sa.Column("preprocessing_pipeline_name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=True),
    )
    op.create_index("ix_inspect_runs_status", "inspect_runs", ["status"])
    op.create_index("ix_inspect_runs_created_at", "inspect_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_inspect_runs_created_at", table_name="inspect_runs")
    op.drop_index("ix_inspect_runs_status", table_name="inspect_runs")
    op.drop_table("inspect_runs")
