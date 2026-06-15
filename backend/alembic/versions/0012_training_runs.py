"""add training runs

Revision ID: 0012_training_runs
Revises: 0011_training_pipelines
Create Date: 2026-06-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0012_training_runs"
down_revision = "0011_training_pipelines"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "training_pipeline_id",
            sa.Integer(),
            sa.ForeignKey("training_pipelines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("enqueued_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("gpu_index", sa.Integer(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("log_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("epochs_total", sa.Integer(), nullable=True),
        sa.Column("epochs_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("best_val_loss", sa.Float(), nullable=True),
        sa.Column("image_count", sa.Integer(), nullable=True),
        sa.Column("artifact_kind", sa.String(length=64), nullable=True),
        sa.Column("artifact_path", sa.Text(), nullable=True),
        sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("training_pipeline_name", sa.String(length=255), nullable=False),
        sa.Column("method_type", sa.String(length=128), nullable=False),
        sa.Column("method_family", sa.String(length=128), nullable=False),
        sa.Column("training_mode", sa.String(length=64), nullable=False),
        sa.Column("builder_kind", sa.String(length=128), nullable=False),
        sa.Column("preprocessing_pipeline_name", sa.String(length=255), nullable=False),
        sa.Column("dataset_names", json_type, nullable=False),
        sa.Column("dataset_names_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("shuffle", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("input_resolution", sa.String(length=32), nullable=True),
        sa.Column("epochs", sa.Integer(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("training_parameters", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("training_pipeline_id", name="uq_training_run_pipeline"),
    )
    op.create_index("ix_training_runs_status", "training_runs", ["status"])
    op.create_index("ix_training_runs_method_type", "training_runs", ["method_type"])
    op.create_index("ix_training_runs_training_mode", "training_runs", ["training_mode"])
    op.create_index("ix_training_runs_builder_kind", "training_runs", ["builder_kind"])
    op.create_index("ix_training_runs_created_at", "training_runs", ["created_at"])
    op.create_index("ix_training_runs_val_loss", "training_runs", ["val_loss"])
    op.create_index("ix_training_runs_train_loss", "training_runs", ["train_loss"])
    op.create_index("ix_training_runs_duration", "training_runs", ["duration_seconds"])

    op.create_table(
        "training_run_metrics",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "training_run_id",
            sa.Integer(),
            sa.ForeignKey("training_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("epoch", sa.Integer(), nullable=False),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "ix_training_run_metrics_run_epoch", "training_run_metrics", ["training_run_id", "epoch"]
    )


def downgrade() -> None:
    op.drop_index("ix_training_run_metrics_run_epoch", table_name="training_run_metrics")
    op.drop_table("training_run_metrics")
    op.drop_index("ix_training_runs_duration", table_name="training_runs")
    op.drop_index("ix_training_runs_train_loss", table_name="training_runs")
    op.drop_index("ix_training_runs_val_loss", table_name="training_runs")
    op.drop_index("ix_training_runs_created_at", table_name="training_runs")
    op.drop_index("ix_training_runs_builder_kind", table_name="training_runs")
    op.drop_index("ix_training_runs_training_mode", table_name="training_runs")
    op.drop_index("ix_training_runs_method_type", table_name="training_runs")
    op.drop_index("ix_training_runs_status", table_name="training_runs")
    op.drop_table("training_runs")
