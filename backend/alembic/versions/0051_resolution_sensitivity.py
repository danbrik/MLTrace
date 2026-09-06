"""add resolution sensitivity runs

Revision ID: 0051_resolution_sensitivity
Revises: 0050_image_distribution_scaling
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0051_resolution_sensitivity"
down_revision = "0050_image_distribution_scaling"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "resolution_sensitivity_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("label_set_id", sa.Integer(), sa.ForeignKey("evaluation_label_sets.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("enqueued_at", sa.DateTime()),
        sa.Column("queue_rank", sa.Integer()),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("ended_at", sa.DateTime()),
        sa.Column("duration_seconds", sa.Float()),
        sa.Column("error_message", sa.Text()),
        sa.Column("gpu_index", sa.Integer()),
        sa.Column("device", sa.String(32)),
        sa.Column("pid", sa.Integer()),
        sa.Column("log_path", sa.Text()),
        sa.Column("current_step", sa.String(64), nullable=False, server_default="queued"),
        sa.Column("total_images", sa.Integer()),
        sa.Column("processed_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("successful_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("heartbeat_at", sa.DateTime()),
        sa.Column("config", JSON_TYPE, nullable=False),
        sa.Column("pipeline_snapshot", JSON_TYPE, nullable=False),
        sa.Column("training_dataset_name", sa.String(255), nullable=False),
        sa.Column("label_set_name", sa.String(255)),
        sa.Column("data_range", sa.Float()),
        sa.Column("detail_csv_path", sa.Text()),
        sa.Column("summary_csv_path", sa.Text()),
        sa.Column("result", JSON_TYPE),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_resolution_sensitivity_runs_status", "resolution_sensitivity_runs", ["status"])
    op.create_index("ix_resolution_sensitivity_runs_created_at", "resolution_sensitivity_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_resolution_sensitivity_runs_created_at", table_name="resolution_sensitivity_runs")
    op.drop_index("ix_resolution_sensitivity_runs_status", table_name="resolution_sensitivity_runs")
    op.drop_table("resolution_sensitivity_runs")
