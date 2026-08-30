"""add scheduled image distribution runs

Revision ID: 0049_image_distribution_runs
Revises: 0048_redundancy_analysis
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0049_image_distribution_runs"
down_revision = "0048_redundancy_analysis"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "image_distribution_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("preprocessing_pipeline_id", sa.Integer(), sa.ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"), nullable=False),
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
        sa.Column("cache_key", sa.String(64)),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("csv_path", sa.Text()),
        sa.Column("result", JSON_TYPE),
        sa.Column("training_dataset_name", sa.String(255), nullable=False),
        sa.Column("usage_label", sa.String(32), nullable=False),
        sa.Column("preprocessing_pipeline_name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_image_distribution_runs_status", "image_distribution_runs", ["status"])
    op.create_index("ix_image_distribution_runs_created_at", "image_distribution_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_image_distribution_runs_created_at", table_name="image_distribution_runs")
    op.drop_index("ix_image_distribution_runs_status", table_name="image_distribution_runs")
    op.drop_table("image_distribution_runs")
