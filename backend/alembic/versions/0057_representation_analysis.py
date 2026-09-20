"""Persist exploratory representation runs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0057_representation_analysis"
down_revision = "0056_characterization"
branch_labels = None
depends_on = None


def upgrade():
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.create_table(
        "representation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("training_dataset_name", sa.String(255), nullable=False),
        *[sa.Column(name, json_type, nullable=False) for name in ("config", "dataset_snapshot", "pipeline_snapshot")],
        sa.Column("model_snapshot", json_type), sa.Column("result", json_type),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("current_step", sa.String(64), nullable=False, server_default="queued"),
        sa.Column("processed_images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_images", sa.Integer()),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_message", sa.Text()), sa.Column("queue_rank", sa.Integer()),
        *[sa.Column(name, sa.DateTime()) for name in ("enqueued_at", "started_at", "ended_at", "heartbeat_at")],
        sa.Column("duration_seconds", sa.Float()), sa.Column("device", sa.String(32)),
        sa.Column("gpu_index", sa.Integer()), sa.Column("pid", sa.Integer()), sa.Column("log_path", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_representation_runs_status", "representation_runs", ["status"])


def downgrade():
    op.drop_table("representation_runs")
