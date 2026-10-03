"""Persist temporal differences and editable plot settings."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0065_temporal_difference"
down_revision = "0064_variance_roi"
branch_labels = None
depends_on = None


def upgrade():
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.create_table(
        "temporal_difference_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("training_dataset_name", sa.String(255), nullable=False),
        *[sa.Column(name, json_type, nullable=False) for name in ("config", "dataset_snapshot", "pipeline_snapshot", "plot_settings")],
        sa.Column("result", json_type),
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
    op.create_index("ix_temporal_difference_runs_status", "temporal_difference_runs", ["status"])

    op.create_table(
        "temporal_difference_values",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("temporal_difference_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("delta_seconds", sa.BigInteger(), nullable=False),
        *[sa.Column(name, sa.Text(), nullable=False) for name in ("first_file", "second_file")],
        *[sa.Column(name, sa.String(40), nullable=False) for name in ("first_timestamp", "second_timestamp", "first_utc", "second_utc")],
        sa.Column("value", sa.Float(), nullable=False),
    )
    op.create_index("ix_temporal_values_run_role_delta", "temporal_difference_values", ["run_id", "role", "delta_seconds", "id"])
    op.create_table(
        "temporal_difference_summaries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("temporal_difference_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("delta_seconds", sa.BigInteger(), nullable=False),
        sa.Column("pair_count", sa.Integer(), nullable=False),
        *[sa.Column(name, sa.Float()) for name in ("median", "q1", "q3", "iqr")],
        sa.UniqueConstraint("run_id", "role", "delta_seconds", name="uq_temporal_summary"),
    )


def downgrade():
    op.drop_table("temporal_difference_summaries")
    op.drop_table("temporal_difference_values")
    op.drop_table("temporal_difference_runs")
