"""Time-series architectures, frozen pipelines, scheduled runs and epoch metrics."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0059_time_series_training"
down_revision = "0058_time_series"
branch_labels = None
depends_on = None


def upgrade():
    j = sa.JSON().with_variant(JSONB(), "postgresql")
    def stamp(name):
        return sa.Column(name, sa.DateTime(), nullable=False, server_default=sa.func.now())
    op.create_table("time_series_models",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False), sa.Column("config", j, nullable=False),
        sa.Column("provenance", j, nullable=False), sa.Column("template_key", sa.String(32), unique=True),
        stamp("created_at"), stamp("updated_at"))
    op.create_table("time_series_pipelines",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(255), nullable=False),
        *[sa.Column(name, sa.Integer(), sa.ForeignKey(target + ".id", ondelete="RESTRICT"), nullable=False) for name, target in
          (("dataset_id", "time_series_datasets"), ("split_id", "time_series_splits"), ("model_id", "time_series_models"))],
        sa.Column("window_length", sa.Integer(), nullable=False), sa.Column("training", j, nullable=False),
        sa.Column("snapshot", j, nullable=False), stamp("created_at"), stamp("updated_at"))
    op.create_table("time_series_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("pipeline_id", sa.Integer(), sa.ForeignKey("time_series_pipelines.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False), sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("snapshot", j, nullable=False), sa.Column("status", sa.String(32), nullable=False),
        sa.Column("current_step", sa.String(64), nullable=False),
        sa.Column("checkpoint_selection", sa.String(64), nullable=False),
        sa.Column("epoch", sa.Integer(), nullable=False), sa.Column("epochs", sa.Integer(), nullable=False),
        sa.Column("processed_windows", sa.Integer(), nullable=False), sa.Column("total_windows", sa.Integer()),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False), sa.Column("selected_epoch", sa.Integer()),
        *[sa.Column(name, sa.Float()) for name in ("train_loss", "val_loss", "selected_metric", "duration_seconds")],
        sa.Column("checkpoint", j), sa.Column("result", j), sa.Column("error_message", sa.Text()),
        *[sa.Column(name, sa.Integer()) for name in ("queue_rank", "gpu_index", "pid")],
        *[sa.Column(name, sa.DateTime()) for name in ("enqueued_at", "started_at", "ended_at", "heartbeat_at")],
        sa.Column("device", sa.String(32)), sa.Column("log_path", sa.Text()), stamp("created_at"))
    op.create_index("ix_time_series_runs_status", "time_series_runs", ["status"])
    op.create_table("time_series_epoch_metrics",
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("time_series_runs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("epoch", sa.Integer(), primary_key=True), sa.Column("train_loss", sa.Float(), nullable=False),
        sa.Column("val_loss", sa.Float()), sa.Column("details", j, nullable=False))


def downgrade():
    for table in ("time_series_epoch_metrics", "time_series_runs", "time_series_pipelines", "time_series_models"):
        op.drop_table(table)
