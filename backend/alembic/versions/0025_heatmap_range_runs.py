"""add heatmap range runs (batch heatmap video jobs)

Revision ID: 0025_heatmap_range_runs
Revises: 0024_heatmap_error_matrix
Create Date: 2026-06-18 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision: str = "0025_heatmap_range_runs"
down_revision: str | None = "0024_heatmap_error_matrix"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "heatmap_range_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "testing_run_id",
            sa.Integer(),
            sa.ForeignKey("testing_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("enqueued_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("gpu_index", sa.Integer(), nullable=True),
        sa.Column("device", sa.String(length=32), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("log_path", sa.Text(), nullable=True),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("stride", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("scale_mode", sa.String(length=16), nullable=False, server_default="per_frame"),
        sa.Column("global_vmax", sa.Float(), nullable=True),
        sa.Column("frame_count", sa.Integer(), nullable=True),
        sa.Column("done_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frames_dir", sa.Text(), nullable=True),
        sa.Column("config_signature", sa.String(length=64), nullable=False),
        sa.Column("testing_run_name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_heatmap_range_runs_status", "heatmap_range_runs", ["status"])
    op.create_index("ix_heatmap_range_runs_created_at", "heatmap_range_runs", ["created_at"])
    op.create_index("ix_heatmap_range_runs_signature", "heatmap_range_runs", ["config_signature"])


def downgrade() -> None:
    op.drop_index("ix_heatmap_range_runs_signature", table_name="heatmap_range_runs")
    op.drop_index("ix_heatmap_range_runs_created_at", table_name="heatmap_range_runs")
    op.drop_index("ix_heatmap_range_runs_status", table_name="heatmap_range_runs")
    op.drop_table("heatmap_range_runs")
