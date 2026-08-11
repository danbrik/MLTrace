"""add anomaly detection runs and events

Revision ID: 0042_anomaly_detection
Revises: 0041_heatmap_range_stae
"""

from alembic import op
import sqlalchemy as sa


revision: str = "0042_anomaly_detection"
down_revision: str | None = "0041_heatmap_range_stae"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_testing_run_results_run_timestamp",
        "testing_run_results",
        ["testing_run_id", "timestamp"],
    )
    op.create_table(
        "anomaly_detection_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("testing_run_name", sa.String(length=255), nullable=False),
        sa.Column("score_series", sa.String(length=32), nullable=False, server_default="score"),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("algorithm_version", sa.String(length=32), nullable=False, server_default="robust_cusum_v1"),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("point_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("warning_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("anomaly_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_anomaly_detection_runs_created_at", "anomaly_detection_runs", ["created_at"])
    op.create_index("ix_anomaly_detection_runs_testing_run", "anomaly_detection_runs", ["testing_run_id"])
    op.create_table(
        "anomaly_detection_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("anomaly_detection_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("warning_start", sa.DateTime(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime()),
        sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("end_reason", sa.String(length=32), nullable=False),
        sa.Column("peak_timestamp", sa.DateTime(), nullable=False),
        sa.Column("max_score", sa.Float(), nullable=False),
        sa.Column("max_robust_z", sa.Float(), nullable=False),
    )
    op.create_index("ix_anomaly_detection_events_run_start", "anomaly_detection_events", ["run_id", "warning_start"])


def downgrade() -> None:
    op.drop_table("anomaly_detection_events")
    op.drop_table("anomaly_detection_runs")
    op.drop_index("ix_testing_run_results_run_timestamp", table_name="testing_run_results")
