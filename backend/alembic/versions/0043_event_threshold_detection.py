"""add event-threshold anomaly detection fields

Revision ID: 0043_event_threshold_detection
Revises: 0042_anomaly_detection
"""

from alembic import op
import sqlalchemy as sa


revision: str = "0043_event_threshold_detection"
down_revision: str | None = "0042_anomaly_detection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("anomaly_detection_runs") as batch:
        batch.add_column(sa.Column("threshold_testing_run_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("threshold_testing_run_name", sa.String(length=255), nullable=True))
        batch.add_column(sa.Column("threshold_start_timestamp", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("threshold_end_timestamp", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("resolved_threshold", sa.Float(), nullable=True))
        batch.create_foreign_key(
            "fk_anomaly_detection_runs_threshold_testing_run",
            "testing_runs",
            ["threshold_testing_run_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.create_index(
            "ix_anomaly_detection_runs_threshold_testing_run",
            ["threshold_testing_run_id"],
        )

    with op.batch_alter_table("anomaly_detection_events") as batch:
        batch.alter_column("max_robust_z", existing_type=sa.Float(), nullable=True)
        batch.add_column(sa.Column("duration_seconds", sa.Float(), nullable=True))
        batch.add_column(sa.Column("max_smoothed_score", sa.Float(), nullable=True))
        batch.add_column(sa.Column("mean_smoothed_score", sa.Float(), nullable=True))
        batch.add_column(sa.Column("threshold", sa.Float(), nullable=True))


def downgrade() -> None:
    op.execute(
        "UPDATE anomaly_detection_events SET max_robust_z = 0 "
        "WHERE max_robust_z IS NULL"
    )
    with op.batch_alter_table("anomaly_detection_events") as batch:
        batch.drop_column("threshold")
        batch.drop_column("mean_smoothed_score")
        batch.drop_column("max_smoothed_score")
        batch.drop_column("duration_seconds")
        batch.alter_column("max_robust_z", existing_type=sa.Float(), nullable=False)

    with op.batch_alter_table("anomaly_detection_runs") as batch:
        batch.drop_index("ix_anomaly_detection_runs_threshold_testing_run")
        batch.drop_constraint(
            "fk_anomaly_detection_runs_threshold_testing_run",
            type_="foreignkey",
        )
        batch.drop_column("resolved_threshold")
        batch.drop_column("threshold_end_timestamp")
        batch.drop_column("threshold_start_timestamp")
        batch.drop_column("threshold_testing_run_name")
        batch.drop_column("threshold_testing_run_id")
