"""add single-model evaluation persistence

Revision ID: 0046_model_evaluations
Revises: 0045_training_run_checkpoints
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0046_model_evaluations"
down_revision: str | None = "0045_training_run_checkpoints"
branch_labels = None
depends_on = None

_json_type = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("training_runs") as batch:
        batch.add_column(sa.Column("artifact_signature", sa.String(length=64), nullable=True))
    with op.batch_alter_table("testing_runs") as batch:
        batch.add_column(sa.Column("artifact_signature", sa.String(length=64), nullable=True))

    op.create_table(
        "evaluation_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("normal_window_duration_seconds", sa.Float(), nullable=False, server_default="3600"),
        sa.Column("normal_window_buffer_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("drift_window_seconds", sa.Float(), nullable=False, server_default="3600"),
        sa.Column("false_alarm_horizon_seconds", sa.Float(), nullable=False, server_default="3600"),
        sa.Column("anticipation_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("epsilon", sa.Float(), nullable=False, server_default="0.000000000001"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_evaluation_profile_name"),
    )

    op.create_table(
        "evaluation_label_sets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "training_dataset_id",
            sa.Integer(),
            sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "training_dataset_id", "name", name="uq_evaluation_label_set_dataset_name"
        ),
    )
    op.create_index(
        "ix_evaluation_label_sets_dataset", "evaluation_label_sets", ["training_dataset_id"]
    )

    op.create_table(
        "evaluation_label_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "label_set_id",
            sa.Integer(),
            sa.ForeignKey("evaluation_label_sets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=255)),
        sa.Column("category", sa.String(length=128)),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.UniqueConstraint("label_set_id", "event_id", name="uq_evaluation_label_event_key"),
    )
    op.create_index(
        "ix_evaluation_label_events_set_range",
        "evaluation_label_events",
        ["label_set_id", "start_timestamp", "end_timestamp"],
    )

    op.create_table(
        "model_evaluations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="draft"),
        sa.Column(
            "evaluation_testing_run_id",
            sa.Integer(),
            sa.ForeignKey("testing_runs.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "reference_testing_run_id",
            sa.Integer(),
            sa.ForeignKey("testing_runs.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "calibration_testing_run_id",
            sa.Integer(),
            sa.ForeignKey("testing_runs.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "profile_id",
            sa.Integer(),
            sa.ForeignKey("evaluation_profiles.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "label_set_id",
            sa.Integer(),
            sa.ForeignKey("evaluation_label_sets.id", ondelete="RESTRICT"),
        ),
        sa.Column("score_series", sa.String(length=32), nullable=False, server_default="score"),
        sa.Column("evaluation_start_timestamp", sa.DateTime()),
        sa.Column("evaluation_end_timestamp", sa.DateTime()),
        sa.Column("reference_start_timestamp", sa.DateTime()),
        sa.Column("reference_end_timestamp", sa.DateTime()),
        sa.Column("calibration_start_timestamp", sa.DateTime()),
        sa.Column("calibration_end_timestamp", sa.DateTime()),
        sa.Column("selected_categories", _json_type),
        sa.Column("normal_window_overrides", _json_type),
        sa.Column("profile_overrides", _json_type),
        sa.Column("profile_snapshot", _json_type),
        sa.Column("label_snapshot", _json_type),
        sa.Column("source_snapshot", _json_type),
        sa.Column("config_signature", sa.String(length=64)),
        sa.Column("separation_status", sa.String(length=24), nullable=False, server_default="not_calculated"),
        sa.Column("separation_config_signature", sa.String(length=64)),
        sa.Column("separation_result", _json_type),
        sa.Column("separation_error", sa.Text()),
        sa.Column("drift_status", sa.String(length=24), nullable=False, server_default="not_calculated"),
        sa.Column("drift_config_signature", sa.String(length=64)),
        sa.Column("drift_result", _json_type),
        sa.Column("drift_error", sa.Text()),
        sa.Column("detection_status", sa.String(length=24), nullable=False, server_default="not_calculated"),
        sa.Column("detection_config_signature", sa.String(length=64)),
        sa.Column("detection_result", _json_type),
        sa.Column("detection_error", sa.Text()),
        sa.Column("warnings", _json_type),
        sa.Column("sep_median", sa.Float()),
        sa.Column("sep_min", sa.Float()),
        sa.Column("drift_mean", sa.Float()),
        sa.Column("drift_max", sa.Float()),
        sa.Column("event_recall", sa.Float()),
        sa.Column("median_delay_seconds", sa.Float()),
        sa.Column("frame_fpr", sa.Float()),
        sa.Column("false_alarm_rate_t0", sa.Float()),
        sa.Column("active_quantile", sa.Float(), nullable=False, server_default="0.999"),
        sa.Column("finalized_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_model_evaluations_status", "model_evaluations", ["status"])
    op.create_index("ix_model_evaluations_created_at", "model_evaluations", ["created_at"])
    op.create_index(
        "ix_model_evaluations_evaluation_run", "model_evaluations", ["evaluation_testing_run_id"]
    )
    op.create_index(
        "ix_model_evaluations_reference_run", "model_evaluations", ["reference_testing_run_id"]
    )
    op.create_index(
        "ix_model_evaluations_calibration_run", "model_evaluations", ["calibration_testing_run_id"]
    )


def downgrade() -> None:
    op.drop_table("model_evaluations")
    op.drop_table("evaluation_label_events")
    op.drop_table("evaluation_label_sets")
    op.drop_table("evaluation_profiles")
    with op.batch_alter_table("testing_runs") as batch:
        batch.drop_column("artifact_signature")
    with op.batch_alter_table("training_runs") as batch:
        batch.drop_column("artifact_signature")
