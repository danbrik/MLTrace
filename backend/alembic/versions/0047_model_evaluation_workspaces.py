"""add model-centred A/B evaluation workspaces

Revision ID: 0047_model_evaluation_workspaces
Revises: 0046_model_evaluations
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0047_model_evaluation_workspaces"
down_revision = "0046_model_evaluations"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    with op.batch_alter_table("testing_runs") as batch:
        batch.add_column(sa.Column("result_revision", sa.Integer(), nullable=False, server_default="0"))

    op.create_table(
        "evaluation_model_workspaces",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_run_id", sa.Integer(), sa.ForeignKey("training_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("artifact_signature", sa.String(64), nullable=False),
        sa.Column("sep_median", sa.Float()), sa.Column("sep_min", sa.Float()),
        sa.Column("drift_mean", sa.Float()), sa.Column("drift_max", sa.Float()),
        sa.Column("active_drift_calculation_id", sa.Integer()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("training_run_id", "artifact_signature", name="uq_evaluation_workspace_artifact"),
    )
    op.create_index("ix_evaluation_workspaces_training_run", "evaluation_model_workspaces", ["training_run_id"])

    op.create_table(
        "evaluation_separation_layouts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False), sa.Column("description", sa.Text()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("training_dataset_id", "name", name="uq_eval_sep_layout_dataset_name"),
    )
    op.create_index("ix_eval_sep_layouts_dataset", "evaluation_separation_layouts", ["training_dataset_id"])
    op.create_table(
        "evaluation_separation_pairs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("layout_id", sa.Integer(), sa.ForeignKey("evaluation_separation_layouts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pair_key", sa.String(64), nullable=False), sa.Column("name", sa.String(255), nullable=False),
        sa.Column("normal_start", sa.DateTime(), nullable=False), sa.Column("normal_end", sa.DateTime(), nullable=False),
        sa.Column("anomaly_start", sa.DateTime(), nullable=False), sa.Column("anomaly_end", sa.DateTime(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("layout_id", "pair_key", name="uq_eval_sep_pair_key"),
    )

    op.create_table(
        "evaluation_drift_layouts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("training_dataset_id", sa.Integer(), sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False), sa.Column("description", sa.Text()),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("reference_start", sa.DateTime(), nullable=False), sa.Column("reference_end", sa.DateTime(), nullable=False),
        sa.Column("analysis_start", sa.DateTime(), nullable=False), sa.Column("analysis_end", sa.DateTime(), nullable=False),
        sa.Column("bucket_seconds", sa.Float(), nullable=False),
        sa.Column("reference_exclusion_action", sa.String(24), nullable=False, server_default="filter_points"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("training_dataset_id", "name", name="uq_eval_drift_layout_dataset_name"),
    )
    op.create_index("ix_eval_drift_layouts_dataset", "evaluation_drift_layouts", ["training_dataset_id"])
    op.create_table(
        "evaluation_drift_exclusions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("layout_id", sa.Integer(), sa.ForeignKey("evaluation_drift_layouts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("exclusion_key", sa.String(64), nullable=False), sa.Column("name", sa.String(255), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False), sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("layout_id", "exclusion_key", name="uq_eval_drift_exclusion_key"),
    )
    op.create_table(
        "evaluation_drift_buckets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("layout_id", sa.Integer(), sa.ForeignKey("evaluation_drift_layouts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("bucket_key", sa.String(64), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False), sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(24), nullable=False, server_default="include"),
        sa.UniqueConstraint("layout_id", "bucket_key", name="uq_eval_drift_bucket_key"),
    )

    op.create_table(
        "evaluation_separation_calculations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("evaluation_model_workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("layout_id", sa.Integer(), sa.ForeignKey("evaluation_separation_layouts.id", ondelete="SET NULL")),
        sa.Column("layout_version", sa.Integer(), nullable=False), sa.Column("layout_snapshot", JSON_TYPE, nullable=False),
        sa.Column("score_series", sa.String(64), nullable=False), sa.Column("artifact_signature", sa.String(64), nullable=False),
        sa.Column("source_result_revision", sa.Integer(), nullable=False), sa.Column("stale", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_eval_sep_calcs_workspace", "evaluation_separation_calculations", ["workspace_id"])
    op.create_table(
        "evaluation_separation_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("calculation_id", sa.Integer(), sa.ForeignKey("evaluation_separation_calculations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pair_key", sa.String(64), nullable=False), sa.Column("pair_name", sa.String(255), nullable=False),
        sa.Column("normal_start", sa.DateTime(), nullable=False), sa.Column("normal_end", sa.DateTime(), nullable=False),
        sa.Column("anomaly_start", sa.DateTime(), nullable=False), sa.Column("anomaly_end", sa.DateTime(), nullable=False),
        sa.Column("normal_median", sa.Float(), nullable=False), sa.Column("normal_mad", sa.Float(), nullable=False),
        sa.Column("robust_scale", sa.Float(), nullable=False), sa.Column("normal_point_count", sa.Integer(), nullable=False),
        sa.Column("anomaly_point_count", sa.Integer(), nullable=False), sa.Column("separation", sa.Float(), nullable=False),
        sa.Column("separation_p95", sa.Float()), sa.Column("included", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("calculation_id", "pair_key", name="uq_eval_sep_result_pair"),
    )

    op.create_table(
        "evaluation_drift_calculations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("evaluation_model_workspaces.id", ondelete="CASCADE"), nullable=False),
        sa.Column("testing_run_id", sa.Integer(), sa.ForeignKey("testing_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("layout_id", sa.Integer(), sa.ForeignKey("evaluation_drift_layouts.id", ondelete="SET NULL")),
        sa.Column("layout_version", sa.Integer(), nullable=False), sa.Column("layout_snapshot", JSON_TYPE, nullable=False),
        sa.Column("score_series", sa.String(64), nullable=False), sa.Column("artifact_signature", sa.String(64), nullable=False),
        sa.Column("source_result_revision", sa.Integer(), nullable=False),
        sa.Column("reference_iqr", sa.Float(), nullable=False), sa.Column("reference_point_count", sa.Integer(), nullable=False),
        sa.Column("drift_mean", sa.Float()), sa.Column("drift_max", sa.Float()),
        sa.Column("stale", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_eval_drift_calcs_workspace", "evaluation_drift_calculations", ["workspace_id"])
    op.create_table(
        "evaluation_drift_bucket_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("calculation_id", sa.Integer(), sa.ForeignKey("evaluation_drift_calculations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("bucket_key", sa.String(64), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False), sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("original_point_count", sa.Integer(), nullable=False), sa.Column("used_point_count", sa.Integer(), nullable=False),
        sa.Column("exclusion_overlap", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("reason", sa.Text()),
        sa.Column("wasserstein_1", sa.Float()), sa.Column("normalized_drift", sa.Float()),
        sa.Column("included", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("calculation_id", "bucket_key", name="uq_eval_drift_result_bucket"),
    )


def downgrade() -> None:
    op.drop_table("evaluation_drift_bucket_results")
    op.drop_index("ix_eval_drift_calcs_workspace", table_name="evaluation_drift_calculations")
    op.drop_table("evaluation_drift_calculations")
    op.drop_table("evaluation_separation_results")
    op.drop_index("ix_eval_sep_calcs_workspace", table_name="evaluation_separation_calculations")
    op.drop_table("evaluation_separation_calculations")
    op.drop_table("evaluation_drift_buckets")
    op.drop_table("evaluation_drift_exclusions")
    op.drop_index("ix_eval_drift_layouts_dataset", table_name="evaluation_drift_layouts")
    op.drop_table("evaluation_drift_layouts")
    op.drop_table("evaluation_separation_pairs")
    op.drop_index("ix_eval_sep_layouts_dataset", table_name="evaluation_separation_layouts")
    op.drop_table("evaluation_separation_layouts")
    op.drop_index("ix_evaluation_workspaces_training_run", table_name="evaluation_model_workspaces")
    op.drop_table("evaluation_model_workspaces")
    with op.batch_alter_table("testing_runs") as batch:
        batch.drop_column("result_revision")
