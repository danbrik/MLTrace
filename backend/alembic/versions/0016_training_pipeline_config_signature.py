"""add config_signature to training pipelines

Revision ID: 0016_training_pipeline_config_signature
Revises: 0015_testing_runs_and_rois
Create Date: 2026-06-15 02:00:00
"""

import hashlib
import json

from alembic import op
import sqlalchemy as sa


revision = "0016_training_pipeline_config_signature"
down_revision = "0015_testing_runs_and_rois"
branch_labels = None
depends_on = None


def _signature(dataset_ids, preprocessing_id, method_id, shuffle, training_parameters) -> str:
    # Must mirror app.services.training_pipeline_signature.
    canonical = {
        "datasets": sorted({int(value) for value in dataset_ids}),
        "preprocessing": int(preprocessing_id),
        "method": int(method_id),
        "shuffle": bool(shuffle),
        "params": training_parameters or {},
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def upgrade() -> None:
    op.add_column("training_pipelines", sa.Column("config_signature", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_training_pipelines_config_signature", "training_pipelines", ["config_signature"]
    )

    # Backfill existing rows by recomputing the signature from stored config.
    bind = op.get_bind()
    pipelines = bind.execute(
        sa.text(
            "SELECT id, preprocessing_pipeline_id, method_configuration_id, shuffle, training_parameters "
            "FROM training_pipelines"
        )
    ).fetchall()
    for row in pipelines:
        dataset_ids = [
            item[0]
            for item in bind.execute(
                sa.text(
                    "SELECT training_dataset_id FROM training_pipeline_datasets "
                    "WHERE training_pipeline_id = :pid"
                ),
                {"pid": row[0]},
            ).fetchall()
        ]
        params = row[4]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (TypeError, ValueError):
                params = {}
        signature = _signature(dataset_ids, row[1], row[2], row[3], params or {})
        bind.execute(
            sa.text("UPDATE training_pipelines SET config_signature = :sig WHERE id = :pid"),
            {"sig": signature, "pid": row[0]},
        )


def downgrade() -> None:
    op.drop_index("ix_training_pipelines_config_signature", table_name="training_pipelines")
    op.drop_column("training_pipelines", "config_signature")
