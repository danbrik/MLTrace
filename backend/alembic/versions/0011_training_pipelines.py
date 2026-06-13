"""add training pipelines

Revision ID: 0011_training_pipelines
Revises: 0010_mean_image_fit_training
Create Date: 2026-06-11 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0011_training_pipelines"
down_revision = "0010_mean_image_fit_training"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "training_pipelines",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "preprocessing_pipeline_id",
            sa.Integer(),
            sa.ForeignKey("preprocessing_pipelines.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "method_configuration_id",
            sa.Integer(),
            sa.ForeignKey("method_configurations.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("shuffle", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("training_parameters", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "training_pipeline_datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "training_pipeline_id",
            sa.Integer(),
            sa.ForeignKey("training_pipelines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "training_dataset_id",
            sa.Integer(),
            sa.ForeignKey("training_datasets.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint("training_pipeline_id", "training_dataset_id", name="uq_training_pipeline_dataset"),
        sa.UniqueConstraint("training_pipeline_id", "position", name="uq_training_pipeline_position"),
    )


def downgrade() -> None:
    op.drop_table("training_pipeline_datasets")
    op.drop_table("training_pipelines")
