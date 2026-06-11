"""add saved model configurations

Revision ID: 0008_model_configurations
Revises: 0007_dataset_folder_image_metadata
Create Date: 2026-06-10 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0008_model_configurations"
down_revision = "0007_dataset_folder_image_metadata"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "model_configurations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("architecture_type", sa.String(length=128), nullable=False),
        sa.Column("architecture_version", sa.String(length=64), nullable=False),
        sa.Column("requires_training", sa.Boolean(), nullable=False),
        sa.Column("supports_training_pipeline", sa.Boolean(), nullable=False),
        sa.Column("builder_kind", sa.String(length=128), nullable=False),
        sa.Column("model_graph", json_type, nullable=False),
        sa.Column("model_config", json_type, nullable=False),
        sa.Column("training_config", json_type, nullable=False),
        sa.Column("inference_config", json_type, nullable=False),
        sa.Column("diagram", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "model_configuration_parameters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "model_configuration_id",
            sa.Integer(),
            sa.ForeignKey("model_configurations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("value_type", sa.String(length=32), nullable=False),
        sa.Column("value_text", sa.Text(), nullable=True),
        sa.Column("value_number", sa.Float(), nullable=True),
        sa.Column("value_bool", sa.Boolean(), nullable=True),
    )
    op.create_index(
        "ix_model_config_parameters_path_text",
        "model_configuration_parameters",
        ["path", "value_text"],
    )
    op.create_index(
        "ix_model_config_parameters_path_number",
        "model_configuration_parameters",
        ["path", "value_number"],
    )
    op.create_index(
        "ix_model_config_parameters_path_bool",
        "model_configuration_parameters",
        ["path", "value_bool"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_config_parameters_path_bool", table_name="model_configuration_parameters")
    op.drop_index("ix_model_config_parameters_path_number", table_name="model_configuration_parameters")
    op.drop_index("ix_model_config_parameters_path_text", table_name="model_configuration_parameters")
    op.drop_table("model_configuration_parameters")
    op.drop_table("model_configurations")
