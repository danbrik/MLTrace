"""rename model configurations to method configurations

Revision ID: 0009_methods_refactor
Revises: 0008_model_configurations
Create Date: 2026-06-10 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0009_methods_refactor"
down_revision = "0008_model_configurations"
branch_labels = None
depends_on = None


def _tables() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    tables = _tables()
    if "model_configurations" in tables and "method_configurations" not in tables:
        op.rename_table("model_configurations", "method_configurations")
    tables = _tables()
    if "model_configuration_parameters" in tables and "method_configuration_parameters" not in tables:
        op.rename_table("model_configuration_parameters", "method_configuration_parameters")

    columns = _columns("method_configurations")
    with op.batch_alter_table("method_configurations") as batch:
        if "architecture_type" in columns and "method_type" not in columns:
            batch.alter_column("architecture_type", new_column_name="method_type", existing_type=sa.String(length=128))
        if "architecture_version" in columns and "method_version" not in columns:
            batch.alter_column("architecture_version", new_column_name="method_version", existing_type=sa.String(length=64))
        if "model_graph" in columns and "method_graph" not in columns:
            batch.alter_column("model_graph", new_column_name="method_graph", existing_type=sa.JSON())
        if "model_config" in columns and "method_config" not in columns:
            batch.alter_column("model_config", new_column_name="method_config", existing_type=sa.JSON())

    columns = _columns("method_configurations")
    with op.batch_alter_table("method_configurations") as batch:
        if "method_family" not in columns:
            batch.add_column(sa.Column("method_family", sa.String(length=128), nullable=True))
        if "training_mode" not in columns:
            batch.add_column(sa.Column("training_mode", sa.String(length=64), nullable=True))
        if "artifact_kind" not in columns:
            batch.add_column(sa.Column("artifact_kind", sa.String(length=128), nullable=True))
        if "validation" not in columns:
            batch.add_column(sa.Column("validation", sa.JSON(), nullable=True))

    op.execute(
        """
        UPDATE method_configurations
        SET method_family = CASE
            WHEN method_type IN ('cnn_autoencoder', 'cnn_vae') THEN 'neural_reconstruction'
            WHEN method_type = 'mean_image' THEN 'statistical_baseline'
            ELSE 'custom'
        END
        WHERE method_family IS NULL
        """
    )
    op.execute(
        """
        UPDATE method_configurations
        SET training_mode = CASE
            WHEN method_type IN ('cnn_autoencoder', 'cnn_vae') THEN 'gradient'
            WHEN method_type = 'mean_image' THEN 'fit'
            ELSE 'none'
        END
        WHERE training_mode IS NULL
        """
    )
    op.execute(
        """
        UPDATE method_configurations
        SET artifact_kind = CASE
            WHEN method_type IN ('cnn_autoencoder', 'cnn_vae') THEN 'weights'
            WHEN method_type = 'mean_image' THEN 'mean_image'
            ELSE 'custom'
        END
        WHERE artifact_kind IS NULL
        """
    )
    op.execute(
        """
        UPDATE method_configurations
        SET supports_training_pipeline = CASE WHEN training_mode = 'gradient' THEN TRUE ELSE FALSE END
        """
    )

    columns = _columns("method_configuration_parameters")
    with op.batch_alter_table("method_configuration_parameters") as batch:
        if "model_configuration_id" in columns and "method_configuration_id" not in columns:
            batch.alter_column(
                "model_configuration_id",
                new_column_name="method_configuration_id",
                existing_type=sa.Integer(),
            )

    op.execute("UPDATE method_configuration_parameters SET path = replace(path, 'architecture_type', 'method_type')")
    op.execute("UPDATE method_configuration_parameters SET path = replace(path, 'architecture_version', 'method_version')")
    op.execute("UPDATE method_configuration_parameters SET path = replace(path, 'model_graph', 'method_graph')")
    op.execute("UPDATE method_configuration_parameters SET path = replace(path, 'model_config', 'method_config')")


def downgrade() -> None:
    columns = _columns("method_configuration_parameters")
    with op.batch_alter_table("method_configuration_parameters") as batch:
        if "method_configuration_id" in columns and "model_configuration_id" not in columns:
            batch.alter_column(
                "method_configuration_id",
                new_column_name="model_configuration_id",
                existing_type=sa.Integer(),
            )

    columns = _columns("method_configurations")
    with op.batch_alter_table("method_configurations") as batch:
        if "validation" in columns:
            batch.drop_column("validation")
        if "artifact_kind" in columns:
            batch.drop_column("artifact_kind")
        if "training_mode" in columns:
            batch.drop_column("training_mode")
        if "method_family" in columns:
            batch.drop_column("method_family")
        if "method_config" in columns and "model_config" not in columns:
            batch.alter_column("method_config", new_column_name="model_config", existing_type=sa.JSON())
        if "method_graph" in columns and "model_graph" not in columns:
            batch.alter_column("method_graph", new_column_name="model_graph", existing_type=sa.JSON())
        if "method_version" in columns and "architecture_version" not in columns:
            batch.alter_column("method_version", new_column_name="architecture_version", existing_type=sa.String(length=64))
        if "method_type" in columns and "architecture_type" not in columns:
            batch.alter_column("method_type", new_column_name="architecture_type", existing_type=sa.String(length=128))

    tables = _tables()
    if "method_configuration_parameters" in tables and "model_configuration_parameters" not in tables:
        op.rename_table("method_configuration_parameters", "model_configuration_parameters")
    tables = _tables()
    if "method_configurations" in tables and "model_configurations" not in tables:
        op.rename_table("method_configurations", "model_configurations")
