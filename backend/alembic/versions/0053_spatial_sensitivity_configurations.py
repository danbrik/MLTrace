"""add reusable spatial sensitivity configurations

Revision ID: 0053_spatial_configs
Revises: 0052_spatial_sensitivity
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0053_spatial_configs"
down_revision = "0052_spatial_sensitivity"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "spatial_sensitivity_configurations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text()),
        sa.Column("config", JSON_TYPE, nullable=False),
        sa.Column("config_signature", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_spatial_sensitivity_configurations_config_signature", "spatial_sensitivity_configurations", ["config_signature"])
    with op.batch_alter_table("spatial_sensitivity_runs") as batch:
        batch.add_column(sa.Column("configuration_id", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("config_signature", sa.String(64), nullable=False, server_default=""))
        batch.create_foreign_key("fk_spatial_run_configuration", "spatial_sensitivity_configurations", ["configuration_id"], ["id"], ondelete="SET NULL")
        batch.create_index("ix_spatial_sensitivity_runs_config_signature", ["config_signature"])


def downgrade() -> None:
    with op.batch_alter_table("spatial_sensitivity_runs") as batch:
        batch.drop_index("ix_spatial_sensitivity_runs_config_signature")
        batch.drop_constraint("fk_spatial_run_configuration", type_="foreignkey")
        batch.drop_column("config_signature")
        batch.drop_column("configuration_id")
    op.drop_index("ix_spatial_sensitivity_configurations_config_signature", table_name="spatial_sensitivity_configurations")
    op.drop_table("spatial_sensitivity_configurations")
