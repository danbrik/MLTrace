"""store preprocessing pipeline preview folder

Revision ID: 0006_pipeline_preview_folder
Revises: 0005_pipeline_design_resolution
Create Date: 2026-05-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_pipeline_preview_folder"
down_revision = "0005_pipeline_design_resolution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("preprocessing_pipelines")}
    if "preview_folder_id" not in columns:
        op.add_column("preprocessing_pipelines", sa.Column("preview_folder_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("preprocessing_pipelines")}
    if "preview_folder_id" in columns:
        op.drop_column("preprocessing_pipelines", "preview_folder_id")
