"""store sampled dataset folder image metadata

Revision ID: 0007_dataset_folder_image_metadata
Revises: 0006_pipeline_preview_folder
Create Date: 2026-06-09 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0007_dataset_folder_image_metadata"
down_revision = "0006_pipeline_preview_folder"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("dataset_folders")}
    if "image_metadata" not in columns:
        op.add_column("dataset_folders", sa.Column("image_metadata", sa.JSON(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("dataset_folders")}
    if "image_metadata" in columns:
        op.drop_column("dataset_folders", "image_metadata")
