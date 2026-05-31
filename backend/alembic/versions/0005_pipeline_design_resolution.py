"""store preprocessing pipeline design resolution

Revision ID: 0005_pipeline_design_resolution
Revises: 0004_pipeline_name_unique
Create Date: 2026-05-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "0005_pipeline_design_resolution"
down_revision = "0004_pipeline_name_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("preprocessing_pipelines", sa.Column("input_width", sa.Integer(), nullable=True))
    op.add_column("preprocessing_pipelines", sa.Column("input_height", sa.Integer(), nullable=True))
    op.add_column("preprocessing_pipelines", sa.Column("output_width", sa.Integer(), nullable=True))
    op.add_column("preprocessing_pipelines", sa.Column("output_height", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("preprocessing_pipelines", "output_height")
    op.drop_column("preprocessing_pipelines", "output_width")
    op.drop_column("preprocessing_pipelines", "input_height")
    op.drop_column("preprocessing_pipelines", "input_width")
