"""unique preprocessing pipeline name

Revision ID: 0004_pipeline_name_unique
Revises: 0003_preprocessing_pipelines
Create Date: 2026-05-31 00:00:00
"""

from alembic import op


revision = "0004_pipeline_name_unique"
down_revision = "0003_preprocessing_pipelines"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "uq_preprocessing_pipelines_name",
        "preprocessing_pipelines",
        ["name"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_preprocessing_pipelines_name", table_name="preprocessing_pipelines")
