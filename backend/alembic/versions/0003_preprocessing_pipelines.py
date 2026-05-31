"""add preprocessing pipelines

Revision ID: 0003_preprocessing_pipelines
Revises: 0002_training_dataset_multi_dataset
Create Date: 2026-05-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0003_preprocessing_pipelines"
down_revision = "0002_training_dataset_multi_dataset"
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "preprocessing_pipelines",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("graph", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("preprocessing_pipelines")

