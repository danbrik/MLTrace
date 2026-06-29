"""add testing run inference config

Revision ID: 0030_testing_run_inference_config
Revises: 0029_stae_result_metadata
Create Date: 2026-06-28 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0030_testing_run_inference_config"
down_revision: str | None = "0029_stae_result_metadata"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column("testing_runs", sa.Column("inference_config", json_type, nullable=True))


def downgrade() -> None:
    op.drop_column("testing_runs", "inference_config")
