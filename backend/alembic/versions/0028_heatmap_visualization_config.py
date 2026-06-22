"""store heatmap visualization configuration

Revision ID: 0028_heatmap_visualization_config
Revises: 0027_heatmap_render_version
Create Date: 2026-06-22 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0028_heatmap_visualization_config"
down_revision: str | None = "0027_heatmap_render_version"
branch_labels = None
depends_on = None

json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.add_column(
        "heatmap_runs",
        sa.Column("visualization_config", json_type, nullable=False, server_default=sa.text("'{}'")),
    )
    op.add_column(
        "heatmap_runs",
        sa.Column("config_signature", sa.String(length=64), nullable=False, server_default=""),
    )
    op.add_column(
        "heatmap_range_runs",
        sa.Column("visualization_config", json_type, nullable=False, server_default=sa.text("'{}'")),
    )
    with op.batch_alter_table("heatmap_runs") as batch:
        batch.drop_constraint("uq_heatmap_result", type_="unique")
        batch.create_unique_constraint(
            "uq_heatmap_result_config",
            ["testing_run_id", "testing_result_id", "config_signature"],
        )


def downgrade() -> None:
    with op.batch_alter_table("heatmap_runs") as batch:
        batch.drop_constraint("uq_heatmap_result_config", type_="unique")
        batch.create_unique_constraint(
            "uq_heatmap_result",
            ["testing_run_id", "testing_result_id"],
        )
    op.drop_column("heatmap_range_runs", "visualization_config")
    op.drop_column("heatmap_runs", "config_signature")
    op.drop_column("heatmap_runs", "visualization_config")
