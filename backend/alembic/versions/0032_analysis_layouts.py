"""analysis layouts

Revision ID: 0032_analysis_layouts
Revises: 0031_inspect_runs
Create Date: 2026-06-29 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

try:
    from sqlalchemy.dialects import postgresql
except ImportError:  # pragma: no cover
    postgresql = None


revision = "0032_analysis_layouts"
down_revision = "0031_inspect_runs"
branch_labels = None
depends_on = None


def _json_type():
    if op.get_bind().dialect.name == "postgresql" and postgresql is not None:
        return postgresql.JSONB(astext_type=sa.Text())
    return sa.JSON()


def upgrade() -> None:
    op.create_table(
        "analysis_layouts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("layout", _json_type(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )


def downgrade() -> None:
    op.drop_table("analysis_layouts")
