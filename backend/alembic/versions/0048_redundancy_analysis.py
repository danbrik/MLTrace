"""add project-local multivariate redundancy analysis

Revision ID: 0048_redundancy_analysis
Revises: 0047_model_evaluation_workspaces
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0048_redundancy_analysis"
down_revision = "0047_model_evaluation_workspaces"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "redundancy_csv_sources",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("delimiter", sa.String(8), nullable=False),
        sa.Column("encoding", sa.String(32), nullable=False, server_default="utf-8"),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("headers", JSON_TYPE, nullable=False),
        sa.Column("column_profiles", JSON_TYPE, nullable=False),
        sa.Column("preview_rows", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("sha256", name="uq_redundancy_csv_source_sha256"),
    )
    op.create_index("ix_redundancy_csv_sources_created", "redundancy_csv_sources", ["created_at"])
    op.create_table(
        "redundancy_analyses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_id", sa.Integer(), sa.ForeignKey("redundancy_csv_sources.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("status", sa.String(24), nullable=False, server_default="draft"),
        sa.Column("job_status", sa.String(24), nullable=False, server_default="not_calculated"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text()),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("time_column", sa.String(255), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(), nullable=False),
        sa.Column("selected_columns", JSON_TYPE, nullable=False),
        sa.Column("config", JSON_TYPE, nullable=False),
        sa.Column("active_cutoff", sa.Float(), nullable=False, server_default="0.9"),
        sa.Column("result", JSON_TYPE),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("finalized_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_redundancy_analyses_source", "redundancy_analyses", ["source_id"])
    op.create_index("ix_redundancy_analyses_status", "redundancy_analyses", ["status", "job_status"])
    op.create_index("ix_redundancy_analyses_created", "redundancy_analyses", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_redundancy_analyses_created", table_name="redundancy_analyses")
    op.drop_index("ix_redundancy_analyses_status", table_name="redundancy_analyses")
    op.drop_index("ix_redundancy_analyses_source", table_name="redundancy_analyses")
    op.drop_table("redundancy_analyses")
    op.drop_index("ix_redundancy_csv_sources_created", table_name="redundancy_csv_sources")
    op.drop_table("redundancy_csv_sources")
