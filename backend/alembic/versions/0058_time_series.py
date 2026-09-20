"""CSV time series datasets and tagged temporal splits."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0058_time_series"
down_revision = "0057_representation_analysis"
branch_labels = None
depends_on = None


def upgrade():
    json_type = sa.JSON().with_variant(JSONB(), "postgresql")
    op.create_table(
        "time_series_datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("source_csv", sa.LargeBinary(), nullable=False),
        sa.Column("columns", json_type, nullable=False),
        sa.Column("selected_columns", json_type, nullable=False),
        sa.Column("timestamp_column", sa.Text(), nullable=False),
        sa.Column("timestamp_format", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("start", sa.String(64), nullable=False),
        sa.Column("end", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "time_series_splits",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("time_series_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tags", json_type, nullable=False),
        sa.Column("intervals", json_type, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_time_series_splits_dataset_id", "time_series_splits", ["dataset_id"])


def downgrade():
    op.drop_table("time_series_splits")
    op.drop_table("time_series_datasets")
