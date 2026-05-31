"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("root_path", sa.Text(), nullable=False, unique=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("timestamp_regex", sa.Text(), nullable=True),
        sa.Column("timestamp_format", sa.String(length=128), nullable=True),
        sa.Column("timestamp_example", sa.String(length=255), nullable=True),
        sa.Column("scan_error", sa.Text(), nullable=True),
        sa.Column("scan_summary", json_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "dataset_folders",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("first_timestamp", sa.DateTime(timezone=False), nullable=True),
        sa.Column("last_timestamp", sa.DateTime(timezone=False), nullable=True),
        sa.Column("extension_summary", json_type, nullable=True),
        sa.Column("resolution_summary", json_type, nullable=True),
        sa.Column("cadence_summary", json_type, nullable=True),
        sa.UniqueConstraint("dataset_id", "relative_path", name="uq_folder_per_dataset"),
    )

    op.create_table(
        "dataset_images",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("folder_id", sa.Integer(), sa.ForeignKey("dataset_folders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False, unique=True),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("extension", sa.String(length=16), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("timestamp_raw", sa.String(length=255), nullable=False),
        sa.Column("timestamp_parsed", sa.DateTime(timezone=False), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("modified_time", sa.DateTime(timezone=False), nullable=True),
    )
    op.create_index("ix_dataset_images_dataset_timestamp", "dataset_images", ["dataset_id", "timestamp_parsed"])
    op.create_index("ix_dataset_images_folder_timestamp", "dataset_images", ["folder_id", "timestamp_parsed"])

    op.create_table(
        "training_datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False),
    )

    op.create_table(
        "training_dataset_rules",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "training_dataset_id",
            sa.Integer(),
            sa.ForeignKey("training_datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("folder_id", sa.Integer(), sa.ForeignKey("dataset_folders.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("end_timestamp", sa.DateTime(timezone=False), nullable=False),
        sa.Column("stride", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("training_dataset_rules")
    op.drop_table("training_datasets")
    op.drop_index("ix_dataset_images_folder_timestamp", table_name="dataset_images")
    op.drop_index("ix_dataset_images_dataset_timestamp", table_name="dataset_images")
    op.drop_table("dataset_images")
    op.drop_table("dataset_folders")
    op.drop_table("datasets")
