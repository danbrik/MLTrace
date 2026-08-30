"""add scalable image distribution progress

Revision ID: 0050_image_distribution_scaling
Revises: 0049_image_distribution_runs
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0050_image_distribution_scaling"
down_revision = "0049_image_distribution_runs"
branch_labels = None
depends_on = None

JSON_TYPE = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    columns = (
        sa.Column("phase_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("phase_total", sa.Integer()),
        sa.Column("processed_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger()),
        sa.Column("throughput_images_per_second", sa.Float()),
        sa.Column("throughput_mb_per_second", sa.Float()),
        sa.Column("eta_seconds", sa.Float()),
        sa.Column("effective_worker_count", sa.Integer()),
        sa.Column("calibration_results", JSON_TYPE),
        sa.Column("stride_projections", JSON_TYPE),
        sa.Column("heartbeat_at", sa.DateTime()),
        sa.Column("resume_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("manifest_path", sa.Text()),
    )
    for column in columns:
        op.add_column("image_distribution_runs", column)


def downgrade() -> None:
    for name in (
        "manifest_path", "resume_count", "heartbeat_at", "stride_projections",
        "calibration_results", "effective_worker_count", "eta_seconds",
        "throughput_mb_per_second", "throughput_images_per_second", "total_bytes",
        "processed_bytes", "phase_total", "phase_processed",
    ):
        op.drop_column("image_distribution_runs", name)
