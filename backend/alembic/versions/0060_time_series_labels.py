"""Preserve CSV annotation column roles without changing existing datasets."""
from alembic import op
import sqlalchemy as sa

revision = "0060_time_series_labels"
down_revision = "0059_time_series_training"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("time_series_datasets", sa.Column("label_column", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("time_series_datasets", "label_column")
