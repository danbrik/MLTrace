"""Persist spatial worker identity and cancellation deadlines."""
from alembic import op
import sqlalchemy as sa

revision = "0054_spatial_abort"
down_revision = "0053_spatial_configs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("spatial_sensitivity_runs") as batch:
        batch.add_column(sa.Column("process_started_at", sa.Float(), nullable=True))
        batch.add_column(sa.Column("process_project_id", sa.String(255), nullable=True))
        batch.add_column(sa.Column("abort_requested_at", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("force_killed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("spatial_sensitivity_runs") as batch:
        for name in ("force_killed_at", "abort_requested_at", "process_project_id", "process_started_at"):
            batch.drop_column(name)
