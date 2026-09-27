"""Project-wide reusable time range presets."""
from alembic import op
import sqlalchemy as sa

revision = "0062_time_range_presets"
down_revision = "0061_reference_image"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "time_range_presets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("name_key", sa.String(765), nullable=False, unique=True),
        sa.Column("start", sa.DateTime(), nullable=False),
        sa.Column("end", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade():
    op.drop_table("time_range_presets")
