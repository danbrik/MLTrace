"""Persistent time-series data quality analyses."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0055_data_quality'
down_revision = '0054_spatial_abort'
branch_labels = None
depends_on = None


def upgrade():
    js = sa.JSON().with_variant(JSONB(), 'postgresql')
    op.create_table('data_quality_analyses',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('source_id', sa.Integer(), sa.ForeignKey('redundancy_csv_sources.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('cache_key', sa.String(64), nullable=False, unique=True),
        sa.Column('parameters', js, nullable=False),
        sa.Column('job_status', sa.String(24), nullable=False, server_default='queued'),
        sa.Column('progress', sa.Float(), nullable=False, server_default='0'),
        sa.Column('stage', sa.String(255), nullable=False, server_default='Queued'),
        sa.Column('elapsed_seconds', sa.Float(), nullable=False, server_default='0'),
        sa.Column('eta_seconds', sa.Float()),
        sa.Column('started_at', sa.DateTime()),
        sa.Column('error_message', sa.Text()),
        sa.Column('cancel_requested', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('result', js), sa.Column('missing_runs', js),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()))
    op.create_index('ix_data_quality_analyses_source_id', 'data_quality_analyses', ['source_id'])


def downgrade():
    op.drop_table('data_quality_analyses')
