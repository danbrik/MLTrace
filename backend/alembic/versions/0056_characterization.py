"""Persist statistical and temporal characterization jobs."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0056_characterization'
down_revision = '0055_data_quality'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('characterization_runs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('analysis_id', sa.Integer(), sa.ForeignKey('data_quality_analyses.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('job_status', sa.String(24), nullable=False, server_default='queued'),
        sa.Column('progress', sa.Float(), nullable=False, server_default='0'),
        sa.Column('stage', sa.String(255), nullable=False, server_default='Queued'),
        sa.Column('elapsed_seconds', sa.Float(), nullable=False, server_default='0'),
        sa.Column('eta_seconds', sa.Float()), sa.Column('error_message', sa.Text()),
        sa.Column('cancel_requested', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('result', sa.JSON().with_variant(JSONB(), 'postgresql')),
        sa.Column('artifact_path', sa.Text()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('analysis_id', 'version', name='uq_characterization_version'))
    op.create_index('ix_characterization_runs_analysis_id', 'characterization_runs', ['analysis_id'])


def downgrade():
    op.drop_table('characterization_runs')
