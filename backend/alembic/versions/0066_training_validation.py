"""Separate validation datasets; existing runs retain their legacy split."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = '0066_training_validation'
down_revision = '0065_temporal_difference'
branch_labels = None
depends_on = None


def upgrade():
    for table in ('training_pipelines', 'training_runs'):
        op.add_column(table, sa.Column('validation_mode', sa.String(32), nullable=False, server_default='legacy_fraction'))
        op.add_column(table, sa.Column('validation_shuffle', sa.Boolean(), nullable=False, server_default=sa.false()))
    json_type = sa.JSON().with_variant(JSONB(), 'postgresql')
    for name in ('validation_dataset_ids', 'validation_dataset_names'):
        op.add_column('training_runs', sa.Column(name, json_type, nullable=False, server_default='[]'))
    op.add_column('training_runs', sa.Column('validation_sample_count', sa.Integer()))
    op.create_table('training_pipeline_validation_datasets',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('training_pipeline_id', sa.Integer(), sa.ForeignKey('training_pipelines.id', ondelete='CASCADE'), nullable=False),
        sa.Column('training_dataset_id', sa.Integer(), sa.ForeignKey('training_datasets.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.UniqueConstraint('training_pipeline_id', 'training_dataset_id', name='uq_pipeline_validation_dataset'),
        sa.UniqueConstraint('training_pipeline_id', 'position', name='uq_pipeline_validation_position'))


def downgrade():
    op.drop_table('training_pipeline_validation_datasets')
    for name in ('validation_sample_count', 'validation_dataset_names', 'validation_dataset_ids'):
        op.drop_column('training_runs', name)
    for table in ('training_runs', 'training_pipelines'):
        op.drop_column(table, 'validation_shuffle')
        op.drop_column(table, 'validation_mode')
