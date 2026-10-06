"""Record best validation and exported model epochs."""
from alembic import op
import sqlalchemy as sa
revision = '0067_training_epoch_selection'
down_revision = '0066_training_validation'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('training_runs', sa.Column('best_epoch', sa.Integer(), nullable=True))
    op.add_column('training_runs', sa.Column('selected_epoch', sa.Integer(), nullable=True))


def downgrade():
    op.drop_column('training_runs', 'selected_epoch')
    op.drop_column('training_runs', 'best_epoch')
