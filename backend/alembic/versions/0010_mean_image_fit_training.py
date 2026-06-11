"""mark mean image as fit-training method

Revision ID: 0010_mean_image_fit_training
Revises: 0009_methods_refactor
Create Date: 2026-06-10 00:00:00
"""

from alembic import op


revision = "0010_mean_image_fit_training"
down_revision = "0009_methods_refactor"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE method_configurations
        SET requires_training = TRUE,
            supports_training_pipeline = TRUE,
            training_mode = 'fit',
            artifact_kind = 'mean_image',
            method_family = 'statistical_baseline'
        WHERE method_type = 'mean_image'
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE method_configurations
        SET requires_training = FALSE,
            supports_training_pipeline = FALSE,
            training_mode = 'fit',
            artifact_kind = 'mean_image',
            method_family = 'statistical_baseline'
        WHERE method_type = 'mean_image'
        """
    )
