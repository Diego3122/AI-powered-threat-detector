"""Add feature_schema column and index to alerts table.

The ORM (models.py) defines feature_schema on Alert but the initial
migration never created the column.  Alert ingestion via alert_router
sets feature_schema on every INSERT, so this is a pipeline-breaking gap.

Revision ID: 002
Revises: 001
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('alerts', sa.Column('feature_schema', sa.String(50), nullable=True))
    op.create_index('idx_alerts_feature_schema', 'alerts', ['feature_schema'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_alerts_feature_schema', table_name='alerts')
    op.drop_column('alerts', 'feature_schema')
