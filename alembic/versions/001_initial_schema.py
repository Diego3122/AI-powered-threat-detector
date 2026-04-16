
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.BigInteger(), nullable=False),
        sa.Column('window_id', sa.String(255), nullable=True),
        sa.Column('model_type', sa.String(50), nullable=True),
        sa.Column('model_score', sa.Float(), nullable=True),
        sa.Column('threshold', sa.Float(), nullable=True),
        sa.Column('triggered', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('explanation_summary', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alerts_timestamp', 'alerts', ['timestamp'], unique=False)
    op.create_index('idx_alerts_model_type', 'alerts', ['model_type'], unique=False)
    op.create_index('idx_alerts_triggered', 'alerts', ['triggered'], unique=False)

    # Create audit_log table
    op.create_table(
        'audit_log',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('action', sa.String(100), nullable=True),
        sa.Column('target', sa.String(255), nullable=True),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_log_user_id', 'audit_log', ['user_id'], unique=False)
    op.create_index('idx_audit_log_action', 'audit_log', ['action'], unique=False)
    op.create_index('idx_audit_log_created_at', 'audit_log', ['created_at'], unique=False)

    # Create alert_investigations table
    op.create_table(
        'alert_investigations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('alert_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['alert_id'], ['alerts.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alert_investigations_alert_id', 'alert_investigations', ['alert_id'], unique=False)
    op.create_index('idx_alert_investigations_status', 'alert_investigations', ['status'], unique=False)

    # Create models table
    op.create_table(
        'models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('version', sa.String(50), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('roc_auc', sa.Float(), nullable=True),
        sa.Column('n_samples', sa.Integer(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('model_metadata', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_models_model_type', 'models', ['model_type'], unique=False)
    op.create_index('idx_models_active', 'models', ['active'], unique=False)

    # Create performance_metrics table
    op.create_table(
        'performance_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('hour_timestamp', sa.BigInteger(), nullable=True),
        sa.Column('alert_count', sa.Integer(), nullable=True),
        sa.Column('avg_score', sa.Float(), nullable=True),
        sa.Column('model_type', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_performance_metrics_hour_timestamp', 'performance_metrics', ['hour_timestamp'], unique=False)
    op.create_index('idx_performance_metrics_model_type', 'performance_metrics', ['model_type'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_performance_metrics_model_type', table_name='performance_metrics')
    op.drop_index('idx_performance_metrics_hour_timestamp', table_name='performance_metrics')
    op.drop_table('performance_metrics')
    op.drop_index('idx_models_active', table_name='models')
    op.drop_index('idx_models_model_type', table_name='models')
    op.drop_table('models')
    op.drop_index('idx_alert_investigations_status', table_name='alert_investigations')
    op.drop_index('idx_alert_investigations_alert_id', table_name='alert_investigations')
    op.drop_table('alert_investigations')
    op.drop_index('idx_audit_log_created_at', table_name='audit_log')
    op.drop_index('idx_audit_log_action', table_name='audit_log')
    op.drop_index('idx_audit_log_user_id', table_name='audit_log')
    op.drop_table('audit_log')
    op.drop_index('idx_alerts_triggered', table_name='alerts')
    op.drop_index('idx_alerts_model_type', table_name='alerts')
    op.drop_index('idx_alerts_timestamp', table_name='alerts')
    op.drop_table('alerts')
