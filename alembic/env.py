from logging.config import fileConfig

from alembic import context
from shoppilot.db.models import Base
from shoppilot.db.session import engine

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
