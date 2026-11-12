from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from shoppilot.config import settings

url = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(url)
Session = sessionmaker(engine)
