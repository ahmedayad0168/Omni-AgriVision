from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional
import logging
from src.data.base import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        if database_url.startswith("sqlite"):
            # SQLite does not support connection-pool kwargs
            self.engine = create_engine(database_url, echo=False)
        else:
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False,
            )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created (if not exist).")