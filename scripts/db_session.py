"""SQLAlchemy session factory and FastAPI dependency for DB access."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session 

from config.settings import settings

_engine = create_engine(settings.database_url)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a DB session, closing it after the request."""
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()