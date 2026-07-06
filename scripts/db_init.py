"""Inisialisasi semua tabel di Supabase PostgreSQL."""

from sqlalchemy import create_engine
from db_models import Base
from config.settings import settings


def init_db() -> None:
    """Buat semua tabel jika belum ada."""
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    print(f"Tables created: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    init_db()