"""Read-only SQLAlchemy engine for executing agent-generated SQL against the curated views.

Uses settings.sql_agent_readonly_url (the sql_agent_readonly Postgres role), never the main
app's full-access database_url — table/column access is restricted at the database level.
"""

from sqlalchemy import create_engine, text

from config.settings import settings

_readonly_engine = create_engine(settings.sql_agent_readonly_url)


def execute_readonly_sql(sql: str) -> list[dict]:
    """Execute a validated read-only SQL statement and return rows as plain dicts."""
    with _readonly_engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]