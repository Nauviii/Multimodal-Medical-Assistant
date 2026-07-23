"""Validate and safety-scope LLM-generated SQL before execution against the read-only role."""

import re

import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"v_image_findings", "v_text_interactions"}

FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_read_file", "pg_ls_dir", "pg_read_binary_file",
    "dblink", "dblink_exec", "lo_import", "lo_export",
    "set_config", "current_setting", "pg_terminate_backend", "pg_cancel_backend",
}

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _reject_multiple_statements(sql: str) -> None:
    """Raise ValueError if the raw SQL text contains more than one statement."""
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) > 1:
        raise ValueError("Multiple SQL statements are not allowed")


def _parse_select_only(sql: str) -> exp.Select:
    """Parse SQL and require it to be a single SELECT statement; raises ValueError otherwise."""
    try:
        parsed = sqlglot.parse_one(sql, read="postgres")
    except Exception as exc:
        raise ValueError(f"Could not parse generated SQL: {exc}") from exc

    if not isinstance(parsed, exp.Select):
        raise ValueError("Only SELECT statements are allowed")
    return parsed


def _check_allowed_tables(parsed: exp.Select) -> None:
    """Raise ValueError if any referenced table (including in joins/subqueries) is not an allowed view."""
    referenced = {t.name.lower() for t in parsed.find_all(exp.Table)}
    disallowed = referenced - ALLOWED_TABLES
    if disallowed:
        raise ValueError(f"Query references disallowed table(s): {sorted(disallowed)}")


def _check_forbidden_functions(parsed: exp.Select) -> None:
    """Raise ValueError if the query calls a blocked side-effecting or DoS-prone function."""
    for func in parsed.find_all((exp.Anonymous, exp.Func)):
        name = (getattr(func, "name", "") or "").lower()
        if name in FORBIDDEN_FUNCTIONS:
            raise ValueError(f"Forbidden function call: {name}")


def _apply_doctor_scope(parsed: exp.Select, doctor_id: str) -> None:
    """Inject a doctor_id filter so a doctor role can only ever see their own rows."""
    if not _UUID_RE.match(doctor_id):
        raise ValueError("doctor_id must be a valid UUID")

    scope = exp.EQ(this=exp.column("doctor_id"), expression=exp.Literal.string(doctor_id))
    existing = parsed.args.get("where")
    if existing:
        parsed.set("where", exp.Where(this=exp.And(this=existing.this, expression=scope)))
    else:
        parsed.set("where", exp.Where(this=scope))


def _apply_row_limit(parsed: exp.Select, max_rows: int) -> None:
    """Cap the result size: add a LIMIT if absent, or clamp it down if it exceeds max_rows."""
    existing_limit = parsed.args.get("limit")
    if existing_limit is None:
        parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return
    try:
        current = int(existing_limit.expression.this)
    except (TypeError, ValueError):
        current = max_rows + 1  # malformed limit value: force clamp
    if current > max_rows:
        parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))


def validate_and_scope_sql(sql: str, role: str, doctor_id: str, max_rows: int) -> str:
    """Validate LLM-generated SQL is a safe read-only SELECT, then enforce scoping and row limit."""
    _reject_multiple_statements(sql)
    parsed = _parse_select_only(sql)
    _check_allowed_tables(parsed)
    _check_forbidden_functions(parsed)

    if role == "doctor":
        _apply_doctor_scope(parsed, doctor_id)

    _apply_row_limit(parsed, max_rows)

    return parsed.sql(dialect="postgres")