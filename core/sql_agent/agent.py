"""Translate natural-language analytics questions into safe, scoped, read-only SQL."""

import json

from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from core.llm.client import call_groq
from core.sql_agent.guardrails import validate_and_scope_sql
from core.sql_agent.db import execute_readonly_sql

SQL_AGENT_SYSTEM = """You are a SQL generation assistant for a hospital analytics tool. Convert the \
user's natural language question into a single read-only PostgreSQL SELECT statement.

You may ONLY query these two views — no other tables exist for you:

v_image_findings (one row per detected condition per chest X-ray analysis):
  interaction_id         text
  conversation_id        text
  doctor_id              text     -- do NOT filter by this yourself; row access is scoped automatically
  interaction_timestamp  timestamptz
  latency_ms             integer
  condition              text     -- one of: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
                                      Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,
                                      Pleural_Thickening, Pneumonia, Pneumothorax
  confidence_score       float    -- CNN sigmoid probability for this condition, 0.0-1.0
  dominant_zones         json     -- JSON array of zone codes, e.g. ["RLZ", "LLZ"]; use
                                      dominant_zones::jsonb @> '["RLZ"]' for membership checks,
                                      NOT array operators like ANY() — this is JSON, not an array
  aligned                boolean  -- whether GradCAM activation matched the expected clinical zone
  low_confidence_flag    boolean  -- true when no condition passed threshold for that image
  is_correct             boolean, nullable  -- doctor feedback: agree/disagree/no feedback yet
  feedback_comment       text, nullable

v_text_interactions (one row per free-text clinical question asked):
  interaction_id         text
  conversation_id        text
  doctor_id              text     -- do NOT filter by this yourself
  interaction_timestamp  timestamptz
  latency_ms             integer
  is_correct             boolean, nullable
  feedback_comment       text, nullable

Rules:
- Output exactly one SELECT statement — no other statement types, no semicolon stacking
- Never reference any table other than the two above; nothing else exists for you
- Use standard PostgreSQL syntax
- Do not add a doctor_id filter yourself — access scoping is applied automatically after generation
- If the question cannot be answered with these two views, explain why in "explanation" and set
  "sql" to "SELECT 1 WHERE false"
- Write "explanation" in the same language as the question (Indonesian or English)

Output ONLY valid JSON with this exact schema:
{"sql": "<SELECT statement>", "explanation": "<1-2 sentence plain-language description>"}"""

SQL_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sql":         {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["sql", "explanation"],
    "additionalProperties": False,
}


def _build_retry_prompt(question: str, failed_sql: str, error: str) -> str:
    """Build a follow-up prompt asking the LLM to correct a SQL query that failed validation/execution."""
    return (
        f"Original question: {question}\n\n"
        f"Your previous SQL failed:\n{failed_sql}\n\n"
        f"Error: {error}\n\n"
        f"Generate a corrected SQL query for the same question."
    )


def run_sql_agent(
    question: str,
    role: str,
    doctor_id: str,
    max_rows: int | None = None,
    max_retries: int = 1,
) -> dict:
    """Generate SQL from a question, validate/scope it, execute, and retry once on failure."""
    max_rows = max_rows or settings.sql_agent_max_rows

    raw = call_groq(
        SQL_AGENT_SYSTEM, f"Question: {question}",
        schema=SQL_AGENT_SCHEMA, schema_name="sql_query",
    )
    parsed = json.loads(raw)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            safe_sql = validate_and_scope_sql(parsed["sql"], role, doctor_id, max_rows)
            rows = execute_readonly_sql(safe_sql)
            return {
                "sql_executed": safe_sql,
                "explanation": parsed["explanation"],
                "rows": rows,
                "row_count": len(rows),
            }
        except (ValueError, SQLAlchemyError) as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break
            retry_prompt = _build_retry_prompt(question, parsed["sql"], last_error)
            raw = call_groq(
                SQL_AGENT_SYSTEM, retry_prompt,
                schema=SQL_AGENT_SCHEMA, schema_name="sql_query",
            )
            parsed = json.loads(raw)

    return {
        "sql_executed": None,
        "explanation": f"Query could not be executed safely: {last_error}",
        "rows": [],
        "row_count": 0,
    }