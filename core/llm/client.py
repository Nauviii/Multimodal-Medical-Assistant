"""Thin wrapper around Groq API for LLM Call 1 and 2."""

from groq import Groq
from config.settings import settings

_client = Groq(api_key=settings.groq_api_key)


def call_groq(
    system_prompt: str,
    user_prompt: str,
    schema: dict | None = None,
    schema_name: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Call Groq chat completion; uses strict JSON schema mode when schema is given."""
    kwargs = {
        "model": settings.groq_model,
        "temperature": settings.llm_temperature,
        "max_tokens":  max_tokens or settings.llm_max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }

    response = _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content