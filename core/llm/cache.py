"""Redis-backed cache for LLM outputs, keyed on clinical pattern rather than patient identity.

Image path outputs are cached across patients: two patients with the same detected
conditions and the same GradCAM zone pattern receive the same cached explanation,
since the explanation is grounded in static KB content, not patient-specific data.
"""

import json
import hashlib
import redis
from config.settings import settings

_r = redis.from_url(settings.redis_url, decode_responses=True)


def _pattern_key(prefix: str, above_threshold: list[str], gradcam_results: dict) -> str:
    """Build a cache key from condition set and zone pattern, ignoring exact CNN scores."""
    pattern = sorted(
        (
            cond,
            tuple(sorted(gradcam_results[cond]["dominant_zones"])),
            gradcam_results[cond]["aligned"],
        )
        for cond in above_threshold
    )
    digest = hashlib.md5(json.dumps(pattern).encode()).hexdigest()[:16]
    return f"cache:{prefix}:{settings.kb_version}:{digest}"


def get_cached(prefix: str, above_threshold: list[str], gradcam_results: dict) -> dict | None:
    """Return cached LLM output for this clinical pattern, or None on miss."""
    raw = _r.get(_pattern_key(prefix, above_threshold, gradcam_results))
    return json.loads(raw) if raw else None


def set_cached(prefix: str, above_threshold: list[str], gradcam_results: dict, value: dict) -> None:
    """Store LLM output under the clinical pattern key with the configured TTL."""
    key = _pattern_key(prefix, above_threshold, gradcam_results)
    _r.set(key, json.dumps(value), ex=settings.redis_cache_ttl_seconds)


def _text_key(query: str) -> str:
    """Build a cache key for the text Q&A path from a normalized query string."""
    normalized = query.strip().lower()
    digest = hashlib.md5(normalized.encode()).hexdigest()[:16]
    return f"cache:text:{settings.kb_version}:{digest}"


def get_cached_text(query: str) -> dict | None:
    """Return cached text Q&A output for a normalized query, or None on miss."""
    raw = _r.get(_text_key(query))
    return json.loads(raw) if raw else None


def set_cached_text(query: str, value: dict) -> None:
    """Store text Q&A output under the normalized query key with the configured TTL."""
    _r.set(_text_key(query), json.dumps(value), ex=settings.redis_cache_ttl_seconds)