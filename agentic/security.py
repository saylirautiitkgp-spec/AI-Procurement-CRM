"""Secret redaction and untrusted-content handling.

Everything that reaches a trace, a log, an API response or an LLM prompt
passes through here first.
"""

from __future__ import annotations

import os
import re
from typing import Any

_SECRET_ENV_NAMES = (
    "SUPABASE_KEY",
    "SUPABASE_URL",
    "TAVILY_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
)

_SECRET_PATTERNS = [
    re.compile(r"sb_secret_[A-Za-z0-9_\-]+"),
    re.compile(r"sb_publishable_[A-Za-z0-9_\-]+"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|authorization|password)\b"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{12,}"
    ),
]

_SECRET_KEY_NAMES = re.compile(
    r"(?i)(api[_-]?key|secret|password|authorization|access[_-]?token|auth[_-]?token|bearer|private[_-]?key)")

REDACTED = "[REDACTED]"


def _known_secret_values() -> list[str]:
    values = []
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            values.append(value)
    return values


def redact(text: str) -> str:
    """Remove anything that looks like a credential from a string."""
    if not isinstance(text, str) or not text:
        return text
    for value in _known_secret_values():
        text = text.replace(value, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_obj(obj: Any, max_str: int = 2000, _depth: int = 0) -> Any:
    """Recursively redact and truncate a JSON-like structure."""
    if _depth > 8:
        return "[TRUNCATED_DEPTH]"
    if isinstance(obj, str):
        out = redact(obj)
        if len(out) > max_str:
            out = out[:max_str] + "...[truncated]"
        return out
    if isinstance(obj, dict):
        clean = {}
        for key, value in obj.items():
            if isinstance(key, str) and _SECRET_KEY_NAMES.search(key):
                clean[key] = REDACTED
            else:
                clean[key] = redact_obj(value, max_str, _depth + 1)
        return clean
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, max_str, _depth + 1) for v in obj[:200]]
    return obj


def wrap_untrusted(text: str, source: str = "") -> str:
    """Mark retrieved web text as data, never as instructions."""
    safe = (text or "").replace("</untrusted_web_content>", "[/untrusted_web_content]")
    safe = redact(safe)
    return (
        f'<untrusted_web_content source="{redact(source)[:300]}">\n'
        f"{safe}\n</untrusted_web_content>"
    )
