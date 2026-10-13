"""Stable identity of a verified claim (leaf module: imported by the change set and by the proposal builders)."""

from __future__ import annotations

import hashlib
import json


def claim_key(agent: str, c: dict) -> str:
    """Lets a reviewed change-set item be traced back to exactly one verified claim."""
    ident = {"a": agent, "f": str(c.get("field") or c.get("kind") or "").lower(),
             "v": json.dumps(c.get("value"), sort_keys=True, default=str).lower(),
             "n": str(c.get("name") or "").lower(), "s": str(c.get("supplier") or "").lower(),
             "c": str(c.get("component") or "").lower()}
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
