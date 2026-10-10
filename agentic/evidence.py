"""Evidence ledger and deterministic claim validation.

The ledger records every document the run actually retrieved (search
results, fetched pages). A claim is accepted only if each cited URL is in
the ledger, the quoted snippet really occurs in that document, and the claimed
value is supported by the snippet. Confidence is computed from that evidence,
never taken from the model.

The ledger is a plain dict so it can live in checkpointed graph state:
    {doc_id: {"doc_id","url","title","text","retrieved_at","tool","call_id"}}
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

MIN_SNIPPET_CHARS = 12
MAX_SNIPPET_CHARS = 600

_TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                    "utm_content", "gclid", "fbclid", "ref"}
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu", "nic"}
_STOPWORDS = {"the", "and", "of", "in", "for", "with", "a", "an", "to", "is",
              "are", "at", "by", "on", "its", "has", "was", "as"}
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# text / url normalisation
# --------------------------------------------------------------------------

def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-"))
    return re.sub(r"\s+", " ", text).strip().lower()


def canonical_url(url: str) -> str:
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return ""
    host = parts.hostname.lower()
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query)
                       if k.lower() not in _TRACKING_PARAMS])
    path = parts.path.rstrip("/") or ""
    port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
    return urlunsplit((parts.scheme, host + port, path, query, ""))


def registrable_domain(url_or_host: str) -> str:
    host = url_or_host
    if "://" in url_or_host:
        host = urlsplit(url_or_host).hostname or ""
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def doc_id_for(url: str) -> str:
    return "d" + hashlib.sha1(canonical_url(url).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------
# ledger operations (pure functions over the dict)
# --------------------------------------------------------------------------

def add_doc(ledger: dict, *, url: str, text: str, title: str = "", tool: str = "",
            call_id: str = "", max_docs: int = 40, max_chars: int = 20_000) -> str | None:
    """Add (or extend) a document. Returns doc_id, or None if URL is invalid
    or the ledger is full."""
    canon = canonical_url(url)
    if not canon:
        return None
    did = doc_id_for(canon)
    text = text or ""
    existing = ledger.get(did)
    if existing is None:
        if len(ledger) >= max_docs:
            return None
        ledger[did] = {"doc_id": did, "url": canon, "title": (title or "")[:300],
                       "text": text[:max_chars], "retrieved_at": now_iso(),
                       "tool": tool, "call_id": call_id}
    else:
        if text and norm_text(text) not in norm_text(existing["text"]):
            merged = (existing["text"] + "\n" + text)[:max_chars]
            existing["text"] = merged
        if title and not existing.get("title"):
            existing["title"] = title[:300]
    return did


def known_domains(ledger: dict) -> set[str]:
    return {registrable_domain(d["url"]) for d in ledger.values()}


def find_doc(ledger: dict, url: str) -> dict | None:
    return ledger.get(doc_id_for(url))


def validate_snippet(ledger: dict, url: str, snippet: str) -> tuple[bool, str, str | None]:
    """(ok, reason, doc_id). The snippet must be a real excerpt of a document
    this run retrieved at that URL."""
    snippet = (snippet or "").strip()
    if len(snippet) < MIN_SNIPPET_CHARS:
        return False, "snippet_too_short", None
    if len(snippet) > MAX_SNIPPET_CHARS:
        return False, "snippet_too_long", None
    doc = find_doc(ledger, url)
    if doc is None:
        return False, "url_not_retrieved_in_this_run", None
    if norm_text(snippet) not in norm_text(doc["text"]):
        return False, "snippet_not_found_in_document", doc["doc_id"]
    return True, "ok", doc["doc_id"]


def _tokens(value: str) -> list[str]:
    value = norm_text(value).replace(",", "")
    return [t for t in re.findall(r"[a-z0-9]+", value)
            if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]


def value_supported(value: Any, snippet: str, threshold: float = 0.6) -> bool:
    """Is the claimed value (or each list item) actually stated by the snippet?"""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return bool(value) and all(value_supported(v, snippet, threshold) for v in value)
    if isinstance(value, (int, float)):
        value = str(value)
    value = str(value).strip()
    if not value:
        return False
    if _EMAIL_RE.fullmatch(value):  # emails must appear literally
        return value.lower() in snippet.lower()
    wanted = _tokens(value)
    if not wanted:
        return False
    have = set(_tokens(snippet))
    hits = sum(1 for t in wanted if t in have)
    return hits / len(wanted) >= threshold


def claim_confidence(evidence: Iterable[dict], official_domain: str | None) -> tuple[float, str]:
    """Deterministic confidence from validated evidence only.
    Base 0.40 for one validated snippet; +0.20 if first-party (verified
    official domain); +0.15 per additional independent domain (max two);
    capped at 0.95."""
    ev = list(evidence)
    if not ev:
        return 0.0, "no validated evidence"
    domains = {registrable_domain(e["url"]) for e in ev}
    score, basis = 0.40, ["1 validated snippet"]
    if official_domain and registrable_domain(official_domain) in domains:
        score += 0.20
        basis.append("first-party source")
    extra = min(max(len(domains) - 1, 0), 2)
    if extra:
        score += 0.15 * extra
        basis.append(f"{len(domains)} independent domains")
    return round(min(score, 0.95), 2), "; ".join(basis)


def contains_email(text: str) -> list[str]:
    return _EMAIL_RE.findall(text or "")
