"""Company-name normalisation and matching.

Replaces the substring-replace normaliser in services/duplicate_detection_service.py
for new code (that one turns "Lincoln" into "ln"); the legacy service is left
untouched for now.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

LEGAL_SUFFIXES = {
    "private", "pvt", "limited", "ltd", "inc", "incorporated", "corp", "corporation",
    "co", "company", "llp", "llc", "plc", "gmbh", "ag", "sa", "bv", "nv", "spa",
    "the", "india", "(india)",
}
_GENERIC = {"technologies", "technology", "systems", "industries", "group", "holdings",
            "solutions", "enterprises", "international", "global"}


def normalize_company_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name or "").lower().replace("&", " and ")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    tokens = [t for t in name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def name_tokens(name: str) -> list[str]:
    return normalize_company_name(name).split()


def distinctive_tokens(name: str) -> list[str]:
    toks = [t for t in name_tokens(name) if t not in _GENERIC and t != "and"]
    return toks or name_tokens(name)


def similarity(a: str, b: str) -> float:
    na, nb = normalize_company_name(a), normalize_company_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return round(max(seq, jac), 4)


def match_type(score: float) -> str:
    if score >= 1.0:
        return "exact"
    if score >= 0.9:
        return "strong"
    if score >= 0.75:
        return "possible"
    return "weak"


_TITLES = {"mr", "mrs", "ms", "miss", "dr", "shri", "smt", "sri", "prof", "er", "ca", "cs"}


def person_tokens(name: str) -> list[str]:
    name = unicodedata.normalize("NFKC", name or "").lower()
    return [t for t in re.sub(r"[^a-z0-9 ]+", " ", name).split() if t not in _TITLES]


def person_similarity(a: str, b: str) -> float:
    """1.0 same tokens (any order/case/title); 0.9 same surname with a matching first name or initial
    ('K. Bhatt' ~ 'Karan Bhatt'); otherwise a string ratio capped below 0.89."""
    ta, tb = person_tokens(a), person_tokens(b)
    if not ta or not tb:
        return 0.0
    if sorted(ta) == sorted(tb):
        return 1.0
    if ta[-1] == tb[-1] and len(ta) > 1 and len(tb) > 1:
        fa, fb = ta[0], tb[0]
        if fa == fb or (len(fa) == 1 and fb.startswith(fa)) or (len(fb) == 1 and fa.startswith(fb)):
            return 0.9
    return min(0.89, round(SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio(), 4))
