"""Canonical mapping of a company's website: what may be stored in `companies.website`, and the derived domain.

* `companies.website` holds the SITE ROOT of the verified official site (scheme + host), never a deep or regional page.
* A deep URL may be turned into its root only when the root page itself was retrieved in this run and the domain's
  retrieved pages demonstrably belong to the company (the same identity test the Company agent uses). Otherwise the
  website stays EMPTY and the candidate is reported for a human decision.
* `company_profile.website_domain` is the registrable domain of that verified root; it is derived, never invented.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from agentic import evidence as ev


def site_root(url: str) -> str | None:
    canon = ev.canonical_url(str(url or ""))
    if not canon:
        return None
    u = urlsplit(canon)
    return f"{u.scheme}://{u.netloc}/" if u.scheme in ("http", "https") and u.netloc else None


def is_root(url: str) -> bool:
    canon = ev.canonical_url(str(url or ""))
    root = site_root(url)
    return bool(canon and root and canon.rstrip("/") == root.rstrip("/"))


def website_domain(url: str) -> str | None:
    root = site_root(url)
    return ev.registrable_domain(root) if root else None


def canonical_website(value: str, ledger: dict, company_name: str) -> tuple[str | None, str]:
    """(root URL or None, reason). A claimed root is accepted as is (its evidence was validated against the ledger); a
    deep page is promoted to its root only if the root was retrieved and the domain verifies for this company."""
    root = site_root(value)
    if root is None:
        return None, "invalid_url"
    if is_root(value):
        return root, "claimed_value_is_the_site_root"
    if ev.find_doc(ledger or {}, root) is None:
        return None, "site_root_not_retrieved_in_this_run"
    from agentic.agents.company import verify_identity
    ok, why, _ = verify_identity(ledger, root, company_name)
    return (root, "root_retrieved_and_domain_verified") if ok else (None, why)
