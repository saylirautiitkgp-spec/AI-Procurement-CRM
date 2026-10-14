"""Page fetching with SSRF, size, timeout, scheme and content-type limits."""

from __future__ import annotations

import re
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Callable, Protocol
from urllib.parse import urljoin, urlsplit

from agentic.evidence import registrable_domain
from agentic.guards.url_policy import Resolver, UrlBlocked, check_url, default_resolver

USER_AGENT = "ProcurementCRM-ResearchBot/0.1 (public pages only; contact: project owner)"
_TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class FetchError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


@dataclass
class FetchedPage:
    url: str
    final_url: str
    status: int
    title: str
    text: str
    links: list[dict] = field(default_factory=list)   # [{"url","text"}] same registrable domain
    truncated: bool = False


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


_LATIN_LABELS = {"iso-8859-1", "latin-1", "latin1", "us-ascii", "ascii", "iso8859-1"}


def decode_body(raw: bytes, content_type: str = "") -> str:
    """Bytes -> text without ever producing U+FFFD or mojibake.

    `requests` reports ISO-8859-1 for 'text/html' without a charset (an RFC default nobody honours), and the old code
    decoded with that label and errors='replace'. Order now: the declared charset (header, then <meta>) when it decodes
    strictly - except that a latin-1/ascii label on a body that is valid UTF-8 with non-ASCII bytes is a mislabel and
    UTF-8 wins; then strict UTF-8; then strict cp1252. Only if everything fails are undecodable bytes DROPPED (never
    replaced), so no replacement character can enter the evidence ledger."""
    declared = []
    m = re.search(r"charset\s*=\s*[\"']?([\w\-]+)", content_type or "", re.I)
    if m:
        declared.append(m.group(1).lower())
    meta = re.search(rb"<meta[^>]+charset\s*=\s*[\"']?([\w\-]+)", raw[:4096], re.I)
    if meta:
        declared.append(meta.group(1).decode("ascii", "ignore").lower())
    utf8 = None
    try:
        utf8 = raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    non_ascii = any(b > 127 for b in raw)
    for enc in declared:
        if enc in _LATIN_LABELS and utf8 is not None and non_ascii:
            return utf8.replace("\ufffd", "")
        try:
            return raw.decode(enc).replace("\ufffd", "")
        except (UnicodeDecodeError, LookupError):
            continue
    if utf8 is not None:
        return utf8.replace("\ufffd", "")
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="ignore")


def extract_page(html: str, base_url: str, max_links: int = 25) -> tuple[str, str, list[dict]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    base_domain = registrable_domain(base_url)
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"].strip())
        parts = urlsplit(href)
        if parts.scheme not in ("http", "https") or registrable_domain(href) != base_domain:
            continue
        key = href.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        links.append({"url": key, "text": re.sub(r"\s+", " ", a.get_text(" ")).strip()[:80]})
        if len(links) >= max_links:
            break
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = re.sub(r"[ \t]+", " ", soup.get_text("\n"))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return title, text, links


class SafeHttpFetcher:
    def __init__(self, *, max_bytes: int = 500_000, timeout: float = 10.0, max_redirects: int = 3,
                 resolver: Resolver = default_resolver, session=None, respect_robots: bool = True,
                 max_text_chars: int = 20_000):
        import requests
        self.max_bytes, self.timeout, self.max_redirects = max_bytes, timeout, max_redirects
        self.resolver = resolver
        self.session = session or requests.Session()
        self.respect_robots = respect_robots
        self.max_text_chars = max_text_chars
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                check_url(origin + "/robots.txt", self.resolver)
                resp = self.session.get(origin + "/robots.txt", timeout=self.timeout,
                                        headers={"User-Agent": USER_AGENT}, allow_redirects=False)
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                    self._robots[origin] = rp
                else:
                    self._robots[origin] = None
            except Exception:
                self._robots[origin] = None
        rp = self._robots[origin]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def fetch(self, url: str) -> FetchedPage:
        current, hops = url, 0
        while True:
            try:
                check_url(current, self.resolver)
            except UrlBlocked as exc:
                raise FetchError("blocked_" + exc.reason) from None
            if not self._allowed_by_robots(current):
                raise FetchError("disallowed_by_robots_txt")
            try:
                resp = self.session.get(current, timeout=self.timeout, stream=True,
                                        allow_redirects=False, headers={"User-Agent": USER_AGENT})
            except Exception as exc:
                raise FetchError("network_error", type(exc).__name__) from None
            if resp.status_code in (301, 302, 303, 307, 308):
                hops += 1
                loc = resp.headers.get("Location")
                if not loc or hops > self.max_redirects:
                    raise FetchError("too_many_redirects")
                current = urljoin(current, loc)
                continue
            break
        if resp.status_code >= 400:
            raise FetchError(f"http_{resp.status_code}")
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype and ctype not in _TEXT_TYPES:
            raise FetchError("unsupported_content_type")
        raw, truncated = b"", False
        for chunk in resp.iter_content(8192):
            raw += chunk
            if len(raw) > self.max_bytes:
                raw, truncated = raw[: self.max_bytes], True
                break
        html = decode_body(raw, resp.headers.get("Content-Type", ""))
        title, text, links = extract_page(html, current)
        if len(text) > self.max_text_chars:
            text, truncated = text[: self.max_text_chars], True
        return FetchedPage(url=url, final_url=current, status=resp.status_code, title=title,
                           text=text, links=links, truncated=truncated)
