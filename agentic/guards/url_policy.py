"""SSRF protection for page fetching.

Only http/https, default ports, no embedded credentials, and the host must
resolve exclusively to public addresses. The check is repeated for every
redirect hop. Residual risk: DNS can change between this check and the
connection (TOCTOU); mitigation is to also pin the checked IP at connect time
if the fetcher is ever moved to a custom transport.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Callable
from urllib.parse import urlsplit

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (None, 80, 443)
_BLOCKED_SUFFIXES = (".local", ".internal", ".localhost", ".lan", ".corp", ".home")


class UrlBlocked(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


Resolver = Callable[[str], list[str]]


def default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return list({info[4][0] for info in infos})


def _is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved or addr.is_unspecified)


def check_url(url: str, resolver: Resolver = default_resolver) -> str:
    """Return the URL if it is safe to fetch, else raise UrlBlocked."""
    try:
        parts = urlsplit((url or "").strip())
        port = parts.port
    except ValueError:
        raise UrlBlocked("malformed_url")
    if parts.scheme not in ALLOWED_SCHEMES:
        raise UrlBlocked("scheme_not_allowed")
    if parts.username or parts.password:
        raise UrlBlocked("credentials_in_url")
    if port not in ALLOWED_PORTS:
        raise UrlBlocked("port_not_allowed")
    host = (parts.hostname or "").lower()
    if not host or "." not in host and not _is_ip(host):
        raise UrlBlocked("host_not_allowed")
    if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
        raise UrlBlocked("internal_hostname")
    if _is_ip(host):
        if not _is_public(host):
            raise UrlBlocked("non_public_address")
        return url
    try:
        addresses = resolver(host)
    except Exception:
        raise UrlBlocked("dns_resolution_failed")
    if not addresses:
        raise UrlBlocked("dns_resolution_failed")
    if not all(_is_public(a) for a in addresses):
        raise UrlBlocked("non_public_address")
    return url


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False
